"""H32-RLIF: robust recurrent latent interaction over a frozen B1 parent."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from train_pdlf_clip import LABELS, PDLFClip


H32_TRAINABLE_PREFIXES = (
    "latent_tokens",
    "text_projection.",
    "image_projection.",
    "text_type_embedding",
    "image_type_embedding",
    "latent_block.",
    "residual_classifier.",
)


def build_b1_parent(model_name: str) -> PDLFClip:
    return PDLFClip(
        model_name=model_name,
        num_labels=len(LABELS),
        rank_dim=128,
        hidden_dim=512,
        fusion_variant="residual_add",
        freeze_clip=True,
        prototype_temperature=0.07,
        use_lightweight_cross_attention=True,
        lca_dim=128,
        lca_heads=4,
        lca_dropout=0.1,
        lca_residual_scale=0.35,
        clean_lca_residual=True,
        fixed_residual_scale=0.5,
        use_global_cross_modal_calibration=False,
        freeze_core_model=False,
    )


class RecurrentLatentBlock(nn.Module):
    """One tied latent/text/image interaction block reused across rounds.

    Ablation flags bypass complete residual sublayers, including their norms and
    trainable parameters, so a removed component cannot contribute indirectly.
    """

    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        ffn_dim: int = 256,
        use_text_cross_attention: bool = True,
        use_image_cross_attention: bool = True,
        use_self_attention: bool = True,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.use_text_cross_attention = bool(use_text_cross_attention)
        self.use_image_cross_attention = bool(use_image_cross_attention)
        self.use_self_attention = bool(use_self_attention)
        self.use_ffn = bool(use_ffn)
        if self.use_text_cross_attention:
            self.latent_text_norm = nn.LayerNorm(dim)
            self.text_norm = nn.LayerNorm(dim)
            self.text_cross_attention = nn.MultiheadAttention(
                dim, heads, dropout=0.1, batch_first=True
            )
        if self.use_image_cross_attention:
            self.latent_image_norm = nn.LayerNorm(dim)
            self.image_norm = nn.LayerNorm(dim)
            self.image_cross_attention = nn.MultiheadAttention(
                dim, heads, dropout=0.1, batch_first=True
            )
        if self.use_self_attention:
            self.latent_self_norm = nn.LayerNorm(dim)
            self.self_attention = nn.MultiheadAttention(
                dim, heads, dropout=0.1, batch_first=True
            )
        if self.use_ffn:
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(ffn_dim, dim)
            )

    def forward(
        self,
        latents: torch.Tensor,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_text_cross_attention:
            query = self.latent_text_norm(latents)
            text = self.text_norm(text_tokens)
            update, _ = self.text_cross_attention(
                query=query,
                key=text,
                value=text,
                key_padding_mask=text_padding_mask,
                need_weights=False,
            )
            latents = latents + update

        if self.use_image_cross_attention:
            query = self.latent_image_norm(latents)
            image = self.image_norm(image_patches)
            update, _ = self.image_cross_attention(
                query=query, key=image, value=image, need_weights=False
            )
            latents = latents + update

        if self.use_self_attention:
            query = self.latent_self_norm(latents)
            update, _ = self.self_attention(query=query, key=query, value=query, need_weights=False)
            latents = latents + update
        if self.use_ffn:
            latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class H32RLIF(nn.Module):
    def __init__(
        self,
        parent: PDLFClip,
        latent_count: int = 8,
        latent_dim: int = 128,
        heads: int = 4,
        rounds: int = 2,
        ffn_dim: int = 256,
        residual_hidden_dim: int = 128,
        residual_scale: float = 0.25,
        use_text_cross_attention: bool = True,
        use_image_cross_attention: bool = True,
        use_self_attention: bool = True,
        use_ffn: bool = True,
    ):
        super().__init__()
        if parent.lightweight_cross_attention is None or not parent.clean_lca_residual:
            raise ValueError("H32 requires a clean-LCA B1 parent")
        if parent.use_global_cross_modal_calibration or parent.adaptive_residual_gate:
            raise ValueError("H32 parent must be B1-Core")
        if latent_count < 1 or rounds < 1 or latent_dim % heads != 0:
            raise ValueError("Invalid H32 latent configuration")
        self.parent = parent
        self.rounds = int(rounds)
        self.residual_scale = float(residual_scale)
        text_dim = int(parent.clip.config.text_config.hidden_size)
        image_dim = int(parent.clip.config.vision_config.hidden_size)
        self.latent_tokens = nn.Parameter(torch.randn(latent_count, latent_dim) * 0.02)
        self.text_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, latent_dim))
        self.image_projection = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, latent_dim))
        self.text_type_embedding = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.image_type_embedding = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.latent_block = RecurrentLatentBlock(
            latent_dim,
            heads,
            ffn_dim,
            use_text_cross_attention=use_text_cross_attention,
            use_image_cross_attention=use_image_cross_attention,
            use_self_attention=use_self_attention,
            use_ffn=use_ffn,
        )
        self.residual_classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(residual_hidden_dim, len(LABELS)),
        )
        nn.init.zeros_(self.residual_classifier[-1].weight)
        nn.init.zeros_(self.residual_classifier[-1].bias)
        self.freeze_parent()

    def freeze_parent(self) -> None:
        for parameter in self.parent.parameters():
            parameter.requires_grad = False
        self.parent.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def _clip_outputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            text_outputs = self.parent.clip.text_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            image_outputs = self.parent.clip.vision_model(pixel_values=batch["pixel_values"])
            raw_text = F.normalize(
                self.parent.clip.text_projection(text_outputs.pooler_output), dim=-1
            )
            raw_image = F.normalize(
                self.parent.clip.visual_projection(image_outputs.pooler_output), dim=-1
            )
            return {
                "raw_text": raw_text,
                "raw_image": raw_image,
                "text_tokens": text_outputs.last_hidden_state,
                "image_patches": image_outputs.last_hidden_state[:, 1:, :],
            }

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        clip = self._clip_outputs(batch)
        with torch.no_grad():
            fine = self.parent.lightweight_cross_attention(
                clip["text_tokens"], clip["image_patches"], batch["attention_mask"]
            )
            fine_text = F.normalize(fine["text"], dim=-1)
            fine_image = F.normalize(fine["image"], dim=-1)
            base_logits = self.parent.base_classifier(
                torch.cat([clip["raw_text"], clip["raw_image"]], dim=-1)
            )
            local_logits = self.parent.aux_classifier(
                torch.cat([fine_text, fine_image], dim=-1)
            )
            b1_logits = base_logits + self.parent.fixed_residual_scale * local_logits

        text_tokens = self.text_projection(clip["text_tokens"]) + self.text_type_embedding
        image_patches = self.image_projection(clip["image_patches"]) + self.image_type_embedding
        latents = self.latent_tokens.unsqueeze(0).expand(text_tokens.shape[0], -1, -1)
        text_padding = batch["attention_mask"] == 0
        for _ in range(self.rounds):
            latents = self.latent_block(latents, text_tokens, image_patches, text_padding)
        pooled_latent = latents.mean(dim=1)
        residual_logits = self.residual_classifier(pooled_latent)
        class_logits = b1_logits + self.residual_scale * residual_logits
        return {
            "class_logits": class_logits,
            "b1_logits": b1_logits,
            "rlif_residual_logits": residual_logits,
            "pooled_latent": pooled_latent,
        }


def trainable_parameter_names(model: H32RLIF) -> list[str]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [name for name in names if not name.startswith(H32_TRAINABLE_PREFIXES)]
    if invalid or not names:
        raise RuntimeError(f"H32 trainable whitelist failed: {invalid}")
    return names


def parameter_counts(model: H32RLIF) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": trainable,
        "rlif_only_parameters": trainable,
        "b1_attention_parameters": sum(
            p.numel() for p in model.parent.lightweight_cross_attention.parameters()
        ),
        "rlif_attention_parameters": sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("latent_block.") and "attention" in name
        ),
    }
