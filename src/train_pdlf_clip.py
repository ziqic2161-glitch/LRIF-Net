import argparse
import html
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup


LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


@dataclass
class TrainConfig:
    data_dir: str = r"D:\CrisisMMD_dataset\processed"
    image_root: str = ""
    output_dir: str = r"D:\CrisisMMD_dataset\runs\pdlf_clip"
    model_name: str = "openai/clip-vit-base-patch32"
    epochs: int = 10
    batch_size: int = 16
    rank_dim: int = 128
    hidden_dim: int = 512
    fusion_variant: str = "replace"  # replace or residual_add
    use_lightweight_cross_attention: bool = False
    lca_dim: int = 128
    lca_heads: int = 4
    lca_dropout: float = 0.1
    lca_residual_scale: float = 0.35
    residual_logit_init: float = -3.0
    clean_lca_residual: bool = False
    standard_cross_attention_comparator: bool = False
    fixed_residual_scale: float = 0.5
    residual_warmup_epochs: int = 0
    adaptive_residual_gate: bool = False
    residual_gate_hidden_dim: int = 32
    use_global_cross_modal_calibration: bool = False
    global_calibration_rank: int = 128
    freeze_core_model: bool = False
    prototype_loss_weight: float = 0.1
    prototype_temperature: float = 0.07
    disable_prototypes: bool = False
    disable_gate: bool = False
    disable_low_rank: bool = False
    lr_head: float = 8e-4
    lr_clip: float = 1e-5
    weight_decay: float = 1e-4
    max_length: int = 77
    num_workers: int = 0
    seed: int = 42
    freeze_clip: bool = True
    unfreeze_last_n_layers: int = 0
    init_checkpoint: str = ""
    init_baseline_checkpoint: str = ""
    freeze_base_classifier: bool = False
    use_class_weights: bool = True
    class_weight_power: float = 1.0
    amp: bool = True
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    label_smoothing: float = 0.0
    warmup_ratio: float = 0.1
    use_cosine_schedule: bool = False
    image_augmentation: bool = False
    text_preprocessing: bool = False
    use_reliability_gate: bool = False
    auxiliary_loss_weight: float = 0.0
    select_metric: str = "weighted_f1"
    early_stop_patience: int = 3
    eval_only: bool = False
    development_only: bool = False
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def residual_scale_for_epoch(target_scale: float, warmup_epochs: int, epoch: int) -> float:
    if not 0.0 <= target_scale <= 1.0:
        raise ValueError("target residual scale must be in [0, 1]")
    if warmup_epochs < 0:
        raise ValueError("warmup epochs must be non-negative")
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if warmup_epochs == 0:
        return float(target_scale)
    return float(target_scale) * min(epoch / warmup_epochs, 1.0)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    required = {"tweet_text", "image_path", "label_5class", "image_exists"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[df["image_exists"].astype(str).str.lower() == "true"].copy()
    df = df[df["label_5class"].isin(LABEL_TO_ID)].copy()
    df["label_id"] = df["label_5class"].map(LABEL_TO_ID).astype(int)
    return df.reset_index(drop=True)


class TrainingImageAugment:
    """Mild augmentation that preserves maps, posters, and embedded text."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(0.90, 1.0)
        crop_width = max(1, int(width * scale))
        crop_height = max(1, int(height * scale))
        left = random.randint(0, max(width - crop_width, 0))
        top = random.randint(0, max(height - crop_height, 0))
        image = image.crop((left, top, left + crop_width, top + crop_height))
        if random.random() < 0.7:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.7:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.90, 1.10))
        return image


class CrisisMmdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_augment=None, image_root: str = ""):
        self.df = df.reset_index(drop=True)
        self.image_augment = image_augment
        self.image_root = Path(image_root) if image_root else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_path = Path(str(row["image_path"]))
        if not image_path.exists() and self.image_root is not None and "image_rel_path" in row:
            image_path = self.image_root / Path(str(row["image_rel_path"]))
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image.thumbnail((512, 512), Image.Resampling.BICUBIC)
                image = image.copy()
                if self.image_augment is not None:
                    image = self.image_augment(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc
        return {
            "text": str(row["tweet_text"]),
            "image": image,
            "label": int(row["label_id"]),
            "sample_id": str(row.get("sample_id", idx)),
        }


def preprocess_tweet_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = re.sub(r"https?://\S+|www\.\S+", " link ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@\w+", " user ", text)
    text = re.sub(r"(?<!\w)#(\w+)", r" \1 ", text)
    text = re.sub(r"^\s*RT\s+", "", text, flags=re.IGNORECASE)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


class ClipCollator:
    def __init__(self, processor: CLIPProcessor, max_length: int, preprocess_text: bool = False):
        self.processor = processor
        self.max_length = max_length
        self.preprocess_text = preprocess_text

    def __call__(self, batch: List[Dict]) -> Dict:
        texts = [item["text"] for item in batch]
        if self.preprocess_text:
            texts = [preprocess_tweet_text(text) for text in texts]
        enc = self.processor(
            text=texts,
            images=[item["image"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        enc["sample_ids"] = [item["sample_id"] for item in batch]
        return enc


class LightweightCrossAttention(nn.Module):
    """Single-layer bidirectional token/patch interaction in a small hidden space."""

    def __init__(
        self,
        text_hidden_dim: int,
        vision_hidden_dim: int,
        output_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("--lca-dim must be divisible by --lca-heads.")
        self.text_in = nn.Sequential(nn.LayerNorm(text_hidden_dim), nn.Linear(text_hidden_dim, attention_dim))
        self.image_in = nn.Sequential(nn.LayerNorm(vision_hidden_dim), nn.Linear(vision_hidden_dim, attention_dim))
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_out = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, output_dim),
            nn.GELU(),
        )
        self.image_out = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, output_dim),
            nn.GELU(),
        )

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=1)
        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_mask: torch.Tensor | None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        text_small = self.text_in(text_tokens)
        image_small = self.image_in(image_patches)
        text_key_padding = text_mask == 0 if text_mask is not None else None

        text_context, text_to_image_weights = self.text_to_image(
            query=text_small,
            key=image_small,
            value=image_small,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        image_context, image_to_text_weights = self.image_to_text(
            query=image_small,
            key=text_small,
            value=text_small,
            key_padding_mask=text_key_padding,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        text_update = self.text_out(self.masked_mean(text_context, text_mask))
        image_update = self.image_out(image_context.mean(dim=1))
        output = {"text": text_update, "image": image_update}
        if return_attention:
            output["text_to_image_attention"] = text_to_image_weights
            output["image_to_text_attention"] = image_to_text_weights
        return output


class GlobalCrossModalCalibration(nn.Module):
    """Rank-constrained bidirectional calibration of global CLIP embeddings."""

    def __init__(self, feature_dim: int, rank_dim: int):
        super().__init__()
        if rank_dim < 1:
            raise ValueError("global calibration rank must be positive")
        self.rank_dim = rank_dim
        self.text_value = nn.Linear(feature_dim, rank_dim, bias=False)
        self.image_value = nn.Linear(feature_dim, rank_dim, bias=False)
        self.text_gate_from_image = nn.Linear(feature_dim, rank_dim)
        self.image_gate_from_text = nn.Linear(feature_dim, rank_dim)
        self.text_update = nn.Linear(rank_dim, feature_dim, bias=False)
        self.image_update = nn.Linear(rank_dim, feature_dim, bias=False)
        self.update_scale = rank_dim ** -0.5

    def forward(self, text_global: torch.Tensor, image_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_gate = torch.sigmoid(self.text_gate_from_image(image_global))
        image_gate = torch.sigmoid(self.image_gate_from_text(text_global))
        calibrated_text = text_global + self.update_scale * self.text_update(
            text_gate * self.text_value(text_global)
        )
        calibrated_image = image_global + self.update_scale * self.image_update(
            image_gate * self.image_value(image_global)
        )
        return {
            "text": nn.functional.normalize(calibrated_text, dim=-1),
            "image": nn.functional.normalize(calibrated_image, dim=-1),
            "text_gate": text_gate,
            "image_gate": image_gate,
        }


class PDLFClip(nn.Module):
    """Baseline-preserving CLIP fusion model with legacy PDLF and prospective clean-LCA modes."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        rank_dim: int,
        hidden_dim: int,
        fusion_variant: str,
        freeze_clip: bool,
        prototype_temperature: float,
        unfreeze_last_n_layers: int = 0,
        use_lightweight_cross_attention: bool = False,
        lca_dim: int = 128,
        lca_heads: int = 4,
        lca_dropout: float = 0.1,
        lca_residual_scale: float = 0.35,
        residual_logit_init: float = -3.0,
        clean_lca_residual: bool = False,
        fixed_residual_scale: float = 0.5,
        adaptive_residual_gate: bool = False,
        residual_gate_hidden_dim: int = 32,
        use_global_cross_modal_calibration: bool = False,
        global_calibration_rank: int = 128,
        freeze_core_model: bool = False,
        disable_prototypes: bool = False,
        disable_gate: bool = False,
        disable_low_rank: bool = False,
        use_reliability_gate: bool = False,
    ):
        super().__init__()
        if fusion_variant not in {"replace", "residual_add"}:
            raise ValueError(f"Unsupported fusion_variant: {fusion_variant}")
        self.fusion_variant = fusion_variant
        self.clip = CLIPModel.from_pretrained(model_name)
        if unfreeze_last_n_layers < 0:
            raise ValueError("unfreeze_last_n_layers must be non-negative")
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        elif unfreeze_last_n_layers > 0:
            self._unfreeze_last_clip_layers(unfreeze_last_n_layers)
        self.freeze_clip = freeze_clip
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.prototype_temperature = prototype_temperature
        self.clean_lca_residual = clean_lca_residual
        self.adaptive_residual_gate = adaptive_residual_gate
        self.use_global_cross_modal_calibration = use_global_cross_modal_calibration
        self.freeze_core_model = freeze_core_model
        if clean_lca_residual and not use_lightweight_cross_attention:
            raise ValueError("clean_lca_residual requires lightweight cross-attention")
        if adaptive_residual_gate and not clean_lca_residual:
            raise ValueError("adaptive_residual_gate requires clean_lca_residual")
        if use_global_cross_modal_calibration and not clean_lca_residual:
            raise ValueError("global cross-modal calibration requires clean_lca_residual")
        if freeze_core_model and not use_global_cross_modal_calibration:
            raise ValueError("freeze_core_model requires global cross-modal calibration")
        if not 0.0 <= fixed_residual_scale <= 1.0:
            raise ValueError("fixed_residual_scale must be in [0, 1]")
        if residual_gate_hidden_dim < 1:
            raise ValueError("residual_gate_hidden_dim must be positive")
        self.fixed_residual_scale = float(fixed_residual_scale)
        self.disable_prototypes = disable_prototypes
        self.disable_gate = disable_gate
        self.disable_low_rank = disable_low_rank
        self.use_lightweight_cross_attention = use_lightweight_cross_attention
        self.lca_residual_scale = lca_residual_scale
        self.use_reliability_gate = use_reliability_gate

        d = self.clip.config.projection_dim
        if use_lightweight_cross_attention:
            self.lightweight_cross_attention = LightweightCrossAttention(
                text_hidden_dim=self.clip.config.text_config.hidden_size,
                vision_hidden_dim=self.clip.config.vision_config.hidden_size,
                output_dim=d,
                attention_dim=lca_dim,
                num_heads=lca_heads,
                dropout=lca_dropout,
            )
        else:
            self.lightweight_cross_attention = None
        if not clean_lca_residual:
            self.visual_low_rank = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, rank_dim), nn.GELU())
            self.text_low_rank = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, rank_dim), nn.GELU())
            self.low_rank_projector = nn.Sequential(
                nn.LayerNorm(rank_dim),
                nn.Linear(rank_dim, d),
                nn.GELU(),
            )
            self.prototype_bank = nn.Parameter(torch.randn(num_labels, d) * 0.02)
        gate_in_dim = d * 4
        if clean_lca_residual:
            pass
        elif use_reliability_gate:
            self.image_aux_classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_labels))
            self.text_aux_classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_labels))
            self.reliability_gate = nn.Sequential(
                nn.LayerNorm(gate_in_dim + 2),
                nn.Linear(gate_in_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )
            self.reliability_logit_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in_dim),
                nn.Linear(gate_in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, d),
                nn.Sigmoid(),
            )
        if clean_lca_residual:
            self.base_classifier = nn.Sequential(
                nn.LayerNorm(d * 2),
                nn.Dropout(0.2),
                nn.Linear(d * 2, num_labels),
            )
            self.aux_classifier = nn.Sequential(
                nn.LayerNorm(d * 2),
                nn.Linear(d * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_labels),
            )
            if use_global_cross_modal_calibration:
                self.global_cross_modal_calibration = GlobalCrossModalCalibration(
                    feature_dim=d,
                    rank_dim=global_calibration_rank,
                )
                self.global_residual_classifier = nn.Sequential(
                    nn.LayerNorm(d * 2),
                    nn.Linear(d * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, num_labels),
                )
                # Exact B1 inheritance: the new H1 branch contributes zero logits
                # until optimization makes the global calibration useful.
                nn.init.zeros_(self.global_residual_classifier[-1].weight)
                nn.init.zeros_(self.global_residual_classifier[-1].bias)
            if adaptive_residual_gate:
                self.residual_gate = nn.Sequential(
                    nn.LayerNorm(num_labels * 2),
                    nn.Linear(num_labels * 2, residual_gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(residual_gate_hidden_dim, 1),
                )
                initial_gate = min(max(self.fixed_residual_scale, 1e-6), 1.0 - 1e-6)
                nn.init.zeros_(self.residual_gate[-1].weight)
                nn.init.constant_(self.residual_gate[-1].bias, math.log(initial_gate / (1.0 - initial_gate)))
        elif fusion_variant == "residual_add":
            self.base_classifier = nn.Sequential(
                nn.LayerNorm(d * 2),
                nn.Dropout(0.2),
                nn.Linear(d * 2, num_labels),
            )
            self.aux_classifier = nn.Sequential(
                nn.LayerNorm(d * 5),
                nn.Linear(d * 5, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_labels),
            )
            # Start close to the strong raw-CLIP baseline and let fusion earn a larger role.
            self.residual_logit_scale = nn.Parameter(torch.tensor(float(residual_logit_init)))
        else:
            cls_in_dim = d * 4
            self.classifier = nn.Sequential(
                nn.LayerNorm(cls_in_dim),
                nn.Linear(cls_in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(hidden_dim, num_labels),
            )

    def set_fixed_residual_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("fixed residual scale must be in [0, 1]")
        self.fixed_residual_scale = float(value)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.clean_lca_residual:
            # Preserve the exact B0 decision path during B1 optimization: its
            # parameters are frozen by the runner and its dropout stays disabled.
            self.base_classifier.eval()
        if self.freeze_core_model:
            self.clip.eval()
            self.lightweight_cross_attention.eval()
            self.aux_classifier.eval()
        return self

    def freeze_b1_core(self) -> None:
        """Freeze every inherited B1-Core parameter while leaving H1 trainable."""
        if not self.use_global_cross_modal_calibration:
            raise ValueError("B1-Core freezing is only defined for the H1 model")
        for module in (self.clip, self.base_classifier, self.lightweight_cross_attention, self.aux_classifier):
            for param in module.parameters():
                param.requires_grad = False
        self.freeze_core_model = True

    def _unfreeze_last_clip_layers(self, layer_count: int) -> None:
        """Fine-tune only the final CLIP blocks and their output projections."""
        for param in self.clip.parameters():
            param.requires_grad = False

        encoder_groups = (
            self.clip.text_model.encoder.layers,
            self.clip.vision_model.encoder.layers,
        )
        for layers in encoder_groups:
            if layer_count > len(layers):
                raise ValueError(
                    f"Cannot unfreeze {layer_count} layers from an encoder with {len(layers)} layers"
                )
            for layer in layers[-layer_count:]:
                for param in layer.parameters():
                    param.requires_grad = True

        output_modules = (
            self.clip.text_model.final_layer_norm,
            self.clip.vision_model.post_layernorm,
            self.clip.text_projection,
            self.clip.visual_projection,
        )
        for module in output_modules:
            for param in module.parameters():
                param.requires_grad = True

    def encode(self, batch: Dict, return_attention: bool = False) -> Dict:
        def run_clip() -> Dict[str, torch.Tensor]:
            text_outputs = self.clip.text_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            image_outputs = self.clip.vision_model(pixel_values=batch["pixel_values"])
            text_features = self.clip.text_projection(text_outputs.pooler_output)
            image_features = self.clip.visual_projection(image_outputs.pooler_output)
            return {
                "text": text_features,
                "image": image_features,
                "text_tokens": text_outputs.last_hidden_state,
                "image_tokens": image_outputs.last_hidden_state,
            }

        if self.freeze_clip:
            with torch.no_grad():
                clip_outputs = run_clip()
        else:
            clip_outputs = run_clip()

        raw_text_features = clip_outputs["text"]
        raw_image_features = clip_outputs["image"]
        text_features = raw_text_features
        image_features = raw_image_features
        if self.lightweight_cross_attention is not None:
            lca = self.lightweight_cross_attention(
                text_tokens=clip_outputs["text_tokens"],
                image_patches=clip_outputs["image_tokens"][:, 1:, :],
                text_mask=batch["attention_mask"],
                return_attention=return_attention,
            )
            text_features = text_features + self.lca_residual_scale * lca["text"]
            image_features = image_features + self.lca_residual_scale * lca["image"]
        output = {
            "text": nn.functional.normalize(text_features, dim=-1),
            "image": nn.functional.normalize(image_features, dim=-1),
            "raw_text": nn.functional.normalize(raw_text_features, dim=-1),
            "raw_image": nn.functional.normalize(raw_image_features, dim=-1),
        }
        if return_attention and self.lightweight_cross_attention is not None:
            output["text_to_image_attention"] = lca["text_to_image_attention"]
            output["image_to_text_attention"] = lca["image_to_text_attention"]
        if self.lightweight_cross_attention is not None:
            output["lca_text"] = nn.functional.normalize(lca["text"], dim=-1)
            output["lca_image"] = nn.functional.normalize(lca["image"], dim=-1)
        return output

    def forward(self, batch: Dict, return_attention: bool = False) -> Dict:
        enc = self.encode(batch, return_attention=return_attention)
        text_features = enc["text"]
        image_features = enc["image"]
        raw_text_features = enc["raw_text"]
        raw_image_features = enc["raw_image"]

        if self.clean_lca_residual:
            base_features = torch.cat([raw_text_features, raw_image_features], dim=-1)
            lca_features = torch.cat([enc["lca_text"], enc["lca_image"]], dim=-1)
            base_logits = self.base_classifier(base_features)
            local_residual_logits = self.aux_classifier(lca_features)
            global_residual_logits = torch.zeros_like(local_residual_logits)
            global_gate_mean = local_residual_logits.new_tensor(math.nan)
            if self.use_global_cross_modal_calibration:
                global_context = self.global_cross_modal_calibration(
                    raw_text_features,
                    raw_image_features,
                )
                global_features = torch.cat([global_context["text"], global_context["image"]], dim=-1)
                global_residual_logits = self.global_residual_classifier(global_features)
                global_gate_mean = torch.cat(
                    [global_context["text_gate"], global_context["image_gate"]], dim=-1
                ).mean().detach()
            residual_logits = local_residual_logits + global_residual_logits
            if self.adaptive_residual_gate:
                gate_input = torch.cat([base_logits.detach(), residual_logits.detach()], dim=-1)
                residual_gate = torch.sigmoid(self.residual_gate(gate_input)).squeeze(-1)
            else:
                residual_gate = residual_logits.new_full((residual_logits.shape[0],), self.fixed_residual_scale)
            class_logits = base_logits + residual_gate.unsqueeze(-1) * residual_logits
            batch_size = class_logits.shape[0]
            proto_logits = class_logits.new_zeros((batch_size, class_logits.shape[-1]))
            output = {
                "class_logits": class_logits,
                "proto_logits": proto_logits,
                "prototype_active": False,
                "sample_text_gate": class_logits.new_full((batch_size,), 0.5),
                "sample_image_gate": class_logits.new_full((batch_size,), 0.5),
                "text_gate_mean": class_logits.new_tensor(0.5),
                "image_gate_mean": class_logits.new_tensor(0.5),
                "sample_residual_gate": residual_gate.detach(),
                "residual_gate_mean": residual_gate.mean().detach(),
                "sample_global_residual_norm": global_residual_logits.norm(dim=-1).detach(),
                "global_calibration_gate_mean": global_gate_mean,
            }
            if return_attention:
                output["text_to_image_attention"] = enc["text_to_image_attention"]
                output["image_to_text_attention"] = enc["image_to_text_attention"]
            return output

        if self.disable_low_rank:
            low_rank_features = torch.zeros_like(image_features)
        else:
            low_rank_interaction = self.visual_low_rank(image_features) * self.text_low_rank(text_features)
            low_rank_features = nn.functional.normalize(self.low_rank_projector(low_rank_interaction), dim=-1)

        if self.disable_prototypes:
            proto_logits = low_rank_features.new_zeros((low_rank_features.shape[0], self.prototype_bank.shape[0]))
            proto_features = torch.zeros_like(low_rank_features)
        else:
            prototypes = nn.functional.normalize(self.prototype_bank, dim=-1)
            proto_logits = low_rank_features @ prototypes.t() / self.prototype_temperature
            proto_weights = torch.softmax(proto_logits, dim=-1)
            proto_features = proto_weights @ prototypes
            proto_features = nn.functional.normalize(proto_features, dim=-1)

        gate_input = torch.cat([image_features, text_features, low_rank_features, proto_features], dim=-1)
        image_aux_logits = None
        text_aux_logits = None
        image_reliability = None
        text_reliability = None
        if self.disable_gate:
            text_gate = torch.full_like(text_features, 0.5)
        elif self.use_reliability_gate:
            image_aux_logits = self.image_aux_classifier(image_features)
            text_aux_logits = self.text_aux_classifier(text_features)
            image_probs = torch.softmax(image_aux_logits, dim=-1)
            text_probs = torch.softmax(text_aux_logits, dim=-1)
            normalizer = math.log(image_probs.shape[-1])
            image_reliability = 1.0 + (image_probs * image_probs.clamp_min(1e-12).log()).sum(dim=-1) / normalizer
            text_reliability = 1.0 + (text_probs * text_probs.clamp_min(1e-12).log()).sum(dim=-1) / normalizer
            reliability_input = torch.cat(
                [gate_input, image_reliability.unsqueeze(-1), text_reliability.unsqueeze(-1)],
                dim=-1,
            )
            gate_logits = self.reliability_gate(reliability_input).squeeze(-1)
            gate_logits = gate_logits + self.reliability_logit_scale * (text_reliability - image_reliability)
            scalar_text_gate = torch.sigmoid(gate_logits)
            text_gate = scalar_text_gate.unsqueeze(-1).expand_as(text_features)
        else:
            text_gate = self.gate(gate_input)
        gated_features = (1.0 - text_gate) * image_features + text_gate * text_features
        fused = nn.functional.normalize(gated_features + low_rank_features + proto_features, dim=-1)

        abs_diff = torch.abs(text_features - image_features)
        product = text_features * image_features
        if self.fusion_variant == "residual_add":
            # Match ClipClassifier's [text, image] ordering for exact warm-start reuse.
            base_features = torch.cat([raw_text_features, raw_image_features], dim=-1)
            aux_features = torch.cat(
                [gated_features, abs_diff, product, low_rank_features, proto_features],
                dim=-1,
            )
            residual_scale = torch.sigmoid(self.residual_logit_scale)
            class_logits = self.base_classifier(base_features) + residual_scale * self.aux_classifier(aux_features)
        else:
            cls_features = torch.cat(
                [
                    fused,
                    abs_diff,
                    low_rank_features,
                    proto_features,
                ],
                dim=-1,
            )
            class_logits = self.classifier(cls_features)
        output = {
            "class_logits": class_logits,
            "proto_logits": proto_logits,
            "sample_text_gate": text_gate.mean(dim=-1).detach(),
            "sample_image_gate": (1.0 - text_gate).mean(dim=-1).detach(),
            "text_gate_mean": text_gate.mean().detach(),
            "image_gate_mean": (1.0 - text_gate).mean().detach(),
        }
        if self.use_reliability_gate:
            output["image_aux_logits"] = image_aux_logits
            output["text_aux_logits"] = text_aux_logits
            output["image_reliability"] = image_reliability.detach()
            output["text_reliability"] = text_reliability.detach()
        if return_attention and self.lightweight_cross_attention is not None:
            output["text_to_image_attention"] = enc["text_to_image_attention"]
            output["image_to_text_attention"] = enc["image_to_text_attention"]
        return output


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_class_weights(train_df: pd.DataFrame, device: torch.device, power: float = 1.0) -> torch.Tensor:
    counts = train_df["label_id"].value_counts().reindex(range(len(LABELS)), fill_value=0).values
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(LABELS) * counts)
    weights = np.power(weights, power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_matrix(preds: List[int], labels: List[int], num_labels: int) -> np.ndarray:
    mat = np.zeros((num_labels, num_labels), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        mat[y_true, y_pred] += 1
    return mat


def metrics_from_confusion(mat: np.ndarray) -> Dict:
    total = mat.sum()
    accuracy = np.trace(mat) / max(total, 1)
    per_class = {}
    f1_values = []
    supports = []
    for i, label in enumerate(LABELS):
        tp = mat[i, i]
        fp = mat[:, i].sum() - tp
        fn = mat[i, :].sum() - tp
        support = mat[i, :].sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        f1_values.append(f1)
        supports.append(support)
    supports = np.asarray(supports)
    f1_values = np.asarray(f1_values)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1_values.mean()),
        "weighted_f1": float((f1_values * supports).sum() / max(supports.sum(), 1)),
        "per_class": per_class,
    }


def compute_loss(
    outputs: Dict,
    labels: torch.Tensor,
    class_criterion: nn.Module,
    proto_weight: float,
    auxiliary_weight: float = 0.0,
) -> Dict:
    class_loss = class_criterion(outputs["class_logits"], labels)
    if outputs.get("prototype_active", True):
        proto_loss = nn.functional.cross_entropy(outputs["proto_logits"], labels)
    else:
        proto_loss = class_loss.new_zeros(())
    auxiliary_loss = class_loss.new_zeros(())
    if "image_aux_logits" in outputs and auxiliary_weight > 0:
        auxiliary_loss = 0.5 * (
            class_criterion(outputs["image_aux_logits"], labels)
            + class_criterion(outputs["text_aux_logits"], labels)
        )
    total_loss = class_loss + proto_weight * proto_loss + auxiliary_weight * auxiliary_loss
    return {
        "loss": total_loss,
        "class_loss": class_loss,
        "proto_loss": proto_loss,
        "auxiliary_loss": auxiliary_loss,
    }


@torch.no_grad()
def initialize_from_clip_baseline(model: PDLFClip, baseline_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Load baseline weights that are structurally compatible with a fusion variant."""
    model_state = model.state_dict()
    clip_updates = {}
    for key, value in baseline_state.items():
        if not key.startswith("clip."):
            continue
        if key not in model_state:
            raise KeyError(f"Model is missing baseline CLIP key: {key}")
        if value.shape != model_state[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(value.shape)} != {tuple(model_state[key].shape)}"
            )
        clip_updates[key] = value
    if not clip_updates:
        raise KeyError("Baseline checkpoint contains no CLIP encoder weights.")
    model_state.update(clip_updates)
    model.load_state_dict(model_state)

    classifier_tensors = 0
    if model.fusion_variant == "residual_add" or model.clean_lca_residual:
        base_state = model.base_classifier.state_dict()
        for key in list(base_state):
            source_key = f"classifier.{key}"
            if source_key not in baseline_state:
                raise KeyError(f"Baseline checkpoint is missing {source_key}")
            if baseline_state[source_key].shape != base_state[key].shape:
                raise ValueError(
                    f"Shape mismatch for {source_key}: "
                    f"{tuple(baseline_state[source_key].shape)} != {tuple(base_state[key].shape)}"
                )
            base_state[key] = baseline_state[source_key]
            classifier_tensors += 1
        model.base_classifier.load_state_dict(base_state)
    return {"clip_tensors": len(clip_updates), "classifier_tensors": classifier_tensors}


@torch.no_grad()
def initialize_from_parent_checkpoint(
    model: PDLFClip,
    parent_state: Dict[str, torch.Tensor],
    allowed_new_prefixes: tuple[str, ...] = (),
) -> Dict[str, List[str]]:
    """Load a parent model strictly except for explicitly declared new modules."""
    incompatible = model.load_state_dict(parent_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [key for key in missing if not key.startswith(allowed_new_prefixes)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Parent checkpoint is not structurally compatible: "
            f"invalid missing={invalid_missing}, unexpected={unexpected}"
        )
    return {"missing_new_tensors": missing, "unexpected_tensors": unexpected}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    proto_weight: float,
    auxiliary_weight: float = 0.0,
) -> Dict:
    model.eval()
    losses = []
    class_losses = []
    proto_losses = []
    auxiliary_losses = []
    preds = []
    labels = []
    text_gates = []
    image_gates = []
    residual_gates = []
    global_residual_norms = []
    global_calibration_gates = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch)
        loss_items = compute_loss(outputs, batch["labels"], criterion, proto_weight, auxiliary_weight)
        losses.append(float(loss_items["loss"].item()))
        class_losses.append(float(loss_items["class_loss"].item()))
        proto_losses.append(float(loss_items["proto_loss"].item()))
        auxiliary_losses.append(float(loss_items["auxiliary_loss"].item()))
        preds.extend(outputs["class_logits"].argmax(dim=-1).detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())
        text_gates.append(float(outputs["text_gate_mean"].item()))
        image_gates.append(float(outputs["image_gate_mean"].item()))
        if "sample_residual_gate" in outputs:
            residual_gates.extend(outputs["sample_residual_gate"].detach().cpu().tolist())
        if "sample_global_residual_norm" in outputs:
            global_residual_norms.extend(outputs["sample_global_residual_norm"].detach().cpu().tolist())
        if "global_calibration_gate_mean" in outputs:
            gate_value = float(outputs["global_calibration_gate_mean"].item())
            if math.isfinite(gate_value):
                global_calibration_gates.append(gate_value)
    mat = confusion_matrix(preds, labels, len(LABELS))
    metrics = metrics_from_confusion(mat)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    metrics["class_loss"] = float(np.mean(class_losses)) if class_losses else math.nan
    metrics["proto_loss"] = float(np.mean(proto_losses)) if proto_losses else math.nan
    metrics["auxiliary_loss"] = float(np.mean(auxiliary_losses)) if auxiliary_losses else math.nan
    metrics["text_gate"] = float(np.mean(text_gates)) if text_gates else math.nan
    metrics["image_gate"] = float(np.mean(image_gates)) if image_gates else math.nan
    metrics["residual_gate_mean"] = float(np.mean(residual_gates)) if residual_gates else math.nan
    metrics["residual_gate_std"] = float(np.std(residual_gates)) if residual_gates else math.nan
    metrics["residual_gate_min"] = float(np.min(residual_gates)) if residual_gates else math.nan
    metrics["residual_gate_max"] = float(np.max(residual_gates)) if residual_gates else math.nan
    metrics["global_residual_norm_mean"] = (
        float(np.mean(global_residual_norms)) if global_residual_norms else math.nan
    )
    metrics["global_residual_norm_std"] = (
        float(np.std(global_residual_norms)) if global_residual_norms else math.nan
    )
    metrics["global_calibration_gate_mean"] = (
        float(np.mean(global_calibration_gates)) if global_calibration_gates else math.nan
    )
    metrics["confusion_matrix"] = mat.tolist()
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    proto_weight: float,
    grad_accumulation_steps: int = 1,
    scheduler=None,
    auxiliary_weight: float = 0.0,
) -> Dict:
    model.train()
    losses = []
    class_losses = []
    proto_losses = []
    auxiliary_losses = []
    if grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1.")
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch)
            loss_items = compute_loss(outputs, batch["labels"], criterion, proto_weight, auxiliary_weight)
            backward_loss = loss_items["loss"] / grad_accumulation_steps
        scaler.scale(backward_loss).backward()
        should_step = (step + 1) % grad_accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        losses.append(float(loss_items["loss"].item()))
        class_losses.append(float(loss_items["class_loss"].item()))
        proto_losses.append(float(loss_items["proto_loss"].item()))
        auxiliary_losses.append(float(loss_items["auxiliary_loss"].item()))
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "class_loss": float(np.mean(class_losses)) if class_losses else math.nan,
        "proto_loss": float(np.mean(proto_losses)) if proto_losses else math.nan,
        "auxiliary_loss": float(np.mean(auxiliary_losses)) if auxiliary_losses else math.nan,
    }


def build_optimizer(model: PDLFClip, cfg: TrainConfig) -> torch.optim.Optimizer:
    head_params = []
    clip_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("clip."):
            clip_params.append(param)
        else:
            head_params.append(param)
    groups = [{"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay}]
    if clip_params:
        groups.append({"params": clip_params, "lr": cfg.lr_clip, "weight_decay": cfg.weight_decay})
    return torch.optim.AdamW(groups)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train PDLF-CLIP on the clean CrisisMMD Task2 split.")
    parser.add_argument("--data-dir", default=TrainConfig.data_dir)
    parser.add_argument(
        "--image-root",
        default=TrainConfig.image_root,
        help="Optional portable root containing image_rel_path entries when CSV image_path is unavailable.",
    )
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--rank-dim", type=int, default=TrainConfig.rank_dim)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--fusion-variant", choices=["replace", "residual_add"], default=TrainConfig.fusion_variant)
    parser.add_argument("--use-lightweight-cross-attention", action="store_true")
    parser.add_argument("--lca-dim", type=int, default=TrainConfig.lca_dim)
    parser.add_argument("--lca-heads", type=int, default=TrainConfig.lca_heads)
    parser.add_argument("--lca-dropout", type=float, default=TrainConfig.lca_dropout)
    parser.add_argument("--lca-residual-scale", type=float, default=TrainConfig.lca_residual_scale)
    parser.add_argument(
        "--residual-logit-init",
        type=float,
        default=TrainConfig.residual_logit_init,
        help="Initial logit for the learned fusion residual scale; -3 is about 4.7%%, 0 is 50%%.",
    )
    parser.add_argument(
        "--clean-lca-residual",
        action="store_true",
        help=(
            "Prospective B1 mode: preserve the raw CLIP classifier and add only a fixed-scale "
            "bidirectional LCA residual; no low-rank, prototype, or modality-gate modules are instantiated."
        ),
    )
    parser.add_argument(
        "--standard-cross-attention-comparator",
        action="store_true",
        help=(
            "Prospective B3 mode: label the clean residual model as the locked 512-d, 8-head "
            "standard cross-attention comparator. This flag enforces the B3 architecture."
        ),
    )
    parser.add_argument(
        "--fixed-residual-scale",
        type=float,
        default=TrainConfig.fixed_residual_scale,
        help="Fixed B1 logit-residual coefficient in [0, 1].",
    )
    parser.add_argument(
        "--residual-warmup-epochs",
        type=int,
        default=TrainConfig.residual_warmup_epochs,
        help=(
            "Validation-only C1 schedule: linearly ramp the clean LCA residual from zero "
            "to --fixed-residual-scale over this many epochs. Zero disables the schedule."
        ),
    )
    parser.add_argument(
        "--adaptive-residual-gate",
        action="store_true",
        help="B2 mode: replace B1's fixed residual coefficient with a compact sample-adaptive scalar gate.",
    )
    parser.add_argument(
        "--residual-gate-hidden-dim",
        type=int,
        default=TrainConfig.residual_gate_hidden_dim,
        help="Hidden dimension of the compact B2 residual gate.",
    )
    parser.add_argument(
        "--use-global-cross-modal-calibration",
        action="store_true",
        help="H1: add rank-constrained bidirectional calibration of global CLIP embeddings.",
    )
    parser.add_argument(
        "--global-calibration-rank",
        type=int,
        default=TrainConfig.global_calibration_rank,
        help="Bottleneck rank of the H1 global cross-modal calibration branch.",
    )
    parser.add_argument(
        "--freeze-core-model",
        action="store_true",
        help="Freeze the inherited B1-Core and optimize only the newly added H1 modules.",
    )
    parser.add_argument("--prototype-loss-weight", type=float, default=TrainConfig.prototype_loss_weight)
    parser.add_argument("--prototype-temperature", type=float, default=TrainConfig.prototype_temperature)
    parser.add_argument(
        "--disable-prototypes",
        action="store_true",
        help="Remove prototype features and prototype supervision while preserving classifier input dimensions.",
    )
    parser.add_argument(
        "--disable-gate",
        action="store_true",
        help="Replace the learned modality gate with a fixed 50/50 image-text average.",
    )
    parser.add_argument(
        "--disable-low-rank",
        action="store_true",
        help="Remove low-rank interaction features; must be combined with --disable-prototypes.",
    )
    parser.add_argument("--lr-head", type=float, default=TrainConfig.lr_head)
    parser.add_argument("--lr-clip", type=float, default=TrainConfig.lr_clip)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--unfreeze-clip", action="store_true")
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        default=TrainConfig.unfreeze_last_n_layers,
        help="Fine-tune only the final N text and vision CLIP encoder layers.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=TrainConfig.init_checkpoint,
        help="Initialize model weights from an existing best_model.pt before training.",
    )
    parser.add_argument(
        "--init-baseline-checkpoint",
        default=TrainConfig.init_baseline_checkpoint,
        help=(
            "Initialize compatible weights from a CLIP baseline checkpoint. Both variants load CLIP; "
            "residual_add also loads its shape-compatible base classifier."
        ),
    )
    parser.add_argument(
        "--freeze-base-classifier",
        action="store_true",
        help="Keep the warm-started CLIP base classifier fixed while training fusion modules.",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=TrainConfig.class_weight_power)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--grad-accumulation-steps", type=int, default=TrainConfig.grad_accumulation_steps)
    parser.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    parser.add_argument("--warmup-ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--cosine-schedule", action="store_true")
    parser.add_argument("--image-augmentation", action="store_true")
    parser.add_argument("--text-preprocessing", action="store_true")
    parser.add_argument("--use-reliability-gate", action="store_true")
    parser.add_argument("--auxiliary-loss-weight", type=float, default=TrainConfig.auxiliary_loss_weight)
    parser.add_argument("--select-metric", choices=["accuracy", "macro_f1", "weighted_f1"], default=TrainConfig.select_metric)
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Never load or evaluate the test split; save validation-only development outputs.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=TrainConfig.device)
    args = parser.parse_args()
    if args.unfreeze_clip and args.unfreeze_last_n_layers > 0:
        parser.error("Use either --unfreeze-clip or --unfreeze-last-n-layers, not both.")
    if args.unfreeze_last_n_layers < 0:
        parser.error("--unfreeze-last-n-layers must be non-negative.")
    if args.disable_gate and args.use_reliability_gate:
        parser.error("Use either --disable-gate or --use-reliability-gate, not both.")
    if args.disable_low_rank and not args.disable_prototypes:
        parser.error("--disable-low-rank requires --disable-prototypes because prototypes depend on low-rank features.")
    if not 0.0 <= args.fixed_residual_scale <= 1.0:
        parser.error("--fixed-residual-scale must be in [0, 1].")
    if args.residual_warmup_epochs < 0:
        parser.error("--residual-warmup-epochs must be non-negative.")
    if args.residual_gate_hidden_dim < 1:
        parser.error("--residual-gate-hidden-dim must be positive.")
    if args.global_calibration_rank < 1:
        parser.error("--global-calibration-rank must be positive.")
    if args.clean_lca_residual:
        if not args.use_lightweight_cross_attention:
            parser.error("--clean-lca-residual requires --use-lightweight-cross-attention.")
        if args.fusion_variant != "residual_add":
            parser.error("--clean-lca-residual requires --fusion-variant residual_add.")
        if args.use_reliability_gate:
            parser.error("--clean-lca-residual cannot be combined with --use-reliability-gate.")
        if args.prototype_loss_weight != 0.0:
            parser.error("--clean-lca-residual requires --prototype-loss-weight 0.")
    if args.standard_cross_attention_comparator:
        if not args.clean_lca_residual:
            parser.error("--standard-cross-attention-comparator requires --clean-lca-residual.")
        if args.adaptive_residual_gate:
            parser.error("B3 cannot be combined with --adaptive-residual-gate.")
        if args.lca_dim != 512 or args.lca_heads != 8:
            parser.error("B3 requires --lca-dim 512 and --lca-heads 8.")
        if args.fixed_residual_scale != 0.5:
            parser.error("B3 requires --fixed-residual-scale 0.5.")
    if args.adaptive_residual_gate and not args.clean_lca_residual:
        parser.error("--adaptive-residual-gate requires --clean-lca-residual.")
    if args.use_global_cross_modal_calibration:
        if not args.clean_lca_residual:
            parser.error("--use-global-cross-modal-calibration requires --clean-lca-residual.")
        if args.standard_cross_attention_comparator or args.adaptive_residual_gate:
            parser.error("H1 cannot be combined with the B2/B3 comparator flags.")
        if args.lca_dim != 128 or args.lca_heads != 4 or args.fixed_residual_scale != 0.5:
            parser.error("H1 requires the locked B1-Core settings: LCA 128-d/4-head and residual scale 0.5.")
        if not args.development_only:
            parser.error("H1 requires --development-only.")
        if not args.init_checkpoint:
            parser.error("H1 requires --init-checkpoint pointing to the matched B1 checkpoint.")
        if not args.freeze_core_model:
            parser.error("H1 requires --freeze-core-model for isolated constructive ablation.")
    elif args.freeze_core_model:
        parser.error("--freeze-core-model requires --use-global-cross-modal-calibration.")
    if args.residual_warmup_epochs > 0:
        if not args.clean_lca_residual:
            parser.error("--residual-warmup-epochs requires --clean-lca-residual.")
        if args.adaptive_residual_gate:
            parser.error("--residual-warmup-epochs cannot be combined with --adaptive-residual-gate.")
        if args.standard_cross_attention_comparator:
            parser.error("--residual-warmup-epochs is not authorized for the B3 comparator.")
        if not args.development_only:
            parser.error("--residual-warmup-epochs requires --development-only.")
    return TrainConfig(
        data_dir=args.data_dir,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rank_dim=args.rank_dim,
        hidden_dim=args.hidden_dim,
        fusion_variant=args.fusion_variant,
        use_lightweight_cross_attention=args.use_lightweight_cross_attention,
        lca_dim=args.lca_dim,
        lca_heads=args.lca_heads,
        lca_dropout=args.lca_dropout,
        lca_residual_scale=args.lca_residual_scale,
        residual_logit_init=args.residual_logit_init,
        clean_lca_residual=args.clean_lca_residual,
        standard_cross_attention_comparator=args.standard_cross_attention_comparator,
        fixed_residual_scale=args.fixed_residual_scale,
        residual_warmup_epochs=args.residual_warmup_epochs,
        adaptive_residual_gate=args.adaptive_residual_gate,
        residual_gate_hidden_dim=args.residual_gate_hidden_dim,
        use_global_cross_modal_calibration=args.use_global_cross_modal_calibration,
        global_calibration_rank=args.global_calibration_rank,
        freeze_core_model=args.freeze_core_model,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_temperature=args.prototype_temperature,
        disable_prototypes=args.disable_prototypes,
        disable_gate=args.disable_gate,
        disable_low_rank=args.disable_low_rank,
        lr_head=args.lr_head,
        lr_clip=args.lr_clip,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=args.seed,
        freeze_clip=not (args.unfreeze_clip or args.unfreeze_last_n_layers > 0),
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        init_checkpoint=args.init_checkpoint,
        init_baseline_checkpoint=args.init_baseline_checkpoint,
        freeze_base_classifier=args.freeze_base_classifier,
        use_class_weights=not args.no_class_weights,
        class_weight_power=args.class_weight_power,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        grad_accumulation_steps=args.grad_accumulation_steps,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio,
        use_cosine_schedule=args.cosine_schedule,
        image_augmentation=args.image_augmentation,
        text_preprocessing=args.text_preprocessing,
        use_reliability_gate=args.use_reliability_gate,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        select_metric=args.select_metric,
        early_stop_patience=args.early_stop_patience,
        eval_only=args.eval_only,
        development_only=args.development_only,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    data_dir = Path(cfg.data_dir)
    if cfg.clean_lca_residual:
        if cfg.use_global_cross_modal_calibration:
            stage = "h1"
            attention_name = f"globalcalib{cfg.global_calibration_rank}_clean_lca"
        elif cfg.standard_cross_attention_comparator:
            stage = "b3"
            attention_name = "standard_cross_attention"
        else:
            if cfg.residual_warmup_epochs > 0:
                stage = "c1"
            else:
                stage = "b2" if cfg.adaptive_residual_gate else "b1"
            attention_name = "clean_lca"
        run_name = f"{stage}_{attention_name}{cfg.lca_dim}h{cfg.lca_heads}_fixedres{cfg.fixed_residual_scale:g}"
        if cfg.adaptive_residual_gate:
            run_name += f"_adaptivegate{cfg.residual_gate_hidden_dim}"
        if cfg.residual_warmup_epochs > 0:
            run_name += f"_warmup{cfg.residual_warmup_epochs}"
        if cfg.freeze_core_model:
            run_name += "_parentb1_freezecore"
    else:
        run_name = f"{cfg.fusion_variant}_rank{cfg.rank_dim}_proto{cfg.prototype_loss_weight:g}"
        if cfg.use_lightweight_cross_attention:
            run_name += f"_lca{cfg.lca_dim}h{cfg.lca_heads}s{cfg.lca_residual_scale:g}"
    if cfg.fusion_variant == "residual_add" and cfg.residual_logit_init != -3.0:
        run_name += f"_rinit{cfg.residual_logit_init:g}"
    if cfg.disable_prototypes:
        run_name += "_noproto"
    if cfg.disable_gate:
        run_name += "_nogate"
    if cfg.disable_low_rank:
        run_name += "_nolowrank"
    if cfg.use_reliability_gate:
        run_name += f"_urg{cfg.auxiliary_loss_weight:g}"
    if cfg.unfreeze_last_n_layers > 0:
        run_name += f"_uft{cfg.unfreeze_last_n_layers}"
    if cfg.init_baseline_checkpoint:
        run_name += "_warmbase"
    if cfg.freeze_base_classifier:
        run_name += "_freezebase"
    if cfg.select_metric != "weighted_f1":
        run_name += f"_select_{cfg.select_metric}"
    if not cfg.use_class_weights:
        run_name += "_no_class_weights"
    elif cfg.class_weight_power != 1.0:
        run_name += f"_cw{cfg.class_weight_power:g}"
    output_dir = Path(cfg.output_dir) / run_name
    if cfg.development_only:
        output_dir = output_dir.with_name(output_dir.name + "_devonly")
    if cfg.development_only and not cfg.eval_only and output_dir.exists():
        conflicts = [
            name
            for name in ("config.json", "history.csv", "best_model.pt", "best_val_metrics.json")
            if (output_dir / name).exists()
        ]
        if conflicts:
            raise FileExistsError(
                f"Development output directory already contains run artifacts: {output_dir} ({', '.join(conflicts)})"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(cfg))

    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    test_df = None if cfg.development_only else load_split(data_dir, "test")

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but it is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    use_amp = cfg.amp and device.type == "cuda"
    print(f"Device: {device}")
    if cfg.development_only:
        print(f"Train/val rows: {len(train_df)} / {len(val_df)}; test split disabled")
    else:
        print(f"Train/val/test rows: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(
        f"PDLF-CLIP rank_dim={cfg.rank_dim}; proto_weight={cfg.prototype_loss_weight}; "
        f"fusion_variant={cfg.fusion_variant}; lca={cfg.use_lightweight_cross_attention}; "
        f"freeze_clip={cfg.freeze_clip}; unfreeze_last_n_layers={cfg.unfreeze_last_n_layers}"
    )

    processor = CLIPProcessor.from_pretrained(cfg.model_name)
    collator = ClipCollator(processor, cfg.max_length, preprocess_text=cfg.text_preprocessing)
    train_loader = DataLoader(
        CrisisMmdDataset(
            train_df,
            image_augment=TrainingImageAugment() if cfg.image_augmentation else None,
            image_root=cfg.image_root,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CrisisMmdDataset(val_df, image_root=cfg.image_root),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    test_loader = None
    if test_df is not None:
        test_loader = DataLoader(
            CrisisMmdDataset(test_df, image_root=cfg.image_root),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )

    model = PDLFClip(
        model_name=cfg.model_name,
        num_labels=len(LABELS),
        rank_dim=cfg.rank_dim,
        hidden_dim=cfg.hidden_dim,
        fusion_variant=cfg.fusion_variant,
        freeze_clip=cfg.freeze_clip,
        prototype_temperature=cfg.prototype_temperature,
        unfreeze_last_n_layers=cfg.unfreeze_last_n_layers,
        use_lightweight_cross_attention=cfg.use_lightweight_cross_attention,
        lca_dim=cfg.lca_dim,
        lca_heads=cfg.lca_heads,
        lca_dropout=cfg.lca_dropout,
        lca_residual_scale=cfg.lca_residual_scale,
        residual_logit_init=cfg.residual_logit_init,
        clean_lca_residual=cfg.clean_lca_residual,
        fixed_residual_scale=cfg.fixed_residual_scale,
        adaptive_residual_gate=cfg.adaptive_residual_gate,
        residual_gate_hidden_dim=cfg.residual_gate_hidden_dim,
        use_global_cross_modal_calibration=cfg.use_global_cross_modal_calibration,
        global_calibration_rank=cfg.global_calibration_rank,
        freeze_core_model=cfg.freeze_core_model,
        disable_prototypes=cfg.disable_prototypes,
        disable_gate=cfg.disable_gate,
        disable_low_rank=cfg.disable_low_rank,
        use_reliability_gate=cfg.use_reliability_gate,
    ).to(device)
    if cfg.init_checkpoint:
        init_path = Path(cfg.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_path}")
        init_state = torch.load(init_path, map_location="cpu")
        if cfg.use_global_cross_modal_calibration:
            load_report = initialize_from_parent_checkpoint(
                model,
                init_state["model_state_dict"],
                allowed_new_prefixes=(
                    "global_cross_modal_calibration.",
                    "global_residual_classifier.",
                ),
            )
            print(
                "Initialized B1-Core parent; new H1 tensors kept at initialization: "
                f"{len(load_report['missing_new_tensors'])}"
            )
        else:
            model.load_state_dict(init_state["model_state_dict"])
        del init_state
        print(f"Initialized model from: {init_path}")
    if cfg.init_baseline_checkpoint:
        baseline_path = Path(cfg.init_baseline_checkpoint)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint does not exist: {baseline_path}")
        baseline_payload = torch.load(baseline_path, map_location="cpu")
        baseline_state = baseline_payload["model_state_dict"]
        load_counts = initialize_from_clip_baseline(model, baseline_state)
        del baseline_state, baseline_payload
        print(
            f"Initialized from CLIP baseline: {baseline_path} "
            f"(CLIP tensors={load_counts['clip_tensors']}, "
            f"classifier tensors={load_counts['classifier_tensors']})"
        )
        if cfg.fusion_variant == "replace":
            print("Replace classifier remains randomly initialized because its input shape is not baseline-compatible.")
    if cfg.freeze_base_classifier:
        if cfg.fusion_variant != "residual_add":
            raise ValueError("--freeze-base-classifier requires --fusion-variant residual_add")
        if not cfg.init_baseline_checkpoint:
            raise ValueError("--freeze-base-classifier requires --init-baseline-checkpoint")
        for param in model.base_classifier.parameters():
            param.requires_grad = False
        print("Frozen warm-started CLIP base classifier.")
    if cfg.freeze_core_model:
        model.freeze_b1_core()
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        invalid_trainable = [
            name
            for name in trainable_names
            if not name.startswith(("global_cross_modal_calibration.", "global_residual_classifier."))
        ]
        if invalid_trainable or not trainable_names:
            raise RuntimeError(
                "H1 trainable-parameter isolation failed: "
                f"invalid={invalid_trainable}, count={len(trainable_names)}"
            )
        print(f"Frozen B1-Core; H1 trainable tensors={len(trainable_names)}")
    class_weights = compute_class_weights(train_df, device, cfg.class_weight_power) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = None
    if cfg.use_cosine_schedule:
        updates_per_epoch = math.ceil(len(train_loader) / cfg.grad_accumulation_steps)
        total_updates = updates_per_epoch * cfg.epochs
        warmup_steps = int(total_updates * cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_metric = -1.0
    best_epoch = 0
    patience = 0
    history = []
    best_path = output_dir / "best_model.pt"

    if cfg.eval_only:
        if not best_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for eval-only mode: {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if cfg.clean_lca_residual:
            model.set_fixed_residual_scale(
                checkpoint.get("fixed_residual_scale_used", cfg.fixed_residual_scale)
            )
        eval_loader = val_loader if cfg.development_only else test_loader
        eval_metrics = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            cfg.prototype_loss_weight,
            cfg.auxiliary_loss_weight,
        )
        metrics_name = "development_val_metrics.json" if cfg.development_only else "test_metrics.json"
        save_json(output_dir / metrics_name, eval_metrics)
        print("Eval-only checkpoint:", best_path)
        split_name = "Validation" if cfg.development_only else "Test"
        print(f"{split_name} accuracy:", f"{eval_metrics['accuracy']:.4f}")
        print(f"{split_name} macro F1:", f"{eval_metrics['macro_f1']:.4f}")
        print(f"{split_name} weighted F1:", f"{eval_metrics['weighted_f1']:.4f}")
        print(f"Outputs saved to: {output_dir}")
        return

    for epoch in range(1, cfg.epochs + 1):
        if cfg.residual_warmup_epochs > 0:
            scheduled_scale = residual_scale_for_epoch(
                cfg.fixed_residual_scale,
                cfg.residual_warmup_epochs,
                epoch,
            )
            model.set_fixed_residual_scale(scheduled_scale)
            print(
                f"Epoch {epoch:02d} clean residual scale: "
                f"{model.fixed_residual_scale:.6f}"
            )
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=cfg.grad_clip,
            proto_weight=cfg.prototype_loss_weight,
            grad_accumulation_steps=cfg.grad_accumulation_steps,
            scheduler=scheduler,
            auxiliary_weight=cfg.auxiliary_loss_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg.prototype_loss_weight,
            cfg.auxiliary_loss_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_class_loss": train_metrics["class_loss"],
            "train_proto_loss": train_metrics["proto_loss"],
            "train_auxiliary_loss": train_metrics["auxiliary_loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_auxiliary_loss": val_metrics["auxiliary_loss"],
            "val_text_gate": val_metrics["text_gate"],
            "val_image_gate": val_metrics["image_gate"],
            "val_residual_gate_mean": val_metrics["residual_gate_mean"],
            "val_residual_gate_std": val_metrics["residual_gate_std"],
            "val_global_residual_norm_mean": val_metrics["global_residual_norm_mean"],
            "val_global_residual_norm_std": val_metrics["global_residual_norm_std"],
            "val_global_calibration_gate_mean": val_metrics["global_calibration_gate_mean"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"residual_gate={val_metrics['residual_gate_mean']:.3f}+/-{val_metrics['residual_gate_std']:.3f}"
        )

        current_val_metric = val_metrics[cfg.select_metric]
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": LABELS,
                    "config": asdict(cfg),
                    "val_metrics": val_metrics,
                    "best_epoch": epoch,
                    "fixed_residual_scale_used": model.fixed_residual_scale,
                },
                best_path,
            )
            save_json(output_dir / "best_val_metrics.json", val_metrics)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if cfg.clean_lca_residual:
        model.set_fixed_residual_scale(
            checkpoint.get("fixed_residual_scale_used", cfg.fixed_residual_scale)
        )
    if cfg.development_only:
        locked_val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg.prototype_loss_weight,
            cfg.auxiliary_loss_weight,
        )
        save_json(output_dir / "development_val_metrics.json", locked_val_metrics)
        save_json(
            output_dir / "development_complete.json",
            {
                "status": "complete",
                "test_split_loaded": False,
                "test_evaluated": False,
                "selection_metric": cfg.select_metric,
                "best_validation_metric": best_val_metric,
                "best_epoch": checkpoint.get("best_epoch", best_epoch),
                "selected_fixed_residual_scale": model.fixed_residual_scale,
                "residual_warmup_epochs": cfg.residual_warmup_epochs,
                "global_cross_modal_calibration": cfg.use_global_cross_modal_calibration,
                "global_calibration_rank": cfg.global_calibration_rank,
                "core_model_frozen": cfg.freeze_core_model,
            },
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
        print(f"Best validation {cfg.select_metric}:", f"{best_val_metric:.4f}")
        print("Development-only run complete; test split was not loaded or evaluated.")
        print(f"Outputs saved to: {output_dir}")
        return
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        cfg.prototype_loss_weight,
        cfg.auxiliary_loss_weight,
    )
    save_json(output_dir / "test_metrics.json", test_metrics)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")

    print(f"Best validation {cfg.select_metric}:", f"{best_val_metric:.4f}")
    print("Test accuracy:", f"{test_metrics['accuracy']:.4f}")
    print("Test macro F1:", f"{test_metrics['macro_f1']:.4f}")
    print("Test weighted F1:", f"{test_metrics['weighted_f1']:.4f}")
    print("Test gate T/I:", f"{test_metrics['text_gate']:.3f}/{test_metrics['image_gate']:.3f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
