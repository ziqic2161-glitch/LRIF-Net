import json
import types
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from h32_rlif_model import H32RLIF, parameter_counts, trainable_parameter_names
from train_h32_rlif import (
    build_different_class_mismatch_indices,
    corruption_view,
    missing_text_view,
)


class DummyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 6)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        return types.SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


class DummyVisionModel(nn.Module):
    def forward(self, pixel_values):
        patches = pixel_values.flatten(2).transpose(1, 2)
        pooled = patches.mean(1)
        hidden = torch.cat([pooled.unsqueeze(1), patches], dim=1)
        return types.SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


class DummyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            projection_dim=8,
            text_config=types.SimpleNamespace(hidden_size=6),
            vision_config=types.SimpleNamespace(hidden_size=7),
        )
        self.text_model = DummyTextModel()
        self.vision_model = DummyVisionModel()
        self.text_projection = nn.Linear(6, 8)
        self.visual_projection = nn.Linear(7, 8)


class DummyLCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Linear(6, 8)
        self.image = nn.Linear(7, 8)

    def forward(self, text_tokens, image_patches, text_mask):
        weights = text_mask.to(text_tokens.dtype).unsqueeze(-1)
        text = (text_tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        return {"text": self.text(text), "image": self.image(image_patches.mean(1))}


class DummyParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = DummyClip()
        self.lightweight_cross_attention = DummyLCA()
        self.base_classifier = nn.Linear(16, 5)
        self.aux_classifier = nn.Linear(16, 5)
        self.fixed_residual_scale = 0.5
        self.clean_lca_residual = True
        self.use_global_cross_modal_calibration = False
        self.adaptive_residual_gate = False


def sample_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
        "pixel_values": torch.randn(2, 7, 4, 4),
        "mismatch_pixel_values": torch.randn(2, 7, 4, 4),
        "labels": torch.tensor([0, 1]),
        "mismatch_labels": torch.tensor([1, 0]),
    }


class H32Tests(unittest.TestCase):
    def test_mismatch_indices_are_deterministic_and_different_class(self):
        labels = [0, 0, 1, 1, 2, 2]
        first = build_different_class_mismatch_indices(labels, 3141)
        second = build_different_class_mismatch_indices(labels, 3141)
        self.assertEqual(first, second)
        self.assertTrue(all(labels[i] != labels[j] for i, j in enumerate(first)))

    def test_missing_text_keeps_only_special_boundaries(self):
        batch = sample_batch()
        view = missing_text_view(batch)
        self.assertEqual(view["attention_mask"].sum(dim=1).tolist(), [2, 2])
        self.assertEqual(view["attention_mask"][:, 0].tolist(), [1, 1])
        self.assertEqual(view["attention_mask"][:, 2].tolist(), [1, 1])

    def test_corruption_views(self):
        batch = sample_batch()
        self.assertEqual(float(corruption_view(batch, "missing_image")["pixel_values"].abs().max()), 0.0)
        mismatch = corruption_view(batch, "image_mismatch")
        self.assertTrue(torch.equal(mismatch["pixel_values"], batch["mismatch_pixel_values"]))
        invalid = dict(batch)
        invalid["mismatch_labels"] = invalid["labels"].clone()
        with self.assertRaisesRegex(RuntimeError, "same-class"):
            corruption_view(invalid, "image_mismatch")

    def test_step_zero_matches_b1_for_every_view(self):
        model = H32RLIF(
            DummyParent(), latent_count=3, latent_dim=8, heads=2, rounds=2,
            ffn_dim=16, residual_hidden_dim=8,
        ).eval()
        batch = sample_batch()
        with torch.no_grad():
            for condition in ("clean", "missing_image", "missing_text", "image_mismatch"):
                view = batch if condition == "clean" else corruption_view(batch, condition)
                outputs = model(view)
                self.assertEqual(
                    float((outputs["class_logits"] - outputs["b1_logits"]).abs().max()), 0.0
                )

    def test_only_rlif_is_trainable_and_receives_gradient(self):
        model = H32RLIF(
            DummyParent(), latent_count=3, latent_dim=8, heads=2, rounds=2,
            ffn_dim=16, residual_hidden_dim=8,
        )
        names = trainable_parameter_names(model)
        self.assertTrue(names)
        self.assertTrue(all(not p.requires_grad for p in model.parent.parameters()))
        outputs = model(sample_batch())
        F.cross_entropy(outputs["class_logits"], sample_batch()["labels"]).backward()
        self.assertGreater(model.residual_classifier[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parent.parameters()))
        counts = parameter_counts(model)
        self.assertEqual(counts["trainable_parameters"], counts["rlif_only_parameters"])

    def test_tied_block_is_reused_not_duplicated(self):
        model = H32RLIF(
            DummyParent(), latent_count=3, latent_dim=8, heads=2, rounds=2,
            ffn_dim=16, residual_hidden_dim=8,
        )
        block_names = [name for name, _ in model.named_modules() if name.startswith("latent_block")]
        self.assertFalse(any(name.startswith("latent_block.1") for name in block_names))
        self.assertEqual(model.rounds, 2)

    def test_protocol_lock(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "h32_rlif_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["architecture"]["latent_count"], 8)
        self.assertEqual(payload["architecture"]["recurrent_rounds"], 2)
        self.assertEqual(payload["training"]["corruption_cycle"], [
            "missing_image", "missing_text", "different_class_image_mismatch"
        ])
        self.assertIn(payload["next_authorized_action"], {
            "implement_and_verify_only_no_gpu_run",
            "run_seed3141_validation_only_manually",
            "run_seed1729_validation_only_manually",
            "run_seed2718_validation_only_manually",
            "close_h32_after_gate_failure",
            "lock_three_seed_test_after_admission",
        })


if __name__ == "__main__":
    unittest.main()
