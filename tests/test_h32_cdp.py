import json
import tempfile
import unittest
from pathlib import Path

import torch

from train_h32_cdp import (
    DISTILLATION_TEMPERATURE,
    DISTILLATION_WEIGHT,
    clean_teacher_kl,
    loss_items,
    stable_smoke_teacher_kl,
)


class H32CDPTests(unittest.TestCase):
    def test_distillation_is_zero_for_identical_logits(self):
        logits = torch.tensor([[1.0, -0.5, 0.2]])
        self.assertLess(abs(float(clean_teacher_kl(logits, logits))), 1e-6)

    def test_distillation_is_positive_and_moves_student_toward_teacher(self):
        teacher = torch.tensor([[2.0, -1.0, 0.5]])
        student = torch.tensor([[-1.0, 2.0, 0.5]], requires_grad=True)
        before = clean_teacher_kl(student, teacher)
        before.backward()
        updated = (student.detach() - 0.1 * student.grad).requires_grad_(True)
        after = clean_teacher_kl(updated, teacher)
        self.assertGreater(float(before), 0.0)
        self.assertLess(float(after), float(before))

    def test_smoke_kl_uses_float64_for_saturated_logits(self):
        teacher = torch.tensor([[0.0, -20.0, -20.0, -20.0, -20.0]])
        student = teacher.clone()
        student[:, 1] += 0.25
        self.assertLess(float(clean_teacher_kl(student, teacher)), 0.0)
        self.assertGreater(float(stable_smoke_teacher_kl(student, teacher)), 0.0)

    def test_total_loss_adds_exact_locked_teacher_term(self):
        labels = torch.tensor([0, 1])
        clean = {
            "class_logits": torch.tensor([[0.2, -0.1], [-0.3, 0.4]], requires_grad=True),
            "b1_logits": torch.tensor([[0.5, -0.2], [-0.1, 0.2]]),
            "rlif_residual_logits": torch.ones(2, 2),
        }
        corrupt = {
            "class_logits": torch.tensor([[0.0, 0.1], [0.2, -0.2]], requires_grad=True),
            "rlif_residual_logits": torch.ones(2, 2) * 2,
        }
        items = loss_items(clean, corrupt, labels)
        expected = items["h32_loss"] + DISTILLATION_WEIGHT * items["clean_distillation"]
        self.assertTrue(torch.allclose(items["loss"], expected))

    def test_protocol_locks_single_change_and_strict_gate(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "h32_cdp_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["architecture_unchanged_from_h32"])
        self.assertEqual(payload["training"]["loss"]["frozen_b1_to_clean_kl"], 2.0)
        self.assertEqual(payload["training"]["loss"]["distillation_temperature"], 1.0)
        self.assertEqual(payload["first_seed_gate"]["minimum_clean_accuracy_delta"], 0.0)
        self.assertEqual(DISTILLATION_WEIGHT, 2.0)
        self.assertEqual(DISTILLATION_TEMPERATURE, 1.0)

    def test_protocol_blocks_gpu_before_verification(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "h32_cdp_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(payload["next_authorized_action"], {
            "implement_and_verify_only_no_gpu_run",
            "run_seed3141_validation_only_manually",
            "run_seed1729_validation_only_manually",
            "run_seed2718_validation_only_manually",
            "close_h32_cdp_after_gate_failure",
            "lock_three_seed_test_after_admission",
            "final_test_seed3141_locked_postdevelopment_selection",
            "final_test_seed1729_locked_after_seed3141_audit",
            "final_test_seed2718_locked_after_seed1729_audit",
            "aggregate_locked_three_seed_equal_probability_once",
            "final_test_complete_no_further_model_or_ensemble_run",
        })


if __name__ == "__main__":
    unittest.main()
