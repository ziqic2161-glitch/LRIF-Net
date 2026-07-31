"""Validation-only trainer for H32-CDP clean decision preservation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import train_h32_rlif as h32


PROTOCOL_PATH = Path("configs/h32_cdp_v1.json")
EXPERIMENT_ID = "H32-CDP"
DISTILLATION_WEIGHT = 2.0
DISTILLATION_TEMPERATURE = 1.0


def clean_teacher_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = DISTILLATION_TEMPERATURE,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits.detach() / temperature
    return (
        F.kl_div(
            F.log_softmax(scaled_student, dim=-1),
            F.softmax(scaled_teacher, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def stable_smoke_teacher_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the smoke-only KL in float64 to avoid tiny negative roundoff."""
    return clean_teacher_kl(student_logits.double(), teacher_logits.double())


def loss_items(clean: Mapping, corrupt: Mapping, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    items = h32.loss_items(clean, corrupt, labels)
    distillation = clean_teacher_kl(clean["class_logits"], clean["b1_logits"])
    return {
        **items,
        "h32_loss": items["loss"],
        "clean_distillation": distillation,
        "loss": items["loss"] + DISTILLATION_WEIGHT * distillation,
    }


def train_epoch(
    model,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    keys = (
        "loss",
        "h32_loss",
        "clean_ce",
        "corrupt_ce",
        "consistency",
        "residual_l2",
        "clean_distillation",
    )
    values = {key: [] for key in keys}
    accumulation = 4
    optimizer_step = 0
    corruption_counts = {name: 0 for name in h32.CORRUPTIONS}
    for step, raw_batch in enumerate(tqdm(loader, desc="H32-CDP train", leave=False)):
        batch = h32.move_batch(raw_batch, device)
        corruption = h32.CORRUPTIONS[optimizer_step % len(h32.CORRUPTIONS)]
        corruption_counts[corruption] += 1
        with torch.cuda.amp.autocast(enabled=use_amp):
            clean = model(batch)
            corrupt = model(h32.corruption_view(batch, corruption))
            items = loss_items(clean, corrupt, batch["labels"])
            backward = items["loss"] / accumulation
        scaler.scale(backward).backward()
        should_step = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
        for key in keys:
            values[key].append(float(items[key].detach().item()))
    result = {f"train_{key}": float(np.mean(rows)) for key, rows in values.items()}
    result.update({f"train_{name}_batches": count for name, count in corruption_counts.items()})
    return result


def cdp_smoke(model, raw_batch: Dict, device: torch.device) -> Dict:
    report = h32.identity_gradient_smoke(model, raw_batch, device)
    batch = h32.move_batch(raw_batch, device)
    model.eval()
    with torch.no_grad():
        clean = model(batch)
        step_zero = float(
            stable_smoke_teacher_kl(clean["class_logits"], clean["b1_logits"]).item()
        )
        final_layer = model.residual_classifier[-1]
        saved_bias = final_layer.bias.detach().clone()
        final_layer.bias[0] += 1.0
        shifted = model(batch)
        controlled = float(
            stable_smoke_teacher_kl(
                shifted["class_logits"], shifted["b1_logits"]
            ).item()
        )
        final_layer.bias.copy_(saved_bias)
    if abs(step_zero) > 1e-6 or not math.isfinite(controlled) or controlled <= 0:
        raise RuntimeError(
            f"H32-CDP distillation smoke failed: step_zero={step_zero}, controlled={controlled}"
        )
    report.update(
        {
            "step_zero_clean_distillation": step_zero,
            "controlled_nonzero_clean_distillation": controlled,
            "distillation_weight": DISTILLATION_WEIGHT,
            "distillation_temperature": DISTILLATION_TEMPERATURE,
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H32-CDP validation-only trainer")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=[3141, 1729, 2718], required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = h32.load_json(PROTOCOL_PATH)
    authorized_seed_by_action = {
        "run_seed3141_validation_only_manually": 3141,
        "run_seed1729_validation_only_manually": 1729,
        "run_seed2718_validation_only_manually": 2718,
    }
    action = protocol["next_authorized_action"]
    if not args.smoke_only and authorized_seed_by_action.get(action) != args.seed:
        raise RuntimeError(
            f"H32-CDP seed {args.seed} is not authorized under action {action!r}"
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"H32-CDP output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    h32.set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = h32.build_model(args.model_name, Path(args.parent_checkpoint), device)
    counts = h32.parameter_counts(model)
    h32.save_json(output_dir / "parameter_counts.json", counts)
    train_df, val_df, train_loader, val_loader, mismatch_audit = h32.make_loaders(args)
    if mismatch_audit["train_same_class_count"] or mismatch_audit["validation_same_class_count"]:
        raise RuntimeError(f"Mismatch audit failed: {mismatch_audit}")
    smoke = cdp_smoke(model, next(iter(train_loader)), device)
    smoke.update(
        {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "mismatch_audit": mismatch_audit,
            "test_split_loaded": False,
            "test_evaluated": False,
        }
    )
    h32.save_json(output_dir / "smoke_report.json", smoke)
    if args.smoke_only:
        print(f"H32-CDP smoke passed: {output_dir}")
        return

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    best_accuracy = -math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, 9):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        clean = h32.evaluate_condition(model, val_loader, device, "clean")
        clean_h32 = clean["h32"]
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_accuracy": clean_h32["accuracy"],
            "val_macro_f1": clean_h32["macro_f1"],
            "val_weighted_f1": clean_h32["weighted_f1"],
            "val_residual_std": clean_h32["rlif_residual_std"],
        }
        history.append(row)
        if float(clean_h32["accuracy"]) > best_accuracy:
            best_accuracy = float(clean_h32["accuracy"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "seed": args.seed,
                    "epoch": epoch,
                    "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
                    "rlif_state_dict": h32.h32_delta_state(model),
                    "clean_validation_metrics": clean,
                    "test_split_loaded": False,
                    "test_evaluated": False,
                },
                output_dir / "best_model.pt",
            )
            h32.save_json(output_dir / "best_val_metrics.json", clean)
        else:
            stale += 1
        if stale >= 4:
            break
    h32.write_history(output_dir / "history.csv", history)
    best = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    h32.load_h32_delta(model, best["rlif_state_dict"])
    conditions = h32.evaluate_all_conditions(model, val_loader, device)
    parent_clean = h32.parent_metrics(Path(args.parent_checkpoint))
    clean_deltas = {
        key: float(conditions["clean"]["h32"][key]) - float(parent_clean[key])
        for key in ("accuracy", "macro_f1", "weighted_f1")
    }
    corrupted_accuracy_deltas = {
        name: float(conditions[name]["deltas"]["accuracy"]) for name in h32.CORRUPTIONS
    }
    mean_corrupted_delta = float(np.mean(list(corrupted_accuracy_deltas.values())))
    gate = protocol["first_seed_gate"]
    checks = {
        "clean_accuracy": clean_deltas["accuracy"] >= gate["minimum_clean_accuracy_delta"],
        "clean_macro_f1": clean_deltas["macro_f1"] >= gate["minimum_clean_macro_f1_delta"],
        "clean_weighted_f1": clean_deltas["weighted_f1"] >= gate["minimum_clean_weighted_f1_delta"],
        "mean_corrupted_accuracy": mean_corrupted_delta >= gate["minimum_mean_corrupted_accuracy_delta"],
        "each_corruption_accuracy": min(corrupted_accuracy_deltas.values()) >= gate["minimum_each_corruption_accuracy_delta"],
        "residual_std": conditions["clean"]["h32"]["rlif_residual_std"] >= gate["minimum_residual_std"],
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "seed": args.seed,
        "data_dir": str(Path(args.data_dir).resolve()),
        "model_name": args.model_name,
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "best_epoch": best_epoch,
        "protocol": protocol,
        "test_split_loaded": False,
        "test_evaluated": False,
    }
    completion = {
        "experiment_id": EXPERIMENT_ID,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "parent_clean_metrics": parent_clean,
        "conditions": conditions,
        "clean_deltas_vs_b1": clean_deltas,
        "corrupted_accuracy_deltas_vs_b1": corrupted_accuracy_deltas,
        "mean_corrupted_accuracy_delta_vs_b1": mean_corrupted_delta,
        "first_seed_gate_checks": checks,
        "first_seed_gate_pass": all(checks.values()),
        "distillation_weight": DISTILLATION_WEIGHT,
        "distillation_temperature": DISTILLATION_TEMPERATURE,
        "test_split_loaded": False,
        "test_evaluated": False,
    }
    h32.save_json(output_dir / "config.json", config)
    h32.save_json(output_dir / "development_complete.json", completion)
    print(f"H32-CDP validation run complete: {output_dir}")


if __name__ == "__main__":
    main()
