"""Read-only artifact auditor for H32-CDP validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict

import torch


B3_TRAINABLE_PARAMETERS = 3_559_947
B3_TOTAL_PARAMETERS = 431_504_140
REQUIRED_FILES = (
    "config.json",
    "history.csv",
    "best_val_metrics.json",
    "best_model.pt",
    "parameter_counts.json",
    "development_complete.json",
    "smoke_report.json",
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def audit(run_dir: Path) -> Dict:
    errors = []
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        return {"audit_pass": False, "errors": [f"missing files: {missing}"]}
    config = load_json(run_dir / "config.json")
    completion = load_json(run_dir / "development_complete.json")
    best = load_json(run_dir / "best_val_metrics.json")
    counts = load_json(run_dir / "parameter_counts.json")
    smoke = load_json(run_dir / "smoke_report.json")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)

    for source_name, source in (
        ("config", config),
        ("completion", completion),
        ("checkpoint", checkpoint),
    ):
        if source.get("experiment_id") != "H32-CDP":
            errors.append(f"{source_name} has wrong experiment_id")
        if source.get("test_split_loaded") is not False or source.get("test_evaluated") is not False:
            errors.append(f"{source_name} does not prove test isolation")
    if completion.get("distillation_weight") != 2.0:
        errors.append("wrong distillation weight")
    if completion.get("distillation_temperature") != 1.0:
        errors.append("wrong distillation temperature")
    if abs(float(smoke.get("step_zero_clean_distillation", math.inf))) > 1e-6:
        errors.append("step-zero clean distillation is not zero")
    controlled = float(smoke.get("controlled_nonzero_clean_distillation", 0.0))
    if not math.isfinite(controlled) or controlled <= 0:
        errors.append("controlled clean distillation is not positive")
    identities = smoke.get("step_zero_max_logit_difference", {})
    if set(identities) != {"clean", "missing_image", "missing_text", "image_mismatch"}:
        errors.append("smoke identity conditions are incomplete")
    elif any(float(value) != 0.0 for value in identities.values()):
        errors.append("step-zero B1 identity failed")
    if smoke.get("frozen_nonzero_gradient_count") != 0:
        errors.append("frozen gradient isolation failed")
    if float(smoke.get("trainable_gradient_l1", 0.0)) <= 0:
        errors.append("no H32-CDP gradient")
    mismatch = smoke.get("mismatch_audit", {})
    if mismatch.get("train_same_class_count") != 0 or mismatch.get("validation_same_class_count") != 0:
        errors.append("different-class mismatch audit failed")
    if mismatch.get("train_rows") != 5119 or mismatch.get("validation_rows") != 1097:
        errors.append("mismatch coverage failed")
    if counts.get("trainable_parameters") != 517_509:
        errors.append("H32-CDP changed the locked trainable parameter count")
    if counts.get("trainable_parameters") != counts.get("rlif_only_parameters"):
        errors.append("trainable/RLIF-only count mismatch")
    if float(counts.get("trainable_parameters", math.inf)) > 0.3 * B3_TRAINABLE_PARAMETERS:
        errors.append("trainable parameter budget failed")
    if int(counts.get("total_parameters", B3_TOTAL_PARAMETERS)) >= B3_TOTAL_PARAMETERS:
        errors.append("total parameter budget failed")
    if not isinstance(checkpoint.get("rlif_state_dict"), dict) or not checkpoint["rlif_state_dict"]:
        errors.append("compact RLIF state missing")
    conditions = completion.get("conditions", {})
    if set(conditions) != {"clean", "missing_image", "missing_text", "image_mismatch"}:
        errors.append("final condition metrics incomplete")
    with (run_dir / "history.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) > 8:
        errors.append("history length outside 1..8")
    elif "train_clean_distillation" not in rows[0]:
        errors.append("clean distillation is absent from history")
    else:
        for row in rows:
            value = float(row["train_clean_distillation"])
            if not math.isfinite(value) or value < 0:
                errors.append("invalid clean distillation history")
                break
    best_epoch = int(completion.get("best_epoch", 0))
    if best_epoch < 1 or best_epoch > len(rows):
        errors.append("best_epoch invalid")
    stored_clean = checkpoint.get("clean_validation_metrics", {})
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        left = stored_clean.get("h32", {}).get(metric)
        right = best.get("h32", {}).get(metric)
        if left is None or right is None or abs(float(left) - float(right)) > 1e-12:
            errors.append(f"checkpoint/best metric mismatch: {metric}")
    return {
        "audit_pass": not errors,
        "errors": errors,
        "run_dir": str(run_dir.resolve()),
        "seed": config.get("seed"),
        "test_split_loaded": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one H32-CDP run")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = audit(run_dir)
    with (run_dir / "h32_cdp_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
