#!/usr/bin/env python3
"""One-time locked clean-test evaluation for one frozen H32-CDP seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

from train_h32_rlif import build_model, load_h32_delta
from train_pdlf_clip import (
    LABELS,
    ClipCollator,
    CrisisMmdDataset,
    confusion_matrix,
    load_split,
    metrics_from_confusion,
    move_batch,
    set_seed,
)


AUTHORIZED_STATUS = "locked_authorized_postdevelopment_multicriteria_selection"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock(
    path: Path,
    token: str,
    seed: int,
    checkpoint: Path,
    parent_checkpoint: Path,
) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != AUTHORIZED_STATUS:
        raise ValueError("H32-CDP final-test lock is not authorized")
    if token != lock.get("authorization_token"):
        raise ValueError("Invalid H32-CDP final-test authorization token")
    if seed not in lock.get("seeds", []):
        raise ValueError(f"Seed {seed} is not locked")
    if not lock.get("original_admission_failure_disclosed", False):
        raise ValueError("Original validation admission failure must remain disclosed")
    if lock.get("checkpoint_or_hyperparameter_change_authorized", True):
        raise ValueError("Lock must prohibit checkpoint and hyperparameter changes")

    evaluator_hash = file_sha256(Path(__file__)).lower()
    if evaluator_hash != lock.get("evaluator_sha256", "").lower():
        raise ValueError("Evaluator hash mismatch")
    code_root = Path(__file__).resolve().parent
    for filename, expected_hash in lock.get("dependency_sha256", {}).items():
        actual_hash = file_sha256(code_root / filename).lower()
        if actual_hash != expected_hash.lower():
            raise ValueError(f"Dependency hash mismatch for {filename}")

    seed_lock = lock["checkpoints"][str(seed)]
    if file_sha256(checkpoint).lower() != seed_lock["h32_cdp_sha256"].lower():
        raise ValueError(f"H32-CDP checkpoint hash mismatch for seed {seed}")
    if file_sha256(parent_checkpoint).lower() != seed_lock["b1_parent_sha256"].lower():
        raise ValueError(f"B1 parent checkpoint hash mismatch for seed {seed}")
    return lock


@torch.inference_mode()
def predict(model, loader: DataLoader, device: torch.device, use_amp: bool):
    model.eval()
    probability_rows, labels, sample_ids = [], [], []
    for batch in tqdm(loader, desc="locked H32-CDP clean test", leave=False):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch)["class_logits"]
        probability_rows.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.extend(batch["labels"].cpu().tolist())
        sample_ids.extend(str(item) for item in batch["sample_ids"])
    return (
        np.concatenate(probability_rows, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sample_ids, dtype=str),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lock-config", required=True)
    result.add_argument("--authorization-token", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--parent-checkpoint", required=True)
    result.add_argument("--seed", required=True, type=int)
    result.add_argument("--data-dir", required=True)
    result.add_argument("--image-root", default="")
    result.add_argument("--model-name", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--no-amp", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    lock_path = Path(args.lock_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse final-test output directory: {output_dir}")
    lock = load_lock(
        lock_path, args.authorization_token, args.seed, checkpoint, parent_checkpoint
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    test_df = load_split(Path(args.data_dir), "test")
    if len(test_df) != int(lock["test_size"]):
        raise ValueError(f"Locked test size mismatch: {len(test_df)} != {lock['test_size']}")
    counts = test_df["label_id"].value_counts().reindex(range(len(LABELS)), fill_value=0).tolist()
    if counts != lock["test_class_counts"]:
        raise ValueError(f"Locked test class counts mismatch: {counts}")

    processor = CLIPProcessor.from_pretrained(args.model_name, local_files_only=True)
    loader = DataLoader(
        CrisisMmdDataset(test_df, image_root=args.image_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ClipCollator(processor, max_length=77, preprocess_text=False),
        pin_memory=device.type == "cuda",
    )
    model = build_model(args.model_name, parent_checkpoint, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != "H32-CDP" or int(payload.get("seed", -1)) != args.seed:
        raise ValueError("Checkpoint identity or seed does not match the lock")
    if payload.get("test_split_loaded") or payload.get("test_evaluated"):
        raise ValueError("Development checkpoint reports prior test access")
    expected_epoch = int(lock["checkpoints"][str(args.seed)]["selected_epoch"])
    if int(payload.get("epoch", -1)) != expected_epoch:
        raise ValueError("Checkpoint epoch does not match the lock")
    load_h32_delta(model, payload["rlif_state_dict"])

    probabilities, labels, sample_ids = predict(
        model, loader, device, use_amp=device.type == "cuda" and not args.no_amp
    )
    predictions = probabilities.argmax(axis=1).tolist()
    matrix = confusion_matrix(predictions, labels.tolist(), len(LABELS))
    metrics = metrics_from_confusion(matrix)
    metrics["confusion_matrix"] = matrix.tolist()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "predictions.npz"
    np.savez_compressed(
        artifact_path, probabilities=probabilities, labels=labels, sample_ids=sample_ids
    )
    audit = {
        "audit_pass": True,
        "evaluation_type": "locked_one_time_h32_cdp_clean_final_test",
        "lock_id": lock["lock_id"],
        "seed": args.seed,
        "selected_epoch": expected_epoch,
        "h32_cdp_checkpoint_sha256": file_sha256(checkpoint),
        "b1_parent_checkpoint_sha256": file_sha256(parent_checkpoint),
        "evaluator_sha256": file_sha256(Path(__file__)),
        "lock_config_sha256": file_sha256(lock_path),
        "test_split_loaded": True,
        "test_evaluated": True,
        "test_size": int(labels.size),
        "test_class_counts": counts,
        "metrics": metrics,
        "prediction_artifact": str(artifact_path),
        "prediction_artifact_sha256": file_sha256(artifact_path),
        "selection_or_tuning_performed": False,
        "rerun_or_model_change_authorized": False,
        "original_admission_failure_disclosed": True,
    }
    audit_path = output_dir / "h32_cdp_final_test_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
