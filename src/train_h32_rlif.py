"""Validation-only H32-RLIF trainer and real-checkpoint smoke runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor

from h32_rlif_model import (
    H32RLIF,
    build_b1_parent,
    parameter_counts,
    trainable_parameter_names,
)
from train_pdlf_clip import (
    ClipCollator,
    CrisisMmdDataset,
    LABELS,
    confusion_matrix,
    load_split,
    metrics_from_confusion,
    move_batch,
    set_seed,
)


PROTOCOL_PATH = Path("configs/h32_rlif_v1.json")
CORRUPTIONS = ("missing_image", "missing_text", "image_mismatch")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_different_class_mismatch_indices(labels: Sequence[int], seed: int) -> list[int]:
    """Build a deterministic different-class partner for every row."""
    labels = [int(label) for label in labels]
    count = len(labels)
    if count < 2 or len(set(labels)) < 2:
        raise ValueError("Mismatch construction requires at least two classes")
    stride = 1 + (int(seed) % (count - 1))
    indices = []
    for index, label in enumerate(labels):
        candidate = (index + stride) % count
        searched = 0
        while labels[candidate] == label and searched < count:
            candidate = (candidate + 1) % count
            searched += 1
        if labels[candidate] == label or candidate == index:
            raise RuntimeError(f"Could not construct mismatch for row {index}")
        indices.append(candidate)
    return indices


class H32Dataset(Dataset):
    def __init__(self, dataframe, image_root: str, mismatch_indices: Sequence[int]):
        if len(dataframe) != len(mismatch_indices):
            raise ValueError("Mismatch index count must equal dataset size")
        self.base = CrisisMmdDataset(dataframe, image_root=image_root)
        self.mismatch_indices = [int(index) for index in mismatch_indices]
        labels = dataframe["label_id"].astype(int).tolist()
        for index, mismatch in enumerate(self.mismatch_indices):
            if labels[index] == labels[mismatch]:
                raise ValueError("H32 mismatch partners must have a different class")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int) -> Dict:
        clean = self.base[index]
        mismatch = self.base[self.mismatch_indices[index]]
        clean["mismatch_image"] = mismatch["image"]
        clean["mismatch_label"] = mismatch["label"]
        clean["mismatch_sample_id"] = mismatch["sample_id"]
        return clean


class H32Collator:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor
        self.clean_collator = ClipCollator(processor, max_length=77, preprocess_text=False)

    def __call__(self, items: list[Dict]) -> Dict:
        clean_items = [
            {key: item[key] for key in ("text", "image", "label", "sample_id")}
            for item in items
        ]
        batch = self.clean_collator(clean_items)
        mismatch = self.processor(
            images=[item["mismatch_image"] for item in items], return_tensors="pt"
        )
        batch["mismatch_pixel_values"] = mismatch["pixel_values"]
        batch["mismatch_labels"] = torch.tensor(
            [item["mismatch_label"] for item in items], dtype=torch.long
        )
        batch["mismatch_sample_ids"] = [item["mismatch_sample_id"] for item in items]
        return batch


def missing_text_view(batch: Mapping) -> Dict:
    view = dict(batch)
    mask = batch["attention_mask"].clone()
    reduced = torch.zeros_like(mask)
    for row in range(mask.shape[0]):
        valid = torch.nonzero(mask[row], as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError("H32 requires at least one valid text token")
        reduced[row, valid[0]] = 1
        reduced[row, valid[-1]] = 1
    view["attention_mask"] = reduced
    return view


def corruption_view(batch: Mapping, corruption: str) -> Dict:
    view = dict(batch)
    if corruption == "missing_image":
        view["pixel_values"] = torch.zeros_like(batch["pixel_values"])
    elif corruption == "missing_text":
        view = missing_text_view(batch)
    elif corruption == "image_mismatch":
        if torch.any(batch["mismatch_labels"] == batch["labels"]):
            raise RuntimeError("Image mismatch view contains a same-class partner")
        view["pixel_values"] = batch["mismatch_pixel_values"]
    else:
        raise ValueError(f"Unknown corruption: {corruption}")
    return view


def extract_parent_state(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("B1 checkpoint is missing model_state_dict")
    return state


def build_model(
    model_name: str,
    parent_checkpoint: Path,
    device: torch.device,
    model_kwargs: Mapping | None = None,
) -> H32RLIF:
    parent = build_b1_parent(model_name)
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    parent.load_state_dict(extract_parent_state(payload), strict=True)
    model = H32RLIF(parent, **dict(model_kwargs or {}))
    trainable_parameter_names(model)
    return model.to(device)


def h32_delta_state(model: H32RLIF) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("parent.")
    }


def load_h32_delta(model: H32RLIF, state: Mapping[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [name for name in incompatible.missing_keys if not name.startswith("parent.")]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"Invalid H32 compact state: missing={invalid_missing}, unexpected={unexpected}"
        )


def make_loaders(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    train_df = load_split(data_dir, "train")
    val_df = load_split(data_dir, "val")
    train_mismatch = build_different_class_mismatch_indices(
        train_df["label_id"].tolist(), args.seed
    )
    val_mismatch = build_different_class_mismatch_indices(
        val_df["label_id"].tolist(), args.seed + 10000
    )
    processor = CLIPProcessor.from_pretrained(args.model_name)
    collator = H32Collator(processor)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        H32Dataset(train_df, args.image_root, train_mismatch),
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        H32Dataset(val_df, args.image_root, val_mismatch),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    mismatch_audit = {
        "train_rows": len(train_mismatch),
        "validation_rows": len(val_mismatch),
        "train_same_class_count": int(
            sum(
                train_df.iloc[i]["label_id"] == train_df.iloc[j]["label_id"]
                for i, j in enumerate(train_mismatch)
            )
        ),
        "validation_same_class_count": int(
            sum(
                val_df.iloc[i]["label_id"] == val_df.iloc[j]["label_id"]
                for i, j in enumerate(val_mismatch)
            )
        ),
    }
    return train_df, val_df, train_loader, val_loader, mismatch_audit


def kl_clean_to_corrupt(clean_logits: torch.Tensor, corrupt_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(corrupt_logits, dim=-1),
        F.softmax(clean_logits.detach(), dim=-1),
        reduction="batchmean",
    )


def loss_items(clean: Mapping, corrupt: Mapping, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    clean_ce = F.cross_entropy(clean["class_logits"], labels)
    corrupt_ce = F.cross_entropy(corrupt["class_logits"], labels)
    consistency = kl_clean_to_corrupt(clean["class_logits"], corrupt["class_logits"])
    residual_l2 = 0.5 * (
        clean["rlif_residual_logits"].pow(2).mean()
        + corrupt["rlif_residual_logits"].pow(2).mean()
    )
    total = clean_ce + 0.5 * corrupt_ce + 0.1 * consistency + 0.01 * residual_l2
    return {
        "loss": total,
        "clean_ce": clean_ce,
        "corrupt_ce": corrupt_ce,
        "consistency": consistency,
        "residual_l2": residual_l2,
    }


def train_epoch(
    model: H32RLIF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = {key: [] for key in ("loss", "clean_ce", "corrupt_ce", "consistency", "residual_l2")}
    accumulation = 4
    optimizer_step = 0
    corruption_counts = {name: 0 for name in CORRUPTIONS}
    for step, raw_batch in enumerate(tqdm(loader, desc="H32 train", leave=False)):
        batch = move_batch(raw_batch, device)
        corruption = CORRUPTIONS[optimizer_step % len(CORRUPTIONS)]
        corruption_counts[corruption] += 1
        with torch.cuda.amp.autocast(enabled=use_amp):
            clean = model(batch)
            corrupt = model(corruption_view(batch, corruption))
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
        for key in values:
            values[key].append(float(items[key].detach().item()))
    result = {f"train_{key}": float(np.mean(rows)) for key, rows in values.items()}
    result.update({f"train_{name}_batches": count for name, count in corruption_counts.items()})
    return result


@torch.no_grad()
def evaluate_condition(
    model: H32RLIF, loader: DataLoader, device: torch.device, condition: str
) -> Dict[str, Dict]:
    model.eval()
    h32_preds, b1_preds, labels, residuals = [], [], [], []
    for raw_batch in tqdm(loader, desc=f"H32 eval {condition}", leave=False):
        batch = move_batch(raw_batch, device)
        view = batch if condition == "clean" else corruption_view(batch, condition)
        outputs = model(view)
        h32_preds.extend(outputs["class_logits"].argmax(-1).cpu().tolist())
        b1_preds.extend(outputs["b1_logits"].argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
        residuals.append(outputs["rlif_residual_logits"].cpu())
    h32 = metrics_from_confusion(confusion_matrix(h32_preds, labels, len(LABELS)))
    b1 = metrics_from_confusion(confusion_matrix(b1_preds, labels, len(LABELS)))
    residual = torch.cat(residuals)
    h32.update(
        {
            "rlif_residual_mean": float(residual.mean().item()),
            "rlif_residual_std": float(residual.std(unbiased=False).item()),
            "rlif_residual_abs_mean": float(residual.abs().mean().item()),
        }
    )
    deltas = {
        key: float(h32[key]) - float(b1[key])
        for key in ("accuracy", "macro_f1", "weighted_f1")
    }
    return {"h32": h32, "b1": b1, "deltas": deltas}


def evaluate_all_conditions(model: H32RLIF, loader: DataLoader, device: torch.device) -> Dict:
    return {
        condition: evaluate_condition(model, loader, device, condition)
        for condition in ("clean", *CORRUPTIONS)
    }


def identity_gradient_smoke(model: H32RLIF, raw_batch: Dict, device: torch.device) -> Dict:
    batch = move_batch(raw_batch, device)
    identities = {}
    model.eval()
    with torch.no_grad():
        for condition in ("clean", *CORRUPTIONS):
            view = batch if condition == "clean" else corruption_view(batch, condition)
            outputs = model(view)
            identities[condition] = float(
                (outputs["class_logits"] - outputs["b1_logits"]).abs().max().item()
            )
    if any(value != 0.0 for value in identities.values()):
        raise RuntimeError(f"H32 failed exact B1 identity: {identities}")
    model.train()
    clean = model(batch)
    corrupt = model(corruption_view(batch, "missing_image"))
    items = loss_items(clean, corrupt, batch["labels"])
    items["loss"].backward()
    trainable_gradient = 0.0
    frozen_nonzero = []
    for name, parameter in model.named_parameters():
        value = 0.0 if parameter.grad is None else float(parameter.grad.abs().sum().item())
        if parameter.requires_grad:
            trainable_gradient += value
        elif value > 0:
            frozen_nonzero.append(name)
    model.zero_grad(set_to_none=True)
    if trainable_gradient <= 0 or frozen_nonzero:
        raise RuntimeError(
            f"H32 gradient isolation failed: trainable={trainable_gradient}, frozen={frozen_nonzero[:5]}"
        )
    return {
        "step_zero_max_logit_difference": identities,
        "trainable_gradient_l1": trainable_gradient,
        "frozen_nonzero_gradient_count": 0,
    }


def parent_metrics(parent_checkpoint: Path) -> Dict:
    for name in ("development_val_metrics.json", "best_val_metrics.json"):
        path = parent_checkpoint.parent / name
        if path.exists():
            return load_json(path)
    return {}


def write_history(path: Path, rows: Iterable[Mapping]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H32-RLIF validation-only trainer")
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
    protocol = load_json(PROTOCOL_PATH)
    authorized_seed_by_action = {
        "run_seed3141_validation_only_manually": 3141,
        "run_seed1729_validation_only_manually": 1729,
        "run_seed2718_validation_only_manually": 2718,
    }
    action = protocol["next_authorized_action"]
    if not args.smoke_only and authorized_seed_by_action.get(action) != args.seed:
        raise RuntimeError(
            f"H32 seed {args.seed} is not the single authorized formal run under action {action!r}"
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"H32 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = build_model(args.model_name, Path(args.parent_checkpoint), device)
    counts = parameter_counts(model)
    save_json(output_dir / "parameter_counts.json", counts)
    train_df, val_df, train_loader, val_loader, mismatch_audit = make_loaders(args)
    if mismatch_audit["train_same_class_count"] or mismatch_audit["validation_same_class_count"]:
        raise RuntimeError(f"Mismatch audit failed: {mismatch_audit}")
    smoke = identity_gradient_smoke(model, next(iter(train_loader)), device)
    smoke.update(
        {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "mismatch_audit": mismatch_audit,
            "test_split_loaded": False,
            "test_evaluated": False,
        }
    )
    save_json(output_dir / "smoke_report.json", smoke)
    if args.smoke_only:
        print(f"H32 smoke passed: {output_dir}")
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
        clean = evaluate_condition(model, val_loader, device, "clean")
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
                    "experiment_id": "H32-RLIF",
                    "seed": args.seed,
                    "epoch": epoch,
                    "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
                    "rlif_state_dict": h32_delta_state(model),
                    "clean_validation_metrics": clean,
                    "test_split_loaded": False,
                    "test_evaluated": False,
                },
                output_dir / "best_model.pt",
            )
            save_json(output_dir / "best_val_metrics.json", clean)
        else:
            stale += 1
        if stale >= 4:
            break
    write_history(output_dir / "history.csv", history)
    best = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    load_h32_delta(model, best["rlif_state_dict"])
    conditions = evaluate_all_conditions(model, val_loader, device)
    parent_clean = parent_metrics(Path(args.parent_checkpoint))
    clean_deltas = {
        key: float(conditions["clean"]["h32"][key]) - float(parent_clean[key])
        for key in ("accuracy", "macro_f1", "weighted_f1")
    }
    corrupted_accuracy_deltas = {
        name: float(conditions[name]["deltas"]["accuracy"]) for name in CORRUPTIONS
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
        "experiment_id": "H32-RLIF",
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
        "experiment_id": "H32-RLIF",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "parent_clean_metrics": parent_clean,
        "conditions": conditions,
        "clean_deltas_vs_b1": clean_deltas,
        "corrupted_accuracy_deltas_vs_b1": corrupted_accuracy_deltas,
        "mean_corrupted_accuracy_delta_vs_b1": mean_corrupted_delta,
        "first_seed_gate_checks": checks,
        "first_seed_gate_pass": all(checks.values()),
        "test_split_loaded": False,
        "test_evaluated": False,
    }
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "development_complete.json", completion)
    print(f"H32 validation run complete: {output_dir}")


if __name__ == "__main__":
    main()
