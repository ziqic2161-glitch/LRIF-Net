import argparse
import csv
import html
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {
    "train": {
        "affected_individuals": 81,
        "infrastructure_and_utility_damage": 497,
        "rescue_volunteering_or_donation_effort": 726,
        "other_relevant_information": 1164,
        "not_humanitarian": 2651,
    },
    "val": {
        "affected_individuals": 17,
        "infrastructure_and_utility_damage": 106,
        "rescue_volunteering_or_donation_effort": 156,
        "other_relevant_information": 250,
        "not_humanitarian": 568,
    },
    "test": {
        "affected_individuals": 17,
        "infrastructure_and_utility_damage": 107,
        "rescue_volunteering_or_donation_effort": 156,
        "other_relevant_information": 250,
        "not_humanitarian": 568,
    },
}

LABEL_ORDER = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(text or "")))
    text = re.sub(r"https?://\S+|www\.\S+", " link ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@\w+", " user ", text)
    text = re.sub(r"(?<!\w)#(\w+)", r" \1 ", text)
    text = re.sub(r"^\s*RT\s+", "", text, flags=re.IGNORECASE)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and convert the canonical 7,314-sample CrisisMMD Task 2 split."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_names = {
        "train": "task02_train.tsv",
        "val": "task02_dev.tsv",
        "test": "task02_test.tsv",
    }
    raw_output = args.output_root / "raw_splits"
    processed_output = args.output_root / "processed"
    raw_output.mkdir(parents=True, exist_ok=True)
    processed_output.mkdir(parents=True, exist_ok=True)

    seen_image_ids: set[str] = set()
    report = {"total": 0, "missing_images": 0, "splits": {}}

    for split, source_name in source_names.items():
        source_path = args.split_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, raw_output / source_name)
        source_rows = read_tsv(source_path)
        label_counts = Counter(row.get("label", "") for row in source_rows)
        if dict(label_counts) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{source_name} does not match the canonical class counts: {dict(label_counts)}"
            )

        converted = []
        for row in source_rows:
            label_text = row.get("label_text", "").strip()
            label_image = row.get("label_image", "").strip()
            label = row.get("label", "").strip()
            if not label_text or label_text != label_image or label != label_text:
                raise ValueError(f"Inconsistent text/image label for image_id={row.get('image_id')}")
            image_id = row.get("image_id", "").strip()
            if image_id in seen_image_ids:
                raise ValueError(f"Image leakage or duplicate across splits: {image_id}")
            seen_image_ids.add(image_id)

            image_path = args.image_root / Path(row.get("image_path", ""))
            image_exists = image_path.exists()
            report["missing_images"] += int(not image_exists)
            raw_text = row.get("tweet_text", "")
            converted.append(
                {
                    "sample_id": f"canonical_{split}_{image_id}",
                    "split": split,
                    "tweet_id": row.get("tweet_id", ""),
                    "image_id": image_id,
                    "tweet_text_raw": raw_text,
                    "tweet_text": clean_text(raw_text),
                    "image_rel_path": row.get("image_path", ""),
                    "image_path": str(image_path.resolve()),
                    "label_5class": label,
                    "label_id": LABEL_TO_ID[label],
                    "label_text": label_text,
                    "label_image": label_image,
                    "image_exists": image_exists,
                }
            )

        write_csv(processed_output / f"{split}.csv", converted)
        report["splits"][split] = {
            "rows": len(converted),
            "class_counts": dict(label_counts),
        }
        report["total"] += len(converted)

    if report["total"] != 7314 or report["missing_images"] != 0:
        raise ValueError(f"Dataset validation failed: {report}")
    with (args.output_root / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Validated {report['total']} label-consistent samples with no split overlap.")
    print(f"Missing images: {report['missing_images']}")
    print(f"Processed splits: {processed_output}")


if __name__ == "__main__":
    main()
