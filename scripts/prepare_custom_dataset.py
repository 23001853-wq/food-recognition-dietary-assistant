"""Prepare user-owned photos with explicit label order and capture-group isolation."""

import argparse
import csv
import hashlib
from pathlib import Path

import yaml
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from food_assistant.ml.datasets import FIELDS, audit_manifest, write_json


def prepare(root, catalog, class_config, output, seed=42):
    root, output = Path(root).resolve(), Path(output)
    config = yaml.safe_load(Path(class_config).read_text(encoding="utf-8"))
    if output.exists() and (output / "dataset_manifest.csv").exists():
        raise ValueError("Use a new dataset version; refusing to overwrite frozen manifest")
    keys = [x["class_key"] for x in config["labels"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated class key")
    rows = []
    with Path(catalog).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = (root / row["relative_path"]).resolve()
            if not path.is_relative_to(root) or row["class_key"] not in keys:
                raise ValueError("Unsafe path or unsupported class")
            if (
                not row.get("capture_group")
                or not row.get("source_url")
                or row.get("rights_confirmed") != "true"
            ):
                raise ValueError(
                    "Each photo requires capture_group, source_url and rights_confirmed=true"
                )
            with Image.open(path) as image:
                image.verify()
            rows.append(
                {
                    "image_id": path.stem,
                    "relative_path": row["relative_path"],
                    "source_dataset": config["dataset_name"],
                    "class_id": keys.index(row["class_key"]),
                    "class_key": row["class_key"],
                    "split": "",
                    "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "validation_status": "valid",
                    "dataset_version": config["dataset_version"],
                    "capture_group": row["capture_group"],
                }
            )
    groups = [r["capture_group"] for r in rows]
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val, test = next(first.split(rows, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_indices, val_indices = next(
        second.split(train_val, groups=[groups[i] for i in train_val])
    )
    for indices, split in [
        (test, "test"),
        ([train_val[i] for i in train_indices], "train"),
        ([train_val[i] for i in val_indices], "val"),
    ]:
        for index in indices:
            rows[index]["split"] = split
    for split in ("train", "val", "test"):
        if {r["class_key"] for r in rows if r["split"] == split} != set(keys):
            raise ValueError(
                "A split lacks classes; collect more capture groups. "
                "Do not silently reassign photos."
            )
    stats = audit_manifest(rows, root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    labels = [{"class_id": i, **label} for i, label in enumerate(config["labels"])]
    write_json(
        output / "labels.json",
        {"schema_version": "1.0", "dataset_version": config["dataset_version"], "labels": labels},
    )
    write_json(
        output / "dataset_metadata.json",
        {
            **stats,
            "root": str(root),
            "dataset_version": config["dataset_version"],
            "method": "capture-group-disjoint 60/20/20; explicit class coverage checks",
            "seed": seed,
            "source": str(Path(catalog).resolve()),
        },
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/classes_vietnamese.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root, args.catalog, args.classes, args.output))
