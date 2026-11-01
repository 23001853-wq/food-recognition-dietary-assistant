"""Versioned dataset contracts; class indices are persisted and never re-sorted at training."""

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

FIELDS = [
    "image_id",
    "relative_path",
    "source_dataset",
    "class_id",
    "class_key",
    "split",
    "checksum",
    "validation_status",
    "dataset_version",
    "capture_group",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_manifest(rows, root):
    paths, hashes, mapping, groups = {}, {}, {}, {}
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        path = (Path(root) / row["relative_path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe image path")
        if row["relative_path"] in paths:
            raise ValueError("Repeated image path")
        paths[row["relative_path"]] = row["split"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["checksum"]:
            raise ValueError("Image checksum mismatch")
        previous = hashes.setdefault(digest, row["split"])
        if previous != row["split"]:
            raise ValueError("Exact duplicate crosses dataset splits; resolve explicitly")
        group = row.get("capture_group")
        if group and groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError("Capture group crosses dataset splits")
        key = int(row["class_id"])
        if mapping.setdefault(key, row["class_key"]) != row["class_key"]:
            raise ValueError("Inconsistent class mapping")
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Class indices must be contiguous")
    return {
        "image_count": len(rows),
        "class_count": len(mapping),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "per_class": dict(Counter(r["class_key"] for r in rows)),
        "exact_duplicates": len(rows) - len(hashes),
    }


def prepare_food101(root, output, classes, version="poc-v1", seed=42, val_fraction=0.2):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.glob("*.csv")):
        raise ValueError("Refusing to overwrite a frozen manifest; use a new output version")
    official_classes = (root / "meta/classes.txt").read_text().splitlines()
    if len(classes) != len(set(classes)) or not set(classes) <= set(official_classes):
        raise ValueError("Invalid class selection")
    mapping = {key: i for i, key in enumerate(classes)}
    rows = []
    for official in ("train", "test"):
        names = [
            p
            for p in (root / f"meta/{official}.txt").read_text().splitlines()
            if p.split("/")[0] in mapping
        ]
        if official == "train":
            train, val = train_test_split(
                names,
                test_size=val_fraction,
                random_state=seed,
                stratify=[x.split("/")[0] for x in names],
            )
            assignments = {name: "train" for name in train} | {name: "val" for name in val}
        else:
            assignments = {name: "test" for name in names}
        for name in names:
            key = name.split("/")[0]
            path = root / "images" / f"{name}.jpg"
            with Image.open(path) as image:
                image.verify()
            rows.append(
                dict(
                    zip(
                        FIELDS,
                        [
                            name,
                            f"images/{name}.jpg",
                            "food101",
                            mapping[key],
                            key,
                            assignments[name],
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                            "valid",
                            version,
                            "",
                        ],
                    )
                )
            )
            if len(rows) % 1000 == 0:
                print(f"Validated {len(rows)} selected Food-101 images", flush=True)
    stats = audit_manifest(rows, root)
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in [("dataset_manifest", rows)] + [
        (s, [r for r in rows if r["split"] == s]) for s in ("train", "val", "test")
    ]:
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)
    write_json(
        output / "labels.json",
        {
            "schema_version": "1.0",
            "dataset_version": version,
            "labels": [
                {
                    "class_id": i,
                    "class_key": key,
                    "display_name_vi": key,
                    "display_name_en": key.replace("_", " "),
                }
                for key, i in mapping.items()
            ],
        },
    )
    write_json(
        output / "dataset_metadata.json",
        {
            **stats,
            "dataset_version": version,
            "root": str(root.resolve()),
            "seed": seed,
            "validation_fraction": val_fraction,
            "method": "stratified original train; official test unchanged",
            "prepared_at": datetime.now(UTC).isoformat(),
            "source": "Food-101 official ETH release",
        },
    )
    return stats


class ManifestDataset:
    def __init__(self, manifest_dir, split, transform=None, limit_per_class=None):
        directory = Path(manifest_dir)
        self.metadata = read_json(directory / "dataset_metadata.json")
        self.labels = read_json(directory / "labels.json")
        self.root = Path(self.metadata["root"])
        with (directory / "dataset_manifest.csv").open(encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        audit_manifest(all_rows, self.root)
        labels = self.labels["labels"]
        if [x["class_id"] for x in labels] != list(range(len(labels))):
            raise ValueError("Invalid labels.json order")
        mapping = {x["class_id"]: x["class_key"] for x in labels}
        if self.labels["dataset_version"] != self.metadata["dataset_version"]:
            raise ValueError("Dataset version mismatch")
        if any(mapping.get(int(r["class_id"])) != r["class_key"] for r in all_rows):
            raise ValueError("Manifest and labels mapping mismatch")
        self.rows = [r for r in all_rows if r["split"] == split]
        if limit_per_class:
            counter = Counter()
            selected = []
            for row in self.rows:
                if counter[row["class_id"]] < limit_per_class:
                    selected.append(row)
                    counter[row["class_id"]] += 1
            self.rows = selected
        if not self.rows:
            raise ValueError(f"Empty {split} split")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            value = self.transform(image) if self.transform else np.asarray(image)
        return value, int(row["class_id"])
