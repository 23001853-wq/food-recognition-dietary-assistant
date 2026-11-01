"""Fast POC contract audit; no training, image transforms or full-image checksum sweep."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


def audit(directory, classes_path):
    directory = Path(directory)
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8"))
    classes = yaml.safe_load(Path(classes_path).read_text(encoding="utf-8"))
    mapping = {i: key for i, key in enumerate(classes["classes"])}
    if (
        len(mapping) != 10
        or len(set(mapping.values())) != 10
        or labels["dataset_version"] != metadata["dataset_version"]
        or labels["dataset_version"] != classes["version"]
    ):
        raise ValueError("POC requires 10 classes and consistent dataset versions")
    if {x["class_id"]: x["class_key"] for x in labels["labels"]} != mapping:
        raise ValueError("Frozen class order differs from POC config")
    if [x["class_id"] for x in labels["labels"]] != list(range(10)):
        raise ValueError("labels.json order is not fixed 0..9")
    with (directory / "dataset_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_split = {key: [] for key in ["train", "val", "test"]}
    hashes, paths = {}, set()
    root = Path(metadata["root"])
    official = {
        s: set((root / f"meta/{s}.txt").read_text().splitlines()) for s in ["train", "test"]
    }
    for row in rows:
        key, split = row["class_key"], row["split"]
        if mapping.get(int(row["class_id"])) != key or split not in by_split:
            raise ValueError("Invalid row class/split")
        if (
            row["dataset_version"] != labels["dataset_version"]
            or row["validation_status"] != "valid"
        ):
            raise ValueError("Invalid row version/validation_status")
        path = row["relative_path"]
        absolute = (root / path).resolve()
        if path in paths or not absolute.is_relative_to(root.resolve()) or not absolute.is_file():
            raise ValueError("Repeated, unsafe or missing image path")
        paths.add(path)
        image_key = str(Path(path).relative_to("images").with_suffix("")).replace("\\", "/")
        if image_key not in official["test" if split == "test" else "train"]:
            raise ValueError("A row violates the original Food-101 train/test boundary")
        if hashes.setdefault(row["checksum"], split) != split:
            raise ValueError("Cross-split hash leakage")
        by_split[split].append(row)
    for split, count in [("train", 600), ("val", 150), ("test", 250)]:
        if Counter(r["class_key"] for r in by_split[split]) != Counter(
            {k: count for k in mapping.values()}
        ):
            raise ValueError(f"Unexpected class counts in {split}")
        with (directory / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            if list(csv.DictReader(handle)) != by_split[split]:
                raise ValueError(f"{split}.csv differs from the shared manifest")
    expected_test = {key for key in official["test"] if key.split("/")[0] in mapping.values()}
    if {row["image_id"] for row in by_split["test"]} != expected_test:
        raise ValueError("POC official test set is incomplete")
    return {
        "passed": True,
        "class_count": 10,
        "counts": {k: len(v) for k, v in by_split.items()},
        "manifest_sha256": hashlib.sha256(
            (directory / "dataset_manifest.csv").read_bytes()
        ).hexdigest(),
        "official_test_preserved": True,
        "class_mapping_fixed": True,
        "full_image_hash_decode_rechecked_this_run": False,
        "deep_audit_command": "python scripts/verify_dataset.py --manifest-dir data/splits/v1",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes_poc.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest_dir, args.classes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
