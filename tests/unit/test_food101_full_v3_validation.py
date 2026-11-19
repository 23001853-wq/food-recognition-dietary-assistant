"""Synthetic integrity and leakage tests for full-v3 validation helpers."""

import csv
import hashlib
import json
import shutil

import pytest
from PIL import Image

from food_assistant.data_pipeline.validate_v3 import (
    classify_scope,
    inspect_v3_rows,
    write_validation_report,
)
from food_assistant.ml.datasets import FIELDS, ManifestDataset
from food_assistant.ml.transforms import build_transform, preprocessing


def test_poc_sized_scope_cannot_be_full_scope_pass():
    """Classify a strict subset as incomplete when the requested scope is full."""
    expected = [f"class_{index}" for index in range(101)]
    assert classify_scope(expected[:10], expected) == "INCOMPLETE"
    assert classify_scope(expected, expected) == "PASS"


def _row(tmp_path, image_id, split, class_key, content, capture_group=""):
    """Create one small real image and a manifest record bound to its bytes."""
    path = tmp_path / "images" / f"{image_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=content).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        {
            "image_id": image_id,
            "relative_path": f"images/{image_id}.jpg",
            "class_id": "0" if class_key == "dish_a" else "1",
            "class_key": class_key,
            "split": split,
            "checksum": digest,
            "dataset_version": "v3",
            "capture_group": capture_group,
        },
        f"images/{image_id}.jpg",
        digest,
    )


def _small_fixture(tmp_path):
    """Build a balanced two-class, three-partition fixture with independent hashes."""
    rows = []
    hashes = {}
    for class_key in ("dish_a", "dish_b"):
        class_id = class_key.split("_")[-1]
        for split in ("train", "val", "test"):
            image_id = f"{class_key}/{split}"
            row, relative, digest = _row(
                tmp_path,
                image_id,
                split,
                class_key,
                (len(split) * 19, ord(class_id) % 255, 123),
            )
            rows.append(row)
            hashes[relative] = digest
    return rows, hashes


def test_small_full_v3_fixture_passes_row_gate(tmp_path):
    """Accept a complete fixture with valid mappings and independent partitions."""
    rows, hashes = _small_fixture(tmp_path)
    result = inspect_v3_rows(rows, ["dish_a", "dish_b"], tmp_path, hashes)
    assert result["status"] == "PASS"
    assert result["split_counts"] == {"train": 2, "val": 2, "test": 2}
    assert result["cross_partition_exact_hash_count"] == 0


def test_v3_manifest_contract_reads_through_manifest_dataset_and_dataloader(tmp_path):
    """Read a synthetic v3 bundle with the production dataset and torchvision pipeline."""
    import torch
    from torch.utils.data import DataLoader

    rows, _ = _small_fixture(tmp_path)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    with (manifest_dir / "dataset_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    labels = [
        {"class_id": index, "class_key": name}
        for index, name in enumerate(("dish_a", "dish_b"))
    ]
    (manifest_dir / "labels.json").write_text(
        json.dumps({"schema_version": "3.0", "dataset_version": "v3", "labels": labels}),
        encoding="utf-8",
    )
    (manifest_dir / "dataset_metadata.json").write_text(
        json.dumps({"dataset_version": "v3", "root": ".."}), encoding="utf-8"
    )
    dataset = ManifestDataset(
        manifest_dir,
        "train",
        transform=build_transform(preprocessing("resnet18")),
        root_override=tmp_path,
    )
    images, target = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert images.shape == (2, 3, 224, 224)
    assert images.dtype == torch.float32
    assert target.tolist() == [0, 1]


def test_cross_partition_hash_and_capture_group_fail(tmp_path):
    """Detect exact hash and capture-group leakage across train and test."""
    rows, hashes = _small_fixture(tmp_path)
    rows[0]["capture_group"] = "capture-x"
    rows[-1]["capture_group"] = "capture-x"
    rows[-1]["checksum"] = rows[0]["checksum"]
    hashes[rows[-1]["relative_path"]] = rows[0]["checksum"]
    result = inspect_v3_rows(rows, ["dish_a", "dish_b"], tmp_path, hashes)
    assert result["status"] == "FAIL"
    assert result["cross_partition_exact_hash_count"] == 1
    assert result["cross_partition_capture_group_count"] == 1


def test_within_partition_duplicate_is_reported_without_automatic_removal(tmp_path):
    """Report a same-split duplicate while retaining both source manifest rows."""
    rows, hashes = _small_fixture(tmp_path)
    duplicate = dict(
        rows[0],
        image_id="dish_a/train-copy",
        relative_path="images/dish_a/train-copy.jpg",
    )
    duplicate_path = tmp_path / duplicate["relative_path"]
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_path / rows[0]["relative_path"], duplicate_path)
    duplicate["checksum"] = rows[0]["checksum"]
    rows.append(duplicate)
    hashes[duplicate["relative_path"]] = duplicate["checksum"]
    result = inspect_v3_rows(rows, ["dish_a", "dish_b"], tmp_path, hashes)
    assert result["same_split_duplicate_group_count"] == 1
    assert result["image_count"] == len(rows)
    assert result["status"] == "PASS"


def test_report_writer_refuses_to_overwrite(tmp_path):
    """Keep prior validation reports immutable."""
    report = tmp_path / "artifacts" / "reports" / "elt" / "result.md"
    report.parent.mkdir(parents=True)
    report.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_validation_report({"status": "PASS"}, report)
    assert report.read_text(encoding="utf-8") == "existing"
