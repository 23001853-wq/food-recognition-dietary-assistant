"""Synthetic tests for quality-gated Food-101 Transform behavior."""

import csv
import hashlib

import pytest
from PIL import Image

from food_assistant.data_pipeline.config import load_pipeline_config
from food_assistant.data_pipeline.transform import (
    TransformQualityError,
    _make_rows,
    _partition_train,
)
from food_assistant.ml.datasets import ManifestDataset, audit_manifest
from food_assistant.ml.transforms import build_transform, preprocessing


def test_transform_config_loads_without_side_effects(tmp_path):
    """Load the checked-in configuration without creating its output directories."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    output_existed = (root / "data/splits/poc-v2").exists()
    config = load_pipeline_config(root / "configs/food101_elt_poc_v2.yaml", root)
    assert config.dataset_scope == "poc"
    assert config.split_version == "poc-v2"
    assert config.manifest_root == root / "data/splits/poc-v2"
    assert config.manifest_root.exists() is output_existed
    assert tmp_path.exists()


def test_hash_groups_are_deterministic_and_never_split(tmp_path):
    """Keep same-content references together and reproduce partitions for a fixed seed."""
    train_ids = ["dish/a", "dish/b", "dish/c", "dish/d", "dish/e", "dish/f"]
    hashes = {
        "images/dish/a.jpg": "same-a",
        "images/dish/b.jpg": "same-a",
        "images/dish/c.jpg": "same-c",
        "images/dish/d.jpg": "same-d",
        "images/dish/e.jpg": "same-e",
        "images/dish/f.jpg": "same-f",
    }
    first = _partition_train(train_ids, hashes, 42, 0.2)
    second = _partition_train(train_ids, hashes, 42, 0.2)
    assert first == second
    assert first["dish/a"] == first["dish/b"]
    assert set(first.values()) == {"train", "val"}


def test_manifest_path_traversal_is_rejected(tmp_path):
    """Reject parent traversal before checking any path or checksum."""
    row = {
        "split": "train",
        "relative_path": "../outside.jpg",
        "checksum": "0" * 64,
        "class_id": 0,
        "class_key": "dish",
        "capture_group": "",
    }
    with pytest.raises(ValueError, match="safe root-relative"):
        audit_manifest([row], tmp_path)


def test_missing_image_and_wrong_checksum_fail_closed(tmp_path):
    """Reject absent source images and stale content digests."""
    import hashlib

    from PIL import Image

    image = tmp_path / "images" / "dish" / "one.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (24, 24)).save(image)
    row = {
        "split": "train", "relative_path": "images/dish/one.jpg", "checksum": "0" * 64,
        "class_id": 0, "class_key": "dish", "capture_group": "",
    }
    with pytest.raises(ValueError, match="checksum"):
        audit_manifest([row], tmp_path)
    row["checksum"] = hashlib.sha256(image.read_bytes()).hexdigest()
    row["relative_path"] = "images/dish/missing.jpg"
    with pytest.raises(ValueError, match="Missing"):
        audit_manifest([row], tmp_path)


def test_cross_split_exact_duplicate_fails_quality_audit(tmp_path):
    """Reject same-content train and test rows without deleting either reference."""
    train_image = tmp_path / "train.jpg"
    test_image = tmp_path / "test.jpg"
    train_image.write_bytes(b"same-image-content")
    test_image.write_bytes(b"same-image-content")
    import hashlib

    digest = hashlib.sha256(train_image.read_bytes()).hexdigest()
    rows = [
        {"split": split, "relative_path": f"{split}.jpg", "checksum": digest,
         "class_id": 0, "class_key": "dish", "capture_group": ""}
        for split in ("train", "test")
    ]
    with pytest.raises(ValueError, match="crosses dataset splits"):
        audit_manifest(rows, tmp_path)
    assert train_image.exists() and test_image.exists()


def test_split_builder_preserves_test_ids_and_keeps_train_hash_groups_together(tmp_path):
    """Preserve all official test IDs and assign repeated train hashes as one group."""
    source = tmp_path / "images"
    selected_train = ["dish/a", "dish/b", "dish/c", "dish/d", "dish/e", "dish/f"]
    selected_test = ["dish/test-a", "dish/test-b"]
    all_ids = selected_train + selected_test
    hashes = {}
    for index, image_id in enumerate(all_ids):
        path = source / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color=(index, 20, 40)).save(path)
        hashes[f"images/{image_id}.jpg"] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["images/dish/b.jpg"] = hashes["images/dish/a.jpg"]
    rows = _make_rows(tmp_path, ["dish"], selected_train, selected_test, hashes, "poc-v2", 42, 0.2)
    assert {row["image_id"] for row in rows if row["split"] == "test"} == set(selected_test)
    assert next(row["split"] for row in rows if row["image_id"] == "dish/a") == next(
        row["split"] for row in rows if row["image_id"] == "dish/b"
    )


def test_manifest_dataset_keeps_legacy_schema_and_torchvision_types(tmp_path):
    """Load a legacy-shaped manifest and use real torchvision tensor transforms."""
    import json

    import torch

    image_root = tmp_path / "images" / "dish"
    image_root.mkdir(parents=True)
    image_path = image_root / "one.jpg"
    Image.new("RGB", (256, 256), color=(80, 120, 160)).save(image_path)
    checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
    fieldnames = [
        "image_id", "relative_path", "source_dataset", "class_id", "class_key", "split",
        "checksum", "validation_status", "dataset_version", "capture_group",
    ]
    row = {
        "image_id": "dish/one", "relative_path": "images/dish/one.jpg",
        "source_dataset": "food101", "class_id": 0, "class_key": "dish", "split": "train",
        "checksum": checksum, "validation_status": "valid", "dataset_version": "poc-v1",
        "capture_group": "",
    }
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    with (manifest / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    (manifest / "labels.json").write_text(
        json.dumps({"schema_version": "1.0", "dataset_version": "poc-v1", "labels": [
            {"class_id": 0, "class_key": "dish"}
        ]}), encoding="utf-8"
    )
    (manifest / "dataset_metadata.json").write_text(
        json.dumps({"dataset_version": "poc-v1", "root": ".."}), encoding="utf-8"
    )
    dataset = ManifestDataset(
        manifest,
        "train",
        transform=build_transform(preprocessing("resnet18")),
        root_override=tmp_path,
    )
    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert label == 0


def test_hash_group_cannot_be_assigned_both_train_and_validation():
    """Fail closed for a class without enough independent hashes to split."""
    with pytest.raises(TransformQualityError, match="Not enough independent"):
        _partition_train(["dish/one", "dish/two"], {
            "images/dish/one.jpg": "same", "images/dish/two.jpg": "same"
        }, 42, 0.2)
