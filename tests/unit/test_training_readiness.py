"""Read-only training preflight tests on isolated synthetic manifests."""

import csv
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from food_assistant.ml.datasets import FIELDS
from food_assistant.ml.metrics import top_k_accuracy_from_scores
from food_assistant.ml.train_utils import (
    _validate_training_binding,
    _validate_training_dry_run,
    estimate_hog_peak_bytes,
    read_resume_contract,
)


@pytest.fixture
def readiness_manifest(tmp_path):
    """Create a tiny, three-class manifest that never touches repository data."""
    root = tmp_path / "images"
    manifest_dir = tmp_path / "manifest"
    root.mkdir()
    manifest_dir.mkdir()
    labels = [
        {"class_id": index, "class_key": f"synthetic_{index}"}
        for index in range(3)
    ]
    rows = []
    for class_row in labels:
        for split in ("train", "val", "test"):
            image_id = f"{class_row['class_key']}_{split}"
            image_path = root / f"{image_id}.png"
            pixels = np.full((32, 32, 3), class_row["class_id"] * 40, dtype=np.uint8)
            pixels[0, 0, 0] = (0 if split == "train" else 40 if split == "val" else 80)
            Image.fromarray(pixels).save(image_path)
            rows.append(
                {
                    "image_id": image_id,
                    "relative_path": image_path.name,
                    "source_dataset": "synthetic-test",
                    "class_id": class_row["class_id"],
                    "class_key": class_row["class_key"],
                    "split": split,
                    "checksum": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "validation_status": "valid",
                    "dataset_version": "synthetic-v1",
                    "capture_group": "",
                }
            )
    with (manifest_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (manifest_dir / "labels.json").write_text(
        json.dumps({"dataset_version": "synthetic-v1", "labels": labels}),
        encoding="utf-8",
    )
    (manifest_dir / "dataset_metadata.json").write_text(
        json.dumps({"dataset_version": "synthetic-v1", "root": str(root)}),
        encoding="utf-8",
    )
    return manifest_dir


def _hog_fixture_config(manifest_dir):
    """Build a small HOG config bound to the supplied synthetic manifest."""
    return {
        "model_name": "hog_svm",
        "manifest_dir": str(manifest_dir),
        "expected_dataset_version": "synthetic-v1",
        "expected_num_classes": 3,
        "image_size": [32, 32],
        "hog": {
            "orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
        },
    }


def test_training_binding_checks_manifest_fingerprint(readiness_manifest):
    """Reject a changed manifest digest before training can acquire output state."""
    config = _hog_fixture_config(readiness_manifest)
    config["expected_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        _validate_training_binding("hog_svm", config)


def test_top5_accuracy_uses_ranked_class_scores():
    """Measure top-k correctness without changing model training or class order."""
    scores = np.array([[0.9, 0.05, 0.03, 0.01, 0.01], [0.1, 0.2, 0.3, 0.25, 0.15]])
    assert top_k_accuracy_from_scores([0, 4], scores, 3) == 0.5
    with pytest.raises(ValueError, match="incompatible"):
        top_k_accuracy_from_scores([0], scores, 6)


def test_training_dry_run_cli_is_read_only(readiness_manifest, tmp_path):
    """Validate a synthetic manifest without creating locks or model artifacts."""
    config_path = tmp_path / "hog_dry_run.yaml"
    config_path.write_text(
        yaml.safe_dump(_hog_fixture_config(readiness_manifest)), encoding="utf-8"
    )
    output_root = tmp_path / "models"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_hog_svm.py",
            "--config",
            str(config_path),
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY_RUN PASS" in result.stdout
    assert "output_shape" in result.stdout
    assert not output_root.exists()


def test_training_dry_run_rejects_duplicate_content(readiness_manifest):
    """Reject an exact checksum collision in the manifest before model setup."""
    manifest_path = readiness_manifest / "dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["checksum"] = rows[0]["checksum"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="exact duplicate"):
        _validate_training_dry_run("hog_svm", _hog_fixture_config(readiness_manifest))


def test_resume_rejects_changed_training_configuration(tmp_path):
    """Do not resume optimizer state under a changed batch size or training setup."""
    checkpoint_dir = tmp_path / "run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_config.json").write_text(
        json.dumps({"model_name": "resnet18", "smoke_only": False}), encoding="utf-8"
    )
    labels = {"dataset_version": "synthetic-v1", "labels": []}
    (checkpoint_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (checkpoint_dir / "preprocessing.json").write_text("{}", encoding="utf-8")
    previous = {"head_epochs": 2, "fine_tune_epochs": 3, "batch_size": 8, "seed": 42}
    (checkpoint_dir / "train_config.yaml").write_text(
        yaml.safe_dump(previous), encoding="utf-8"
    )
    changed = {**previous, "batch_size": 16}
    dataset = SimpleNamespace(labels=labels)
    with pytest.raises(ValueError, match="batch_size"):
        read_resume_contract(checkpoint_dir / "resume.pth", "resnet18", dataset, {}, changed)


def test_hog_memory_estimate_accounts_for_float32_and_float64_copies():
    """Estimate the dense estimator peak without allocating full-v3 matrices."""
    estimate = estimate_hog_peak_bytes(68_166, 7_574)
    assert estimate > 10 * 1024**3
    assert estimate_hog_peak_bytes(2, 1) < 1024**3 + 1_000_000

