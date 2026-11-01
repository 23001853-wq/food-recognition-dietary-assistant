"""Synthetic image fixtures exercise code; never evidence of food recognition quality."""

import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from food_assistant.ml.datasets import FIELDS, audit_manifest, write_json
from food_assistant.services.model_registry import Release


@pytest.fixture
def synthetic_dataset(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    rng = np.random.default_rng(42)
    labels = [
        {
            "class_id": i,
            "class_key": f"synthetic_{i}",
            "display_name_en": f"Synthetic {i}",
            "display_name_vi": f"Synthetic {i}",
        }
        for i in range(3)
    ]
    rows = []
    for i in range(3):
        for split in ["train", "val", "test"]:
            for j in range(2):
                path = root / f"{i}_{split}_{j}.png"
                image = rng.integers(0, 100, (64, 64, 3), dtype=np.uint8)
                image[:, :, i] += 100
                Image.fromarray(image).save(path)
                rows.append(
                    {
                        "image_id": path.stem,
                        "relative_path": path.name,
                        "source_dataset": "synthetic-test",
                        "class_id": i,
                        "class_key": f"synthetic_{i}",
                        "split": split,
                        "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "validation_status": "valid",
                        "dataset_version": "synthetic-v1",
                        "capture_group": "",
                    }
                )
    with (manifest / "dataset_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        manifest / "labels.json",
        {"schema_version": "1.0", "dataset_version": "synthetic-v1", "labels": labels},
    )
    write_json(
        manifest / "dataset_metadata.json",
        {
            "root": str(root),
            "dataset_version": "synthetic-v1",
            "source": "synthetic test fixture, not food images",
            "is_synthetic": True,
        },
    )
    return manifest, root, rows


def test_detect_cross_split_duplicates(synthetic_dataset):
    manifest, root, rows = synthetic_dataset
    assert audit_manifest(rows, root)["image_count"] == 18
    rows[-1]["relative_path"] = rows[0]["relative_path"]
    rows[-1]["checksum"] = rows[0]["checksum"]
    with pytest.raises(ValueError):
        audit_manifest(rows, root)


@pytest.mark.ml
@pytest.mark.parametrize("name", ["hog_svm", "resnet18", "efficientnet_b0"])
def test_train_load_and_evaluate_smoke(name, synthetic_dataset, tmp_path):
    manifest, root, rows = synthetic_dataset
    command = [
        sys.executable,
        f"scripts/train_{name}.py",
        "--manifest-dir",
        str(manifest),
        "--output-root",
        str(tmp_path / "models"),
        "--run-id",
        "synthetic-smoke",
        "--smoke",
    ]
    if name != "hog_svm":
        command += ["--no-pretrained", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    artifact = tmp_path / "models" / name / "synthetic-smoke"
    release = Release(artifact)
    if name != "hog_svm":
        import torch

        resume = torch.load(artifact / "resume.pth", map_location="cpu", weights_only=True)
        assert resume["best_metric"]["macro_f1"] >= 0
        assert resume["best_model"]
    assert release.config["smoke_only"] and release.config["trained"]
    predictions = release.predict(Image.new("RGB", (64, 64)), 3)
    assert len(predictions) == 3
    assert [p.class_id for p in predictions] == [
        p.class_id for p in sorted(predictions, key=lambda p: -p.score)
    ]
    evaluate = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_models.py",
            "--artifact",
            str(artifact),
            "--manifest-dir",
            str(manifest),
            "--split",
            "val",
            "--smoke",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr
    metrics = json.loads((artifact / "evaluation_smoke/metrics.json").read_text())
    assert metrics["smoke_only"] and metrics["sample_count"] == 6
    (artifact / "labels.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        Release(artifact)
