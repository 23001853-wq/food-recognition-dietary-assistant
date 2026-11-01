import json

import pytest

from food_assistant.ml.protocol import digest, freeze_poc, verify_frozen_candidate


@pytest.fixture
def releases(tmp_path):
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    (manifest / "dataset_manifest.csv").write_text("synthetic protocol fixture\n")
    labels = {
        "dataset_version": "poc-v1",
        "labels": [{"class_id": i, "class_key": str(i)} for i in range(10)],
    }
    (manifest / "labels.json").write_text(json.dumps(labels))
    artifacts = []
    for name in ["hog_svm", "resnet18", "efficientnet_b0"]:
        root = tmp_path / name
        root.mkdir()
        config = {
            "model_name": name,
            "run_id": "fixture",
            "trained": True,
            "smoke_only": False,
            "manifest_sha256": digest(manifest / "dataset_manifest.csv"),
            "num_classes": 10,
            "dataset_version": "poc-v1",
            "training_samples": 6000,
            "validation_samples": 1500,
        }
        (root / "model_config.json").write_text(json.dumps(config))
        (root / "labels.json").write_text(json.dumps(labels))
        (root / "metrics_val.json").write_text('{"macro_f1": 0.5}')
        (root / "preprocessing.json").write_text("{}")
        (root / "model.pth").write_bytes(b"synthetic unit fixture, not model weights")
        checks = {p.name: digest(p) for p in root.iterdir()}
        (root / "checksum.sha256").write_text(json.dumps(checks))
        artifacts.append(root)
    return manifest, artifacts, tmp_path / "protocol.json"


def test_freeze_and_detect_checkpoint_change(releases):
    manifest, artifacts, output = releases
    freeze_poc(artifacts, manifest, output)
    assert verify_frozen_candidate(output, artifacts[0], "cpu")["test_samples"] == 2500
    (artifacts[0] / "model.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after freeze"):
        verify_frozen_candidate(output, artifacts[0], "cpu")


def test_frozen_protocol_cannot_be_overwritten(releases):
    manifest, artifacts, output = releases
    freeze_poc(artifacts, manifest, output)
    with pytest.raises(ValueError, match="already exists"):
        freeze_poc(artifacts, manifest, output)
    with pytest.raises(ValueError, match="same declared CPU"):
        verify_frozen_candidate(output, artifacts[0], "cuda")


def test_cannot_freeze_after_test_started(releases):
    manifest, artifacts, output = releases
    (artifacts[0] / "evaluation_test").mkdir()
    with pytest.raises(ValueError, match="retrospectively"):
        freeze_poc(artifacts, manifest, output)
