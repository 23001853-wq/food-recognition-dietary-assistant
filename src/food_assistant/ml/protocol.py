"""Freeze all POC candidates before reading official test outcomes."""

import hashlib
import json
from pathlib import Path

METHODS = {"hog_svm", "resnet18", "efficientnet_b0"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze_poc(artifacts, manifest_dir, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Frozen protocol already exists; do not overwrite it")
    manifest_dir = Path(manifest_dir)
    manifest_hash = digest(manifest_dir / "dataset_manifest.csv")
    labels = json.loads((manifest_dir / "labels.json").read_text(encoding="utf-8"))
    if labels["dataset_version"] != "poc-v1" or len(labels["labels"]) != 10:
        raise ValueError("This protocol is restricted to POC v1 with 10 classes")
    candidates = []
    for artifact in map(Path, artifacts):
        config = json.loads((artifact / "model_config.json").read_text(encoding="utf-8"))
        if not config.get("trained") or config.get("smoke_only"):
            raise ValueError(f"Incomplete or smoke-only candidate: {artifact}")
        if (
            config.get("num_classes") != 10
            or config.get("dataset_version") != "poc-v1"
            or config.get("training_samples") != 6000
            or config.get("validation_samples") != 1500
        ):
            raise ValueError(
                "POC release requires all 6000 train / 1500 val samples and 10 classes"
            )
        if (artifact / "metrics_test.json").exists() or (artifact / "evaluation_test").exists():
            raise ValueError(
                "Test already started; cannot retrospectively declare a frozen protocol"
            )
        if config["manifest_sha256"] != manifest_hash:
            raise ValueError("Candidate manifest differs from current POC manifest")
        if json.loads((artifact / "labels.json").read_text(encoding="utf-8")) != labels:
            raise ValueError("Candidate label map differs from frozen POC labels")
        files = json.loads((artifact / "checksum.sha256").read_text(encoding="utf-8"))
        if (
            not {"model_config.json", "metrics_val.json", "preprocessing.json", "labels.json"}
            <= files.keys()
        ):
            raise ValueError("Missing required release checksums")
        if not {"model.pth", "model.joblib"} & files.keys():
            raise ValueError("Missing model checkpoint checksum")
        for name, expected in files.items():
            path = (artifact / name).resolve()
            if not path.is_relative_to(artifact.resolve()) or digest(path) != expected:
                raise ValueError("Release checksum failed")
        candidates.append(
            {
                "model_name": config["model_name"],
                "run_id": config["run_id"],
                "artifact": str(artifact.resolve()),
                "release_files": files,
            }
        )
    if len(candidates) != 3 or {x["model_name"] for x in candidates} != METHODS:
        raise ValueError("Freeze exactly HOG/SVM, ResNet18 and EfficientNet-B0")
    result = {
        "schema_version": "1.0",
        "dataset_version": "poc-v1",
        "num_classes": 10,
        "manifest_sha256": manifest_hash,
        "test_samples": 2500,
        "evaluation_device": "cpu",
        "selection_metric": "validation_macro_f1",
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify_frozen_candidate(protocol_path, artifact, device):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if device != protocol["evaluation_device"]:
        raise ValueError("Use the same declared CPU inference protocol for all three methods")
    artifact = Path(artifact).resolve()
    candidate = next(
        (x for x in protocol["candidates"] if Path(x["artifact"]).resolve() == artifact), None
    )
    if candidate is None:
        raise ValueError("Artifact not included in the frozen comparison")
    for name, expected in candidate["release_files"].items():
        path = (artifact / name).resolve()
        if not path.is_relative_to(artifact) or digest(path) != expected:
            raise ValueError("Candidate changed after freeze; do not tune after test evaluation")
    return protocol
