"""Synthetic full-v3 selection, freeze, and final-test gate contracts."""

import hashlib
import json
import shutil
from copy import deepcopy

import pytest

from food_assistant.ml.full_v3_release import (
    DATASET_BINDING,
    PROTOCOL_SHA256,
    claim_final_test_attempt,
    complete_final_test_attempt,
    freeze_selected_record,
    validate_candidate,
    validate_final_test_request,
    verify_freeze_record,
    write_selection_record,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def candidate(tmp_path):
    """Build a synthetic artifact bound to the read-only approved v3 labels."""
    source = __import__("pathlib").Path("data/splits/v3/labels.json")
    protocol = __import__("pathlib").Path("docs/ml/full_v3_101class_training_protocol_v1.md")
    labels_path = tmp_path / "labels.json"
    shutil.copyfile(source, labels_path)
    labels_doc = json.loads(labels_path.read_text(encoding="utf-8"))
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    config = {
        **DATASET_BINDING,
        "model_name": "synthetic_model",
        "run_id": "synthetic-run",
        "trained": True,
        "smoke_only": False,
    }
    validation = {
        "split": "val",
        "smoke_only": False,
        "sample_count": 101,
        **{
            key: DATASET_BINDING[key]
            for key in (
                "manifest_sha256",
                "labels_sha256",
                "dataset_version",
                "num_classes",
            )
        },
        "protocol_sha256": PROTOCOL_SHA256,
        "device": "cpu",
        "batch_size": 1,
        "warmup_iterations": 3,
        "latency_scope": "image preprocessing plus inference; excludes file decode",
        "macro_f1": 0.75,
        "top1_accuracy": 0.76,
        "p95_ms": 1.2,
    }
    (artifact / "labels.json").write_bytes(labels_path.read_bytes())
    (artifact / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    (artifact / "metrics_val.json").write_text(json.dumps(validation), encoding="utf-8")
    (artifact / "preprocessing.json").write_text(
        json.dumps({"profile": "fixture"}), encoding="utf-8"
    )
    (artifact / "train_config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (artifact / "source_commit.txt").write_text("synthetic-fixture\n", encoding="utf-8")
    (artifact / "source_state.json").write_text("{}\n", encoding="utf-8")
    (artifact / "weights.bin").write_bytes(b"synthetic-only")
    inventory = {
        path.name: _sha(path)
        for path in artifact.iterdir()
        if path.name != "checksum.sha256"
    }
    (artifact / "checksum.sha256").write_text(json.dumps(inventory), encoding="utf-8")
    return {
        "artifact_dir": artifact,
        "config": config,
        "val": validation,
        "labels": labels_doc["labels"],
        "protocol": protocol,
        "labels_path": labels_path,
    }


def test_freeze_is_immutable_idempotent_and_never_writes_pointer(candidate, tmp_path):
    """Freeze valid evidence once and preserve a separate production pointer."""
    pointer = tmp_path / "current.json"
    pointer.write_text('{"existing":true}\n', encoding="utf-8")
    target = tmp_path / "freeze.json"
    candidates, selection_path = _selection(candidate, tmp_path)
    first = freeze_selected_record(
        candidates, selection_path, candidate["protocol"], target
    )
    second = freeze_selected_record(
        candidates, selection_path, candidate["protocol"], target
    )
    assert first["status"] == "CREATED"
    assert second["status"] == "SKIPPED_IDENTICAL"
    assert verify_freeze_record(target)["dataset"]["num_classes"] == 101
    assert pointer.read_text(encoding="utf-8") == '{"existing":true}\n'
    validate_final_test_request(target, candidate["artifact_dir"], candidate["protocol"])
    evidence = tmp_path / "test-evidence"
    marker = claim_final_test_attempt(
        target,
        evidence,
        candidate["artifact_dir"],
        candidate["protocol"],
        attempt_root=tmp_path / "attempts",
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        claim_final_test_attempt(
            target,
            tmp_path / "another-evidence",
            candidate["artifact_dir"],
            candidate["protocol"],
            attempt_root=tmp_path / "attempts",
        )
    complete_final_test_attempt(marker, evidence, target)
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "COMPLETED"


@pytest.mark.parametrize(
    "mutation", ["artifact", "classes", "fingerprint", "protocol", "test_metric"]
)
def test_freeze_rejects_invalid_candidate_evidence(candidate, tmp_path, mutation):
    """Reject missing/corrupt or protocol-incompatible candidate evidence."""
    protocol = candidate["protocol"]
    if mutation == "artifact":
        (candidate["artifact_dir"] / "weights.bin").write_bytes(b"changed")
    elif mutation == "classes":
        candidate["config"]["num_classes"] = 100
    elif mutation == "fingerprint":
        candidate["config"]["manifest_sha256"] = "0" * 64
    elif mutation == "protocol":
        protocol = tmp_path / "wrong_protocol.md"
        protocol.write_text("not the frozen protocol", encoding="utf-8")
    else:
        candidate["val"]["test_metrics"] = {"macro_f1": 1.0}
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_candidate(candidate, protocol)


def test_final_test_gate_rejects_other_candidate(candidate, tmp_path):
    """A frozen candidate cannot authorize test evaluation of another artifact."""
    target = tmp_path / "freeze.json"
    candidates, selection_path = _selection(candidate, tmp_path)
    freeze_selected_record(candidates, selection_path, candidate["protocol"], target)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="only for the frozen candidate"):
        validate_final_test_request(target, other, candidate["protocol"])


def test_conflicting_freeze_target_is_never_overwritten(candidate, tmp_path):
    """An existing freeze target with different bytes remains untouched."""
    target = tmp_path / "freeze.json"
    target.write_text('{"not":"a freeze"}', encoding="utf-8")
    candidates, selection_path = _selection(candidate, tmp_path)
    with pytest.raises(ValueError):
        freeze_selected_record(candidates, selection_path, candidate["protocol"], target)
    assert target.read_text(encoding="utf-8") == '{"not":"a freeze"}'


def test_synthetic_validation_select_freeze_test_gate_workflow(candidate, tmp_path):
    """Run selection then freeze contracts on two isolated synthetic candidates."""
    candidates, selection_path = _selection(candidate, tmp_path)
    freeze_path = tmp_path / "freeze.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = freeze_selected_record(candidates, selection_path, candidate["protocol"], freeze_path)
    assert selection["selected_run_id"] == candidate["config"]["run_id"]
    assert result["status"] == "CREATED"
    validate_final_test_request(freeze_path, candidate["artifact_dir"], candidate["protocol"])


def test_corrupted_frozen_record_is_rejected(candidate, tmp_path):
    """Detect freeze-record edits before granting final-test eligibility."""
    target = tmp_path / "freeze.json"
    candidates, selection_path = _selection(candidate, tmp_path)
    freeze_selected_record(candidates, selection_path, candidate["protocol"], target)
    record = json.loads(target.read_text(encoding="utf-8"))
    record["candidate"]["run_id"] = "tampered"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        verify_freeze_record(target)


def _selection(candidate, tmp_path):
    """Create two isolated synthetic candidates and validation selection evidence."""
    second_dir = tmp_path / "candidate_b"
    shutil.copytree(candidate["artifact_dir"], second_dir)
    second = deepcopy(candidate)
    second["artifact_dir"] = second_dir
    second["config"]["model_name"] = "second_synthetic_model"
    second["config"]["run_id"] = "second-synthetic-run"
    (second_dir / "model_config.json").write_text(json.dumps(second["config"]), encoding="utf-8")
    second["val"]["macro_f1"] = 0.70
    (second_dir / "metrics_val.json").write_text(json.dumps(second["val"]), encoding="utf-8")
    inventory = {
        path.name: _sha(path)
        for path in second_dir.iterdir()
        if path.name != "checksum.sha256"
    }
    (second_dir / "checksum.sha256").write_text(json.dumps(inventory), encoding="utf-8")
    candidates = [candidate, second]
    selection_path = tmp_path / "selection.json"
    write_selection_record(candidates, candidate["protocol"], selection_path)
    return candidates, selection_path
