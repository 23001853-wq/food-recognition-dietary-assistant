"""Immutable selection and final-test isolation contracts for Food-101 v3."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import yaml

from food_assistant.config import PROJECT_ROOT
from food_assistant.data_pipeline.atomic import atomic_write_json, checkpoint_digest, resource_lock

PROTOCOL_SHA256 = "3108334d356452bfb2c735cada8f4dd586dfbaaf7ea6a3e2e99d20d2d31224ca"
DATASET_BINDING = {
    "dataset_version": "v3",
    "manifest_sha256": "f135b5492bc3767a5f18268838de67a5535004f8d361c3a659ff9946d4f9f3c4",
    "labels_sha256": "b80ce882769587fdaf2b1ed0534ccfffb1e6c70a8c7b6c3deac45ef167a0ae43",
    "source_fingerprint": "f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1",
    "inventory_sha256": "b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9",
    "v3_fingerprint": "aa43cfa766982006fe6a3611a49e0f07d35d33d56e54310e923cb5e58fbdbb59",
    "num_classes": 101,
}


def sha256_file(path):
    """Return a streamed SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol(path):
    """Verify that the frozen protocol file matches the approved v1.0 digest."""
    if sha256_file(path) != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol digest mismatch")
    return PROTOCOL_SHA256


def validate_candidate(candidate, protocol_path):
    """Validate trained artifact, validation provenance, and full-v3 bindings."""
    config = candidate["config"]
    validation = candidate["val"]
    if "test" in candidate or "metrics_test" in candidate or "test_metrics" in validation:
        raise ValueError("Test evidence is forbidden before candidate freeze")
    directory = Path(candidate["artifact_dir"]).resolve()
    if not directory.is_dir() or config.get("trained") is not True or config.get("smoke_only"):
        raise ValueError("Candidate must be a completed non-smoke training artifact")
    if any((directory / name).exists() for name in ("metrics_test.json", "evaluation_test")):
        raise ValueError("Candidate with test evaluation cannot enter selection/freeze")
    verify_protocol(protocol_path)
    source_dir = verify_v3_source()
    for key, value in DATASET_BINDING.items():
        if config.get(key) != value:
            raise ValueError(f"Candidate dataset binding mismatch: {key}")
    if validation.get("split") != "val" or validation.get("smoke_only"):
        raise ValueError("Candidate requires formal validation-only evidence")
    if not isinstance(validation.get("sample_count"), int) or validation["sample_count"] <= 0:
        raise ValueError("Validation evidence must include a positive sample count")
    for metric in ("macro_f1", "top1_accuracy", "p95_ms"):
        value = validation.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Validation evidence has invalid {metric}")
    if (
        not 0 <= validation["macro_f1"] <= 1
        or not 0 <= validation["top1_accuracy"] <= 1
        or validation["p95_ms"] < 0
    ):
        raise ValueError("Validation metric is outside its protocol range")
    if validation.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Validation evidence protocol digest mismatch")
    if (
        validation.get("device") != "cpu"
        or validation.get("batch_size") != 1
        or validation.get("warmup_iterations") != 3
        or validation.get("latency_scope")
        != "image preprocessing plus inference; excludes file decode"
    ):
        raise ValueError("Validation latency evidence differs from the frozen CPU protocol")
    for key in ("manifest_sha256", "labels_sha256", "dataset_version", "num_classes"):
        expected = DATASET_BINDING[key]
        if validation.get(key) != expected:
            raise ValueError(f"Validation evidence binding mismatch: {key}")
    labels_path = directory / "labels.json"
    checksums_path = directory / "checksum.sha256"
    config_path = directory / "model_config.json"
    metrics_path = directory / "metrics_val.json"
    preprocessing_path = directory / "preprocessing.json"
    training_config_path = directory / "train_config.yaml"
    source_commit_path = directory / "source_commit.txt"
    source_state_path = directory / "source_state.json"
    required_paths = (
        labels_path,
        checksums_path,
        config_path,
        metrics_path,
        preprocessing_path,
        training_config_path,
        source_commit_path,
        source_state_path,
    )
    if not all(path.is_file() for path in required_paths):
        raise ValueError("Candidate artifact is missing sealed provenance files")
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    if len(labels_data.get("labels", [])) != DATASET_BINDING["num_classes"]:
        raise ValueError("Candidate label map is not 101 classes")
    source_labels = json.loads((source_dir / "labels.json").read_text(encoding="utf-8"))
    if labels_data != source_labels or candidate.get("labels") != labels_data["labels"]:
        raise ValueError("Candidate labels differ from its sealed label map")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        file_path = (directory / relative).resolve()
        if not file_path.is_relative_to(directory) or not file_path.is_file():
            raise ValueError("Unsafe or missing file in candidate checksum inventory")
        if sha256_file(file_path) != expected:
            raise ValueError(f"Candidate artifact checksum mismatch: {relative}")
    actual_files = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksum.sha256"
    }
    if actual_files != set(checksums):
        raise ValueError("Candidate artifact has unsealed or missing files")
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Candidate config differs from sealed model_config.json")
    if json.loads(metrics_path.read_text(encoding="utf-8")) != validation:
        raise ValueError("Validation evidence differs from metrics_val.json")
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    seed = training_config.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Candidate training configuration has no integer seed")
    return {
        "artifact_checksum": sha256_file(checksums_path),
        "artifact_size_bytes": sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        ),
        "artifact_dir": str(directory),
        "model_name": config["model_name"],
        "run_id": config["run_id"],
        "architecture": config["model_name"],
        "preprocessing_sha256": sha256_file(preprocessing_path),
        "training_config_sha256": sha256_file(training_config_path),
        "seed": seed,
        "source_commit": source_commit_path.read_text(encoding="utf-8").strip(),
        "source_state_sha256": sha256_file(source_state_path),
        "validation_metrics_sha256": sha256_file(metrics_path),
        "validation": validation,
    }


def read_candidate(artifact_dir, protocol_path):
    """Load and verify a full-v3 candidate using its sealed local evidence."""
    directory = Path(artifact_dir).resolve()
    candidate = {
        "artifact_dir": directory,
        "config": json.loads((directory / "model_config.json").read_text(encoding="utf-8")),
        "val": json.loads((directory / "metrics_val.json").read_text(encoding="utf-8")),
        "labels": json.loads((directory / "labels.json").read_text(encoding="utf-8"))["labels"],
    }
    evidence = validate_candidate(candidate, protocol_path)
    candidate["artifact_checksum"] = evidence["artifact_checksum"]
    candidate["artifact_size_bytes"] = evidence["artifact_size_bytes"]
    return candidate


def write_selection_record(candidates, protocol_path, output):
    """Rank verified v3 candidates and publish an immutable validation selection record."""
    from food_assistant.ml.release_selection import rank_candidates

    verify_protocol(protocol_path)
    ranked_input = []
    for candidate in candidates:
        evidence = validate_candidate(candidate, protocol_path)
        ranked_input.append(
            {
                **candidate,
                "artifact_checksum": evidence["artifact_checksum"],
                "artifact_size_bytes": evidence["artifact_size_bytes"],
            }
        )
    ranked = rank_candidates(ranked_input)
    chosen = ranked[0]
    record = {
        "schema_version": "1.0",
        "record_type": "full_v3_validation_selection",
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": DATASET_BINDING,
        "ranked_candidates": [
            {
                "model_name": item["config"]["model_name"],
                "run_id": item["config"]["run_id"],
                "artifact_dir": str(Path(item["artifact_dir"]).resolve()),
                "artifact_checksum": item["artifact_checksum"],
                "validation_macro_f1": item["val"]["macro_f1"],
                "validation_top1_accuracy": item["val"]["top1_accuracy"],
                "validation_p95_ms": item["val"]["p95_ms"],
                "artifact_size_bytes": item["artifact_size_bytes"],
            }
            for item in ranked
        ],
        "selected_run_id": chosen["config"]["run_id"],
        "selected_model_name": chosen["config"]["model_name"],
    }
    record["selection_sha256"] = checkpoint_digest(record)
    _write_immutable(Path(output), record)
    return record


def freeze_selected_record(candidates, selection_path, protocol_path, output):
    """Recompute candidate ranking and freeze only its recorded validation winner."""
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    digest = selection.pop("selection_sha256", None)
    if digest != checkpoint_digest(selection):
        raise ValueError("Selection record digest mismatch")
    selection["selection_sha256"] = digest
    if (
        selection.get("protocol_sha256") != PROTOCOL_SHA256
        or selection.get("dataset") != DATASET_BINDING
    ):
        raise ValueError("Selection record dataset or protocol binding mismatch")
    from food_assistant.ml.release_selection import rank_candidates

    ranked_candidates = []
    for candidate in candidates:
        evidence = validate_candidate(candidate, protocol_path)
        ranked_candidates.append(
            {
                **candidate,
                "artifact_checksum": evidence["artifact_checksum"],
                "artifact_size_bytes": evidence["artifact_size_bytes"],
            }
        )
    ranked = rank_candidates(ranked_candidates)
    winner = ranked[0]
    if (
        winner["config"]["run_id"] != selection.get("selected_run_id")
        or winner["config"]["model_name"] != selection.get("selected_model_name")
    ):
        raise ValueError("Selection record does not identify the validation winner")
    selected = selection.get("ranked_candidates", [{}])[0]
    if (
        selected.get("artifact_checksum") != winner["artifact_checksum"]
        or selected.get("artifact_dir") != str(Path(winner["artifact_dir"]).resolve())
    ):
        raise ValueError("Selected candidate artifact differs from the selection record")
    return _publish_freeze_record(winner, selection, protocol_path, output)


def _write_immutable(path, value):
    """Create a JSON record once, allowing only byte-equivalent retries."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with resource_lock(target, timeout_seconds=0):
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing == value:
                return
            raise FileExistsError("Immutable selection target contains different evidence")
        atomic_write_json(target, value)


def _publish_freeze_record(candidate, selection, protocol_path, output):
    """Atomically publish a freeze record after verified validation selection.

    The target is create-only: an equivalent record is idempotent and different
    evidence at the same target is rejected. This function never evaluates test
    data or changes a production pointer.
    """
    evidence = validate_candidate(candidate, protocol_path)
    record = {
        "schema_version": "1.0",
        "record_type": "full_v3_candidate_freeze",
        "protocol_version": "1.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": DATASET_BINDING,
        "selection_sha256": selection["selection_sha256"],
        "candidate": evidence,
        "selection": {
            "metric": "validation_macro_f1",
            "tie_breakers": [
                "validation_top1_accuracy_desc",
                "validation_p95_ms_asc",
                "artifact_size_bytes_asc",
                "model_name_lexicographic",
                "run_id_lexicographic",
            ],
        },
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    record["record_sha256"] = checkpoint_digest(record)
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with resource_lock(target, timeout_seconds=0):
        if target.exists():
            existing = verify_freeze_record(target)
            old_comparison = dict(existing)
            new_comparison = dict(record)
            for item in (old_comparison, new_comparison):
                item.pop("selected_at_utc", None)
                item.pop("record_sha256", None)
            if old_comparison == new_comparison:
                return {"status": "SKIPPED_IDENTICAL", "record": existing}
            raise FileExistsError("Freeze target contains different evidence")
        atomic_write_json(target, record)
    return {"status": "CREATED", "record": record}


def verify_freeze_record(path):
    """Validate freeze record digest, protocol and dataset bindings."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = record.pop("record_sha256", None)
    if digest != checkpoint_digest(record):
        raise ValueError("Freeze record digest mismatch")
    record["record_sha256"] = digest
    if record.get("protocol_sha256") != PROTOCOL_SHA256 or record.get("dataset") != DATASET_BINDING:
        raise ValueError("Freeze record protocol or dataset binding mismatch")
    return record


def validate_final_test_request(freeze_path, artifact_dir, protocol_path):
    """Gate a final evaluation request against the single frozen v3 candidate."""
    record = verify_freeze_record(freeze_path)
    verify_protocol(protocol_path)
    verify_v3_source()
    artifact = Path(artifact_dir).resolve()
    selected = record["candidate"]
    if artifact != Path(selected["artifact_dir"]).resolve():
        raise ValueError("Test evaluation is allowed only for the frozen candidate")
    candidate = {
        "artifact_dir": artifact,
        "config": json.loads((artifact / "model_config.json").read_text(encoding="utf-8")),
        "val": json.loads((artifact / "metrics_val.json").read_text(encoding="utf-8")),
        "labels": json.loads((artifact / "labels.json").read_text(encoding="utf-8"))["labels"],
    }
    current = validate_candidate(candidate, protocol_path)
    if current["artifact_checksum"] != selected["artifact_checksum"]:
        raise ValueError("Frozen candidate artifact changed after selection")
    return record


def claim_final_test_attempt(
    freeze_path, evidence_output, artifact_dir, protocol_path, *, attempt_root=None
):
    """Create a durable single-attempt marker before final test data are opened."""
    record = validate_final_test_request(freeze_path, artifact_dir, protocol_path)
    freeze_path = Path(freeze_path).resolve()
    candidate = record["candidate"]
    candidate_key = checkpoint_digest(
        {"run_id": candidate["run_id"], "artifact_checksum": candidate["artifact_checksum"]}
    )
    marker_dir = Path(attempt_root or PROJECT_ROOT / "artifacts/reports/elt/full_v3_test_attempts")
    marker = marker_dir / f"{candidate_key}.json"
    target = Path(evidence_output).resolve()
    marker_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "schema_version": "1.0",
        "status": "STARTED",
        "freeze_record_sha256": sha256_file(freeze_path),
        "candidate_run_id": candidate["run_id"],
        "artifact_checksum": candidate["artifact_checksum"],
        "evidence_output": str(target),
    }
    with resource_lock(marker, timeout_seconds=0):
        if marker.exists():
            raise FileExistsError("Final test attempt already claimed for this frozen candidate")
        atomic_write_json(marker, marker_payload)
    return marker


def complete_final_test_attempt(marker_path, evidence_path, freeze_path):
    """Mark an already claimed attempt complete after separate evidence is written."""
    marker = Path(marker_path).resolve()
    with resource_lock(marker, timeout_seconds=0):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "STARTED":
            raise ValueError("Final test attempt is not in STARTED state")
        if payload.get("freeze_record_sha256") != sha256_file(freeze_path):
            raise ValueError("Freeze record changed during final test evaluation")
        payload.update(status="COMPLETED", evidence_path=str(Path(evidence_path).resolve()))
        atomic_write_json(marker, payload)


def verify_v3_source():
    """Recheck published v3 manifest, labels and metadata bindings read-only."""
    source_dir = PROJECT_ROOT / "data/splits/v3"
    for name, expected in (
        ("dataset_manifest.csv", DATASET_BINDING["manifest_sha256"]),
        ("labels.json", DATASET_BINDING["labels_sha256"]),
    ):
        if sha256_file(source_dir / name) != expected:
            raise ValueError(f"Published v3 source changed: {name}")
    source_metadata = json.loads(
        (source_dir / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    for key in ("source_fingerprint", "inventory_sha256", "v3_fingerprint"):
        if source_metadata.get(key) != DATASET_BINDING[key]:
            raise ValueError(f"Published v3 metadata binding changed: {key}")
    return source_dir
