"""Validation-only candidate eligibility and deterministic ranking rules."""

import math


def rank_candidates(candidates):
    """Rank eligible trained candidates using only bound validation evidence.

    The function deliberately rejects test evidence in the input. Keeping test
    metrics out of the selection object makes accidental leakage fail closed.
    """
    if len(candidates) < 2:
        raise ValueError("At least two distinct methods are required")

    seen = set()
    reference = None
    for candidate in candidates:
        if "test" in candidate or "metrics_test" in candidate:
            raise ValueError("Test evidence is forbidden in candidate selection")
        try:
            config = candidate["config"]
            validation = candidate["val"]
            labels = candidate["labels"]
            checksum = candidate["artifact_checksum"]
            artifact_size = candidate["artifact_size_bytes"]
        except KeyError as exc:
            raise ValueError(f"Candidate evidence is incomplete: {exc.args[0]}") from exc

        if not config.get("trained") or config.get("smoke_only"):
            raise ValueError("Only completed non-smoke training is eligible")
        if validation.get("smoke_only") or validation.get("split", "val") != "val":
            raise ValueError("Selection requires formal validation-split metrics")
        if not isinstance(validation.get("sample_count"), int) or validation["sample_count"] <= 0:
            raise ValueError("Validation metrics must include evaluated samples")
        bindings = (
            "manifest_sha256",
            "dataset_version",
            "num_classes",
            "labels_sha256",
        )
        if any(validation.get(key) != config.get(key) for key in bindings):
            raise ValueError("Validation evidence differs from the trained candidate binding")
        if not validation.get("protocol_sha256"):
            raise ValueError("Validation evidence must identify its frozen protocol")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("Candidate artifact checksum is missing or invalid")
        if not isinstance(artifact_size, int) or artifact_size < 1:
            raise ValueError("Candidate artifact size is missing or invalid")
        if not labels or len(labels) != config.get("num_classes"):
            raise ValueError("Candidate labels do not match its configured class count")
        if [row.get("class_id") for row in labels] != list(range(len(labels))):
            raise ValueError("Candidate labels must have stable contiguous class IDs")
        if len({row.get("class_key") for row in labels}) != len(labels):
            raise ValueError("Candidate class keys must be unique")

        scores = tuple(validation.get(key) for key in ("macro_f1", "top1_accuracy", "p95_ms"))
        if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
            raise ValueError("Validation ranking metrics must be finite numbers")
        if not 0 <= scores[0] <= 1 or not 0 <= scores[1] <= 1 or scores[2] < 0:
            raise ValueError("Validation metric values are outside their valid ranges")

        identity = (
            *(config.get(key) for key in bindings),
            validation["sample_count"],
            validation["protocol_sha256"],
            tuple((row["class_id"], row["class_key"]) for row in labels),
        )
        if reference is not None and reference != identity:
            raise ValueError("Candidates use different datasets, samples, protocols or class maps")
        reference = identity
        model_name, run_id = config.get("model_name"), config.get("run_id")
        if not model_name or not run_id or model_name in seen:
            raise ValueError("Selection requires distinct model methods and safe run identities")
        seen.add(model_name)

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate["val"]["macro_f1"],
            -candidate["val"]["top1_accuracy"],
            candidate["val"]["p95_ms"],
            candidate["artifact_size_bytes"],
            candidate["config"]["model_name"],
            candidate["config"]["run_id"],
        ),
    )
