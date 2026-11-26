"""Validation-only ranking contract tests using synthetic evidence."""

from copy import deepcopy

import pytest

from food_assistant.ml.release_selection import rank_candidates


def candidates():
    """Return two compatible synthetic candidate records."""
    labels = [{"class_id": 0, "class_key": "a"}, {"class_id": 1, "class_key": "b"}]
    return [
        {
            "config": {
                "model_name": name,
                "run_id": f"{name}-run",
                "trained": True,
                "smoke_only": False,
                "manifest_sha256": "a" * 64,
                "labels_sha256": "b" * 64,
                "dataset_version": "fixture-v1",
                "num_classes": 2,
            },
            "val": {
                "split": "val",
                "smoke_only": False,
                "sample_count": 4,
                "manifest_sha256": "a" * 64,
                "labels_sha256": "b" * 64,
                "dataset_version": "fixture-v1",
                "num_classes": 2,
                "protocol_sha256": "c" * 64,
                "macro_f1": score,
                "top1_accuracy": score,
                "p95_ms": 1.0,
            },
            "artifact_checksum": "d" * 64,
            "artifact_size_bytes": 100,
            "labels": deepcopy(labels),
        }
        for name, score in [("hog_svm", 0.8), ("resnet18", 0.7)]
    ]


def test_ranking_uses_validation_only_without_test_artifacts():
    """Select eligible candidates without requiring test evidence."""
    assert rank_candidates(candidates())[0]["config"]["model_name"] == "hog_svm"


@pytest.mark.parametrize("key", ["test", "metrics_test"])
def test_test_evidence_is_rejected_fail_closed(key):
    """Reject test metrics rather than risk allowing them to influence selection."""
    data = candidates()
    data[0][key] = {"macro_f1": 1.0}
    with pytest.raises(ValueError, match="Test evidence"):
        rank_candidates(data)


@pytest.mark.parametrize("change", ["smoke", "manifest", "labels", "protocol", "class_count"])
def test_incompatible_validation_evidence_is_rejected(change):
    """Reject validation evidence whose provenance differs from its candidate."""
    data = candidates()
    second = data[1]
    if change == "smoke":
        second["config"]["smoke_only"] = True
    elif change == "manifest":
        second["val"]["manifest_sha256"] = "f" * 64
    elif change == "labels":
        second["val"]["labels_sha256"] = "f" * 64
    elif change == "protocol":
        second["val"]["protocol_sha256"] = "f" * 64
    else:
        second["config"]["num_classes"] = 3
    with pytest.raises(ValueError):
        rank_candidates(data)


def test_tie_break_is_deterministic_and_does_not_read_test_values():
    """Resolve complete metric ties using artifact size then stable identifiers."""
    data = candidates()
    data[0]["val"].update(macro_f1=0.8, top1_accuracy=0.8, p95_ms=1.0)
    data[1]["val"].update(macro_f1=0.8, top1_accuracy=0.8, p95_ms=1.0)
    data[1]["artifact_size_bytes"] = 50
    assert rank_candidates(data)[0]["config"]["model_name"] == "resnet18"
    assert rank_candidates(data)[0]["config"]["model_name"] == "resnet18"
