from copy import deepcopy

import pytest

from food_assistant.ml.release_selection import rank_candidates


def candidates():
    # Synthetic dictionaries exercise ranking rules, not experiment results.
    return [
        {
            "config": {
                "model_name": name,
                "trained": True,
                "smoke_only": False,
                "manifest_sha256": "fixture",
                "dataset_version": "fixture-v1",
                "num_classes": 2,
            },
            "val": {"macro_f1": score},
            "test": {
                "split": "test",
                "smoke_only": False,
                "sample_count": 4,
                "manifest_sha256": "fixture",
                "dataset_version": "fixture-v1",
                "p95_ms": 1,
            },
            "labels": [{"class_id": 0, "class_key": "a"}, {"class_id": 1, "class_key": "b"}],
        }
        for name, score in [("hog_svm", 0.8), ("resnet18", 0.7)]
    ]


def test_ranking_uses_validation():
    data = candidates()
    assert rank_candidates(data)[0]["config"]["model_name"] == "hog_svm"


@pytest.mark.parametrize("change", ["smoke", "manifest", "version", "count", "labels", "method"])
def test_reject_incompatible_evidence(change):
    data = deepcopy(candidates())
    other = data[1]
    if change == "smoke":
        other["config"]["smoke_only"] = True
    elif change == "manifest":
        other["test"]["manifest_sha256"] = "wrong"
    elif change == "version":
        other["test"]["dataset_version"] = "wrong"
    elif change == "count":
        other["test"]["sample_count"] = 3
    elif change == "labels":
        other["labels"][0]["class_key"] = "wrong"
    else:
        other["config"]["model_name"] = "hog_svm"
    with pytest.raises(ValueError):
        rank_candidates(data)
