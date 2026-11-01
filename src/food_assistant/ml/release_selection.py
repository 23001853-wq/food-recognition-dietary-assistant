"""Pure validation/ranking of measured model releases, without loading ML libraries."""


def rank_candidates(candidates):
    if len(candidates) < 2:
        raise ValueError("At least two distinct methods are required")
    seen, reference = set(), None
    for candidate in candidates:
        config, test = candidate["config"], candidate["test"]
        if not config.get("trained") or config.get("smoke_only") or test.get("smoke_only"):
            raise ValueError("Only completed non-smoke training/evaluation is eligible")
        if test.get("split") != "test" or test.get("sample_count", 0) <= 0:
            raise ValueError("Final test evaluation is required")
        if test.get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("Test manifest differs from training")
        if test.get("dataset_version") != config["dataset_version"]:
            raise ValueError("Test dataset version differs from training")
        identity = (
            config["manifest_sha256"],
            config["dataset_version"],
            config["num_classes"],
            test["sample_count"],
            tuple((x["class_id"], x["class_key"]) for x in candidate["labels"]),
            test.get("protocol_sha256"),
            test.get("device"),
        )
        if reference is not None and reference != identity:
            raise ValueError("Candidates use different samples, versions or class maps")
        reference = identity
        if config["model_name"] in seen:
            raise ValueError("Comparison requires distinct methods")
        seen.add(config["model_name"])
    return sorted(candidates, key=lambda c: (-c["val"]["macro_f1"], c["test"]["p95_ms"]))
