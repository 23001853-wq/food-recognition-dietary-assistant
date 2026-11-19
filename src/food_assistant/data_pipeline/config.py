"""Read and validate the side-effect-free Food-101 ELT configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Validated paths and policy values for a Food-101 ELT run."""

    raw_root: Path
    staging_root: Path
    archive: Path
    manifest_root: Path
    report_root: Path
    approval_file: Path
    class_selection_file: Path
    dataset_scope: str
    split_version: str
    seed: int
    val_fraction: float
    download_policy: str
    retries: int
    backoff_seconds: float
    require_full_integrity: bool
    reject_cross_split_duplicates: bool
    lock_timeout_seconds: float
    fixtures_enabled: bool
    fixture_root: Path
    preserve_official_test: bool
    exclude_train_hashes_in_test: bool
    create_validation: bool
    reject_cross_partition_duplicates: bool
    source_manifest_root: Path | None
    deduplicate_train: bool
    same_label_duplicate_policy: str
    cross_label_duplicate_policy: str
    validation_algorithm: str


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    """Require a named configuration section to be a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _contained_path(root: Path, value: Any, name: str) -> Path:
    """Resolve a non-empty configured path and reject paths outside the workspace."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{name} must remain inside the workspace")
    return resolved


def load_pipeline_config(path: Path, workspace: Path | None = None) -> PipelineConfig:
    """Load and validate YAML without creating files, directories, or locks."""
    config_path = Path(path)
    workspace_root = Path(workspace or Path.cwd()).resolve()
    config_path = _contained_path(workspace_root, str(config_path), "config path")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load ELT config: {exc}") from exc
    document = _required_mapping(raw, "config")
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported config schema_version")
    paths = _required_mapping(document.get("paths"), "paths")
    download = _required_mapping(document.get("download"), "download")
    quality = _required_mapping(document.get("quality"), "quality")
    locks = _required_mapping(document.get("locks"), "locks")
    fixtures = _required_mapping(document.get("fixtures"), "fixtures")
    split_policy = _required_mapping(document.get("split_policy", {}), "split_policy")

    resolved = {
        name: _contained_path(workspace_root, value, f"paths.{name}")
        for name, value in paths.items()
    }
    required_paths = {"raw_root", "staging_root", "archive", "manifest_root", "report_root"}
    if set(resolved) != required_paths:
        raise ValueError(
            "paths must define exactly raw_root, staging_root, archive, manifest_root, report_root"
        )
    raw_root = resolved["raw_root"]
    protected = [raw_root, workspace_root / "data" / "splits" / "v1"]
    for key in ("staging_root", "archive", "manifest_root", "report_root"):
        candidate = resolved[key]
        if any(candidate == item or candidate.is_relative_to(item) for item in protected):
            raise ValueError(f"paths.{key} overlaps raw data or frozen manifest v1")
    if resolved["manifest_root"] == resolved["report_root"]:
        raise ValueError("manifest_root and report_root must be distinct")

    scope = document.get("dataset_scope")
    if scope not in {"full", "poc"}:
        raise ValueError("dataset_scope must be 'full' or 'poc'")
    version = document.get("split_version")
    if not isinstance(version, str) or not version or version == "v1":
        raise ValueError("split_version must name a new version; v1 is frozen")
    source_manifest_value = document.get("source_manifest_root")
    source_manifest_root = (
        _contained_path(workspace_root, source_manifest_value, "source_manifest_root")
        if source_manifest_value is not None
        else None
    )
    if version == "v3" and source_manifest_root is None:
        raise ValueError("split v3 requires source_manifest_root")
    if version != "v3" and source_manifest_root is not None:
        raise ValueError("source_manifest_root is supported only for split v3")
    seed = document.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    fraction = document.get("val_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    policy = download.get("policy")
    if policy not in {"disabled", "approved_only"}:
        raise ValueError("download.policy must be disabled or approved_only")
    retries, backoff = download.get("retries"), download.get("backoff_seconds")
    timeout = locks.get("timeout_seconds")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("download.retries must be a non-negative integer")
    if not isinstance(backoff, (int, float)) or backoff < 0:
        raise ValueError("download.backoff_seconds must be non-negative")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("locks.timeout_seconds must be positive")
    for key in ("require_full_integrity", "reject_cross_split_duplicates"):
        if not isinstance(quality.get(key), bool):
            raise ValueError(f"quality.{key} must be boolean")
    if not isinstance(fixtures.get("enabled"), bool):
        raise ValueError("fixtures.enabled must be boolean")
    policy_defaults = {
        "preserve_official_test": True,
        "exclude_train_hashes_in_test": False,
        "create_validation": True,
        "reject_cross_partition_duplicates": True,
        "deduplicate_train": False,
        "same_label_duplicate_policy": "retain",
        "cross_label_duplicate_policy": "reject",
        "validation_algorithm": "stratified_after_sha256_group_collapse",
    }
    if set(split_policy) - set(policy_defaults):
        raise ValueError("split_policy contains unsupported fields")
    policy_values = {**policy_defaults, **split_policy}
    boolean_policy_keys = {
        "preserve_official_test",
        "exclude_train_hashes_in_test",
        "create_validation",
        "reject_cross_partition_duplicates",
        "deduplicate_train",
    }
    if any(not isinstance(policy_values[key], bool) for key in boolean_policy_keys):
        raise ValueError("split_policy boolean values must be boolean")
    if (
        scope == "full"
        and version == "v2"
        and (
            not policy_values["preserve_official_test"]
            or not policy_values["exclude_train_hashes_in_test"]
            or policy_values["create_validation"]
            or not policy_values["reject_cross_partition_duplicates"]
        )
    ):
        raise ValueError(
            "Full v2 policy must preserve test, exclude leakage, omit validation, "
            "and reject cross-partition duplicates"
        )
    if (
        scope == "full"
        and version == "v3"
        and (
            not policy_values["preserve_official_test"]
            or policy_values["exclude_train_hashes_in_test"]
            or not policy_values["create_validation"]
            or not policy_values["reject_cross_partition_duplicates"]
            or not policy_values["deduplicate_train"]
            or policy_values["same_label_duplicate_policy"] != "keep_lexicographic_min"
            or policy_values["cross_label_duplicate_policy"] != "quarantine_all"
            or policy_values["validation_algorithm"] != "stratified_after_sha256_group_collapse"
        )
    ):
        raise ValueError(
            "Full v3 policy does not match the approved deduplication and validation contract"
        )
    for key in (
        "same_label_duplicate_policy",
        "cross_label_duplicate_policy",
        "validation_algorithm",
    ):
        if not isinstance(policy_values[key], str) or not policy_values[key]:
            raise ValueError(f"split_policy.{key} must be a non-empty string")
    if source_manifest_root is not None:
        if any(
            source_manifest_root == protected_path
            or source_manifest_root.is_relative_to(protected_path)
            or protected_path.is_relative_to(source_manifest_root)
            for protected_path in protected
        ):
            raise ValueError("source_manifest_root overlaps raw data or frozen manifest v1")
        if source_manifest_root == resolved["manifest_root"]:
            raise ValueError("source_manifest_root and manifest_root must be distinct")
        if source_manifest_root.is_relative_to(resolved["manifest_root"]) or resolved[
            "manifest_root"
        ].is_relative_to(source_manifest_root):
            raise ValueError("source and output manifest roots must not overlap")
        if (
            source_manifest_root == resolved["report_root"]
            or source_manifest_root.is_relative_to(resolved["report_root"])
            or resolved["report_root"].is_relative_to(source_manifest_root)
        ):
            raise ValueError("source manifest and report roots must be distinct")
    if version == "v3":
        v3_resources = {
            "raw_root": raw_root,
            "staging_root": resolved["staging_root"],
            "archive": resolved["archive"],
            "manifest_root": resolved["manifest_root"],
            "report_root": resolved["report_root"],
            "source_manifest_root": source_manifest_root,
            "frozen_v1": protected[1].resolve(),
        }
        resource_names = list(v3_resources)
        for index, left_name in enumerate(resource_names):
            left_path = v3_resources[left_name]
            for right_name in resource_names[index + 1 :]:
                right_path = v3_resources[right_name]
                if (
                    left_path == right_path
                    or left_path.is_relative_to(right_path)
                    or right_path.is_relative_to(left_path)
                ):
                    raise ValueError(f"v3 resources {left_name} and {right_name} must not overlap")

    return PipelineConfig(
        **resolved,
        approval_file=_contained_path(
            workspace_root, document.get("approval_file"), "approval_file"
        ),
        class_selection_file=_contained_path(
            workspace_root, document.get("class_selection_file"), "class_selection_file"
        ),
        dataset_scope=scope,
        split_version=version,
        seed=seed,
        val_fraction=float(fraction),
        download_policy=policy,
        retries=retries,
        backoff_seconds=float(backoff),
        require_full_integrity=quality["require_full_integrity"],
        reject_cross_split_duplicates=quality["reject_cross_split_duplicates"],
        lock_timeout_seconds=float(timeout),
        fixtures_enabled=fixtures["enabled"],
        fixture_root=_contained_path(workspace_root, fixtures.get("root"), "fixtures.root"),
        **policy_values,
        source_manifest_root=source_manifest_root,
    )
