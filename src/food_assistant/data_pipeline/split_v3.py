"""Create immutable training-quality and validation split v3 from verified v2."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn
from sklearn.model_selection import train_test_split

from food_assistant.data_pipeline.atomic import (
    atomic_write_csv,
    atomic_write_json,
    checkpoint_digest,
    resource_lock,
)
from food_assistant.data_pipeline.config import PipelineConfig, load_pipeline_config
from food_assistant.data_pipeline.splits import (
    REPORT_PATH,
    RawBaselineMismatch,
    _config_hash,
    _inventory,
    _read_report,
    _split_fingerprint,
)
from food_assistant.ml.datasets import FIELDS, ManifestDataset

V2_CONFIG_PATH = Path("configs/food101_elt_full_v2.yaml")
DUPLICATE_POLICY_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject malformed or non-object evidence."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawBaselineMismatch(f"Cannot read split evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RawBaselineMismatch(f"Split evidence is not a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV manifest and require the versioned dataset fields."""
    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("Manifest columns do not match the v2/v3 reader contract")
            return list(reader)
    except (OSError, csv.Error, ValueError) as exc:
        raise RawBaselineMismatch(f"Cannot read split manifest {path}: {exc}") from exc


def _validate_v2_source(
    config: PipelineConfig,
    v2_config_path: Path,
    baseline_report_path: Path,
) -> dict[str, Any]:
    """Validate v2 metadata, manifest, fingerprints and raw checkpoint evidence."""
    source_root = config.source_manifest_root
    if source_root is None:
        raise RawBaselineMismatch("v3 config does not declare source_manifest_root")
    v2_config_path = Path(v2_config_path).resolve()
    v2_config = load_pipeline_config(v2_config_path, v2_config_path.parents[1])
    if v2_config.split_version != "v2" or v2_config.manifest_root != source_root:
        raise RawBaselineMismatch("Configured source is not the approved full Food-101 v2 root")
    baseline = _read_report(baseline_report_path)
    checkpoint_path = Path(baseline_report_path).parent / "food101_checkpoint.json"
    raw_hashes, inventory_sha256, source_fingerprint = _inventory(
        config.raw_root, baseline, checkpoint_path
    )
    metadata = _read_json(source_root / "dataset_metadata.json")
    internal_report = _read_json(source_root / "split_quality_report.json")
    external_report = _read_json(v2_config.report_root / "clean_split_v2_report.json")
    labels = _read_json(source_root / "labels.json")
    expected_config_hash = _config_hash(v2_config_path, v2_config)
    if (
        metadata.get("schema_version") != "2.0"
        or metadata.get("dataset_version") != "v2"
        or metadata.get("complete") is not True
        or metadata.get("clean_split_status") != "PASS"
        or internal_report.get("clean_split_status") != "PASS"
        or external_report.get("clean_split_status") != "PASS"
        or metadata.get("source_fingerprint") != source_fingerprint
        or metadata.get("inventory_sha256") != inventory_sha256
        or metadata.get("config_hash") != expected_config_hash
        or internal_report.get("config_hash") != expected_config_hash
        or external_report.get("config_hash") != expected_config_hash
        or metadata.get("split_fingerprint") != internal_report.get("split_fingerprint")
        or metadata.get("split_fingerprint") != external_report.get("split_fingerprint")
        or metadata.get("split_fingerprint")
        != _split_fingerprint(
            _read_csv(source_root / "dataset_manifest.csv"),
            metadata.get("excluded_train_images", []),
            expected_config_hash,
        )
        or external_report.get("source_fingerprint") != source_fingerprint
        or external_report.get("inventory_sha256") != inventory_sha256
        or external_report.get("split_counts") != metadata.get("split_counts")
        or external_report.get("clean_cross_split_duplicate_hash_count") != 0
    ):
        raise RawBaselineMismatch(
            "Clean Split v2 metadata/report/fingerprint is invalid or mismatched"
        )
    class_names = (config.raw_root / "meta/classes.txt").read_text(encoding="utf-8").splitlines()
    label_names = [entry.get("class_key") for entry in labels.get("labels", [])]
    if (
        labels.get("schema_version") != "2.0"
        or labels.get("dataset_version") != "v2"
        or label_names != class_names
    ):
        raise RawBaselineMismatch("v2 label mapping differs from official classes.txt")
    rows = _read_csv(source_root / "dataset_manifest.csv")
    official_test_ids = (config.raw_root / "meta/test.txt").read_text(encoding="utf-8").splitlines()
    train_rows = [row for row in rows if row.get("split") == "train"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if len(rows) != sum(metadata.get("split_counts", {}).values()):
        raise RawBaselineMismatch("v2 manifest row count differs from its metadata")
    if [row.get("image_id") for row in test_rows] != official_test_ids:
        raise RawBaselineMismatch("v2 test IDs/order differ from official test metadata")
    if len({row.get("image_id") for row in rows}) != len(rows):
        raise RawBaselineMismatch("v2 manifest contains duplicate image IDs")
    class_ids = {name: index for index, name in enumerate(class_names)}
    seen_hashes: dict[str, str] = {}
    for row in rows:
        image_id = row.get("image_id", "")
        relative = row.get("relative_path", "")
        split = row.get("split")
        if (
            split not in {"train", "test"}
            or relative != f"images/{image_id}.jpg"
            or image_id.split("/", 1)[0] not in class_ids
            or int(row.get("class_id", -1)) != class_ids[image_id.split("/", 1)[0]]
            or row.get("class_key") != image_id.split("/", 1)[0]
            or row.get("dataset_version") != "v2"
            or raw_hashes.get(relative) != row.get("checksum")
        ):
            raise RawBaselineMismatch(f"v2 manifest row is invalid: {image_id}")
        prior_split = seen_hashes.setdefault(row["checksum"], split)
        if prior_split != split:
            raise RawBaselineMismatch("v2 contains a cross-partition exact duplicate")
    if len(train_rows) != metadata["split_counts"].get("train") or len(test_rows) != metadata[
        "split_counts"
    ].get("test"):
        raise RawBaselineMismatch("v2 split counts differ from metadata")
    return {
        "metadata": metadata,
        "report": external_report,
        "labels": labels,
        "rows": rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "class_names": class_names,
        "raw_hashes": raw_hashes,
        "inventory_sha256": inventory_sha256,
        "source_fingerprint": source_fingerprint,
        "baseline": baseline,
        "checkpoint_path": checkpoint_path,
    }


def _clean_train_rows(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate same-label groups and quarantine every cross-label group."""
    test_hashes = {row["checksum"] for row in test_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        grouped[row["checksum"]].append(row)
    candidates: list[dict[str, str]] = []
    exclusions: list[dict[str, Any]] = []
    conflict_groups: list[dict[str, Any]] = []
    for digest, group in sorted(grouped.items()):
        image_ids = sorted(row["image_id"] for row in group)
        class_keys = sorted({row["class_key"] for row in group})
        if digest in test_hashes:
            raise RawBaselineMismatch(
                "v2 has a new train/test duplicate; refusing to silently alter test"
            )
        if len(group) == 1:
            candidates.append(group[0])
            continue
        if len(class_keys) > 1:
            conflict_groups.append(
                {
                    "sha256": digest,
                    "image_ids": image_ids,
                    "classes": class_keys,
                    "status": "LABEL_CONFLICT_QUARANTINED",
                    "source_split_version": "v2",
                }
            )
            for row in group:
                exclusions.append(
                    {
                        "image_id": row["image_id"],
                        "class_key": row["class_key"],
                        "sha256": digest,
                        "original_partition": "train",
                        "reason": "cross_label_exact_duplicate_quarantined",
                        "representative_image_id": None,
                        "matching_image_ids": [
                            name for name in image_ids if name != row["image_id"]
                        ],
                        "status": "LABEL_CONFLICT_QUARANTINED",
                        "source_split_version": "v2",
                    }
                )
            continue
        representative = min(group, key=lambda row: row["image_id"])
        candidates.append(representative)
        for row in group:
            if row["image_id"] == representative["image_id"]:
                continue
            exclusions.append(
                {
                    "image_id": row["image_id"],
                    "class_key": row["class_key"],
                    "sha256": digest,
                    "original_partition": "train",
                    "reason": "same_label_exact_duplicate_deduplicated",
                    "representative_image_id": representative["image_id"],
                    "matching_image_ids": [representative["image_id"]],
                    "status": "DEDUPLICATED",
                    "source_split_version": "v2",
                }
            )
    candidates.sort(key=lambda row: row["image_id"])
    exclusions.sort(key=lambda row: row["image_id"])
    if len({row["checksum"] for row in candidates}) != len(candidates):
        raise ValueError("Train deduplication left a repeated SHA-256 candidate")
    return candidates, exclusions, conflict_groups


def _stratified_assignment(
    candidates: list[dict[str, str]], seed: int, fraction: float, class_names: list[str]
) -> dict[str, str]:
    """Split hash-unique train candidates deterministically with class stratification."""
    counts = Counter(row["class_key"] for row in candidates)
    missing = sorted(set(class_names) - set(counts))
    if missing or any(count < 2 for count in counts.values()):
        raise ValueError(f"Cannot stratify while retaining every class in train: {missing}")
    train_rows, validation_rows = train_test_split(
        candidates,
        test_size=fraction,
        random_state=seed,
        stratify=[row["class_key"] for row in candidates],
    )
    assignments = {row["image_id"]: "train" for row in train_rows}
    assignments.update({row["image_id"]: "val" for row in validation_rows})
    train_counts = Counter(row["class_key"] for row in train_rows)
    if set(train_counts) != set(class_names) or any(value < 1 for value in train_counts.values()):
        raise ValueError("Stratified split would remove a class from training")
    return assignments


def _fingerprint_v3(
    rows: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    source_v2_fingerprint: str,
    config_hash: str,
) -> str:
    """Bind source v2, config, partition rows and all exclusions into v3 identity."""
    return checkpoint_digest(
        {
            "source_v2_fingerprint": source_v2_fingerprint,
            "config_hash": config_hash,
            "rows": sorted(
                (
                    row["image_id"],
                    row["split"],
                    int(row["class_id"]),
                    row["checksum"],
                )
                for row in rows
            ),
            "exclusions": sorted(exclusions, key=lambda item: item["image_id"]),
        }
    )


def _check_output_rows(
    rows: list[dict[str, Any]],
    classes: list[str],
    test_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate paths, mappings, IDs, test preservation and cross-partition hashes."""
    identities: set[str] = set()
    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_train_count: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    class_to_id = {name: index for index, name in enumerate(classes)}
    for row in rows:
        image_id = row.get("image_id", "")
        relative = Path(row.get("relative_path", ""))
        split = row.get("split")
        if (
            not image_id
            or image_id in identities
            or relative.is_absolute()
            or ".." in relative.parts
            or split not in per_class
            or row.get("class_key") not in class_to_id
            or int(row.get("class_id", -1)) != class_to_id[row["class_key"]]
            or relative.as_posix() != f"images/{image_id}.jpg"
        ):
            raise ValueError(f"Invalid or duplicate manifest row: {image_id}")
        identities.add(image_id)
        hash_splits[row["checksum"]].add(split)
        if split == "train":
            hash_train_count[row["checksum"]] += 1
        per_class[split][row["class_key"]] += 1
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Exact duplicate hash crosses v3 partitions")
    if any(count > 1 for count in hash_train_count.values()):
        raise ValueError("Exact duplicate remains inside v3 train")
    test_output = [
        (row["image_id"], row["class_id"], row["class_key"], row["checksum"])
        for row in rows
        if row["split"] == "test"
    ]
    test_source = [
        (row["image_id"], row["class_id"], row["class_key"], row["checksum"]) for row in test_rows
    ]
    if test_output != test_source:
        raise ValueError("v3 test IDs, labels, order or content differ from v2")
    if set(per_class["train"]) != set(classes):
        raise ValueError("v3 train does not contain all configured classes")
    split_counts = Counter(row["split"] for row in rows)
    if set(split_counts) != {"train", "val", "test"}:
        raise ValueError("One or more required v3 partitions are empty")
    return {
        "split_counts": dict(split_counts),
        "per_class_counts": {
            split: {key: per_class[split].get(key, 0) for key in classes} for split in per_class
        },
        "cross_partition_duplicate_hash_count": 0,
        "train_duplicate_hash_count": 0,
    }


def _make_rows(
    candidates: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    assignments: dict[str, str],
    version: str,
) -> list[dict[str, Any]]:
    """Build deterministic v3 manifest rows from cleaned candidates and fixed test."""
    rows: list[dict[str, Any]] = []
    for source_row in candidates:
        row = dict(source_row)
        row["split"] = assignments[source_row["image_id"]]
        row["dataset_version"] = version
        rows.append(row)
    for source_row in test_rows:
        row = dict(source_row)
        row["dataset_version"] = version
        rows.append(row)
    non_test_rows = [row for row in rows if row["split"] != "test"]
    non_test_rows.sort(key=lambda row: (row["split"], row["image_id"]))
    return non_test_rows + rows[len(candidates) :]


def _validate_existing_v3(
    output: Path,
    raw_root: Path,
    expected_fingerprint: str,
    expected_source: str,
    expected_config: str,
    expected_rows: list[dict[str, Any]],
    expected_exclusions: list[dict[str, Any]],
    expected_v2_fingerprint: str,
) -> dict[str, Any]:
    """Validate matching v3 output through its metadata, reports and manifest reader."""
    metadata = _read_json(output / "dataset_metadata.json")
    report = _read_json(output / "quality_report.json")
    manifest_rows = _read_csv(output / "dataset_manifest.csv")
    actual_fingerprint = _fingerprint_v3(
        manifest_rows,
        metadata.get("excluded_images", []),
        expected_v2_fingerprint,
        expected_config,
    )
    expected_identity = [
        (row["image_id"], row["split"], int(row["class_id"]), row["checksum"])
        for row in expected_rows
    ]
    actual_identity = [
        (row["image_id"], row["split"], int(row["class_id"]), row["checksum"])
        for row in manifest_rows
    ]
    if (
        metadata.get("complete") is not True
        or metadata.get("quality_gates", {}).get("overall") != "PASS"
        or metadata.get("v3_fingerprint") != expected_fingerprint
        or metadata.get("source_fingerprint") != expected_source
        or metadata.get("config_hash") != expected_config
        or report.get("v3_fingerprint") != expected_fingerprint
        or report.get("quality_gates", {}).get("overall") != "PASS"
        or actual_fingerprint != expected_fingerprint
        or actual_identity != expected_identity
        or metadata.get("excluded_images") != expected_exclusions
        or report.get("excluded_images") != expected_exclusions
    ):
        raise FileExistsError("Existing v3 output conflicts with source/config fingerprint")
    ManifestDataset(output, "train")
    for split in ("val", "test"):
        if not (output / f"{split}.csv").is_file():
            raise FileExistsError(f"Existing v3 output is missing {split}.csv")
    return {**report, "state": "SKIPPED", "output": str(output)}


def build_training_split_v3(
    config_path: Path,
    v2_config_path: Path = V2_CONFIG_PATH,
    baseline_report_path: Path = REPORT_PATH,
    *,
    publish: bool = True,
) -> dict[str, Any]:
    """Validate clean v2, quarantine label conflicts and atomically publish v3."""
    config_path = Path(config_path).resolve()
    workspace = config_path.parents[1]
    config = load_pipeline_config(config_path, workspace)
    if config.split_version != "v3" or config.dataset_scope != "full":
        raise ValueError("The v3 builder requires a full-scope v3 config")
    source = _validate_v2_source(config, v2_config_path, baseline_report_path)
    classes = source["class_names"]
    config_hash = _config_hash(config_path, config)
    candidates, exclusions, conflict_groups = _clean_train_rows(
        source["train_rows"], source["test_rows"]
    )
    assignments = _stratified_assignment(candidates, config.seed, config.val_fraction, classes)
    rows = _make_rows(candidates, source["test_rows"], assignments, config.split_version)
    stats = _check_output_rows(rows, classes, source["test_rows"])
    if not stats["split_counts"].get("train") or set(stats["per_class_counts"]["train"]) != set(
        classes
    ):
        raise ValueError("v3 quality gate would remove a class from train")
    v3_fingerprint = _fingerprint_v3(
        rows,
        exclusions,
        source["metadata"]["split_fingerprint"],
        config_hash,
    )
    output = config.manifest_root
    report_root = config.report_root
    output_lock = resource_lock(output, config.lock_timeout_seconds) if publish else nullcontext()
    with output_lock:
        if output.exists():
            result = _validate_existing_v3(
                output,
                config.raw_root,
                v3_fingerprint,
                source["source_fingerprint"],
                config_hash,
                rows,
                exclusions,
                source["metadata"]["split_fingerprint"],
            )
            if publish:
                report_root.mkdir(parents=True, exist_ok=True)
                atomic_write_json(report_root / "quality_report.json", result)
            return result
        gates = {
            "source_integrity": "PASS",
            "train_data_quality": "PASS",
            "partition_independence": "PASS",
            "manifest_quality": "PASS",
            "overall": "PASS",
        }
        per_class_before = {
            "train": dict(Counter(row["class_key"] for row in source["train_rows"])),
            "test": dict(Counter(row["class_key"] for row in source["test_rows"])),
        }
        policy = {
            "version": DUPLICATE_POLICY_VERSION,
            "same_label_duplicates": "keep lexicographically smallest image_id per SHA-256",
            "cross_label_duplicates": "quarantine every reference; no label inference",
            "test": "preserve source v2 IDs, labels, order and checksums",
            "validation": "stratified after SHA-256 group collapse",
        }
        report = {
            "schema_version": "3.0",
            "dataset_version": "v3",
            "quality_gates": gates,
            "source_split_fingerprint": source["metadata"]["split_fingerprint"],
            "source_fingerprint": source["source_fingerprint"],
            "inventory_sha256": source["inventory_sha256"],
            "config_hash": config_hash,
            "v3_fingerprint": v3_fingerprint,
            "seed": config.seed,
            "validation_fraction": config.val_fraction,
            "validation_algorithm": {
                "name": config.validation_algorithm,
                "library": "scikit-learn train_test_split",
                "library_version": sklearn.__version__,
                "stratify_by": "class_key",
                "hash_grouping": "one candidate per SHA-256 after dedup/quarantine",
            },
            "deduplication_policy": policy,
            "before_counts": {
                "train": len(source["train_rows"]),
                "validation": 0,
                "test": len(source["test_rows"]),
            },
            "after_dedup_and_quarantine_counts": {
                "train_candidates": len(candidates),
                "test": len(source["test_rows"]),
            },
            **stats,
            "per_class_before": per_class_before,
            "excluded_images": exclusions,
            "label_conflict_groups": conflict_groups,
            "same_label_duplicate_group_count": sum(
                len({row["checksum"] for row in group}) == 1
                and len(group) > 1
                and len({row["class_key"] for row in group}) == 1
                for group in _group_rows(source["train_rows"])
            ),
            "cross_label_duplicate_group_count": len(conflict_groups),
            "verified_at_utc": datetime.now(UTC).isoformat(),
        }
        if not publish:
            return report
        report_root.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        try:
            for name, selected in [("dataset_manifest", rows)] + [
                (split, [row for row in rows if row["split"] == split])
                for split in ("train", "val", "test")
            ]:
                atomic_write_csv(staging / f"{name}.csv", selected, FIELDS)
            labels = {
                "schema_version": "3.0",
                "dataset_version": config.split_version,
                "labels": [
                    {
                        "class_id": index,
                        "class_key": name,
                        "display_name_vi": name,
                        "display_name_en": name.replace("_", " "),
                    }
                    for index, name in enumerate(classes)
                ],
            }
            metadata = {
                "schema_version": "3.0",
                "manifest_schema_version": "3.0",
                "dataset_version": config.split_version,
                "root": os.path.relpath(config.raw_root, output).replace(os.sep, "/"),
                "source_split_version": "v2",
                "source_split_fingerprint": source["metadata"]["split_fingerprint"],
                "source_fingerprint": source["source_fingerprint"],
                "inventory_sha256": source["inventory_sha256"],
                "config_hash": config_hash,
                "v3_fingerprint": v3_fingerprint,
                "seed": config.seed,
                "validation_fraction": config.val_fraction,
                "validation_algorithm": report["validation_algorithm"],
                "deduplication_policy": policy,
                "split_counts": stats["split_counts"],
                "per_class_counts": stats["per_class_counts"],
                "before_counts": report["before_counts"],
                "after_dedup_and_quarantine_counts": report["after_dedup_and_quarantine_counts"],
                "excluded_images": exclusions,
                "label_conflict_groups": conflict_groups,
                "quality_gates": gates,
                "class_mapping": labels["labels"],
                "complete": True,
            }
            atomic_write_json(staging / "labels.json", labels)
            atomic_write_json(staging / "excluded_images.json", exclusions)
            atomic_write_json(staging / "label_conflicts.json", conflict_groups)
            atomic_write_json(staging / "quality_report.json", report)
            atomic_write_json(staging / "dataset_metadata.json", metadata)
            staged_reader = ManifestDataset(staging, "train", root_override=config.raw_root)
            if len(staged_reader) != stats["split_counts"]["train"]:
                raise ValueError("Staged ManifestDataset train count is invalid")
            for split in ("val", "test"):
                if len(_read_csv(staging / f"{split}.csv")) != stats["split_counts"][split]:
                    raise ValueError(f"Staged {split} manifest count is invalid")
            raw_hashes_after, inventory_after, source_after = _inventory(
                config.raw_root, source["baseline"], source["checkpoint_path"]
            )
            if (
                raw_hashes_after != source["raw_hashes"]
                or inventory_after != source["inventory_sha256"]
                or source_after != source["source_fingerprint"]
            ):
                raise RawBaselineMismatch("Raw source changed before v3 publish")
            if output.exists():
                raise FileExistsError("v3 target appeared during build; refusing overwrite")
            staging.rename(output)
            atomic_write_json(report_root / "quality_report.json", report)
            atomic_write_json(report_root / "excluded_images.json", exclusions)
            atomic_write_json(report_root / "label_conflicts.json", conflict_groups)
            return {**report, "state": "PASS", "output": str(output)}
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def _group_rows(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Group manifest rows by exact content hash for duplicate statistics."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["checksum"]].append(row)
    return [group for group in groups.values() if len(group) > 1]
