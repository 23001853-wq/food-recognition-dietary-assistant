"""Build and verify immutable Food-101 clean split artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from food_assistant.data_pipeline.atomic import (
    atomic_write_csv,
    atomic_write_json,
    checkpoint_digest,
    resource_lock,
)
from food_assistant.data_pipeline.config import PipelineConfig, load_pipeline_config
from food_assistant.ml.datasets import FIELDS, audit_manifest

REPORT_PATH = Path("data/reports/elt/v2/phase_2_5_verification_retry/food101_report.json")
SPLIT_POLICY_VERSION = 1


class RawBaselineMismatch(RuntimeError):
    """Raised when current raw evidence differs from the approved baseline."""


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> dict[str, Any]:
    """Load the Phase 2.5 report and fail closed on malformed evidence."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawBaselineMismatch(f"Cannot load verification baseline: {exc}") from exc
    if not isinstance(value, dict) or value.get("mode") != "full":
        raise RawBaselineMismatch("Verification baseline is not a full audit report")
    return value


def _inventory(
    root: Path, baseline: dict[str, Any], checkpoint_path: Path
) -> tuple[dict[str, str], str, str]:
    """Validate the Phase 2.5 full checkpoint against current file signatures."""
    image_root = root / "images"
    actual = {
        path.relative_to(root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    }
    expected = {
        path
        for path in baseline.get("filesystem_paths", [])
        if path.startswith("images/") and path.endswith(".jpg")
    }
    non_jpg = [
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".jpg"
    ]
    if not expected or actual != expected:
        raise RawBaselineMismatch(
            "Raw image inventory differs from Phase 2.5 "
            f"(expected={len(expected)}, actual={len(actual)})"
        )
    if non_jpg:
        raise RawBaselineMismatch(f"Unexpected non-JPG file under images/: {non_jpg[0]}")
    class_count = len((root / "meta/classes.txt").read_text(encoding="utf-8").splitlines())
    if (
        baseline.get("actual_image_count") != len(expected)
        or baseline.get("class_count") != class_count
    ):
        raise RawBaselineMismatch(
            "Phase 2.5 class/image counts do not match the expected inventory"
        )
    try:
        envelope = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        checkpoint = envelope["payload"]
        checkpoint_valid = envelope["sha256"] == checkpoint_digest(checkpoint)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RawBaselineMismatch(f"Cannot validate Phase 2.5 checkpoint: {exc}") from exc
    if (
        not checkpoint_valid
        or checkpoint.get("schema_version") != 2
        or checkpoint.get("complete") is not True
        or Path(checkpoint.get("dataset_root", "")).resolve() != root.resolve()
        or checkpoint.get("expected_files") != sorted(expected)
        or set(checkpoint.get("files", {})) != expected
    ):
        raise RawBaselineMismatch(
            "Phase 2.5 checkpoint is incomplete or does not bind this raw inventory"
        )
    signatures: dict[str, dict[str, int]] = {}
    fingerprint_paths = sorted(set(baseline.get("filesystem_paths", [])))
    for relative in fingerprint_paths:
        path = root / Path(relative)
        if not path.is_file():
            raise RawBaselineMismatch(f"Raw file is missing since Phase 2.5: {relative}")
        stat = path.stat()
        signatures[relative] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    filesystem_fingerprint = checkpoint_digest(signatures)
    if (
        filesystem_fingerprint != baseline.get("filesystem_fingerprint")
        or checkpoint.get("input_hash") != filesystem_fingerprint
    ):
        raise RawBaselineMismatch("Raw file signatures differ from the Phase 2.5 checkpoint")
    hashes: dict[str, str] = {}
    inventory_digest = hashlib.sha256()
    for relative in sorted(expected):
        checkpoint_entry = checkpoint["files"][relative]
        if checkpoint_entry.get("signature") != signatures.get(relative):
            raise RawBaselineMismatch(f"Checkpoint signature mismatch for {relative}")
        result = checkpoint_entry.get("result", {})
        if result.get("valid") is not True or not isinstance(result.get("sha256"), str):
            raise RawBaselineMismatch(f"Phase 2.5 has no valid decode/hash evidence for {relative}")
        digest = result["sha256"]
        hashes[relative] = digest
        inventory_digest.update(f"{relative}:{digest}\n".encode())
    inventory_sha256 = inventory_digest.hexdigest()
    metadata_hashes = baseline.get("metadata_sha256", {})
    current_metadata_hashes = {
        relative: _sha256_file(root / Path(relative)) for relative in metadata_hashes
    }
    if current_metadata_hashes != metadata_hashes:
        raise RawBaselineMismatch("Official metadata changed since the Phase 2.5 baseline")
    marker = json.loads((root / "download_metadata.json").read_text(encoding="utf-8"))
    marker_claims = (
        baseline.get("criteria", {})
        .get("download_completion_marker", {})
        .get("details", {})
        .get("marker_claims")
    )
    if marker_claims is not None and marker != marker_claims:
        raise RawBaselineMismatch("Raw provenance marker differs from Phase 2.5 evidence")
    source_fingerprint = checkpoint_digest(
        {
            "inventory_sha256": inventory_sha256,
            "metadata_sha256": current_metadata_hashes,
            "source_sha256": marker.get("download_sha256", ""),
        }
    )
    if inventory_sha256 != baseline.get("inventory_sha256") or source_fingerprint != baseline.get(
        "source_fingerprint"
    ):
        raise RawBaselineMismatch("Raw inventory/source fingerprint differs from Phase 2.5")
    criteria = baseline.get("criteria", {})
    if criteria and any(
        details.get("status") not in {"PASS", True}
        for name, details in criteria.items()
        if name != "train_test_duplicate_content"
    ):
        raise RawBaselineMismatch("Phase 2.5 raw-integrity criteria are not all passing")
    return hashes, inventory_sha256, source_fingerprint


def _load_classes(config: PipelineConfig, official_classes: list[str]) -> list[str]:
    """Validate configured full class mapping against official classes.txt order."""
    try:
        class_document = yaml.safe_load(config.class_selection_file.read_text(encoding="utf-8"))
        classes = class_document["classes"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid class selection file: {exc}") from exc
    if config.dataset_scope != "full" or classes != official_classes:
        raise ValueError("Full-scope class list must exactly match official classes.txt order")
    return classes


def _config_hash(config_path: Path, config: PipelineConfig) -> str:
    """Bind exact config and class-list bytes into one stable digest."""
    payload = {
        "config_sha256": _sha256_file(config_path),
        "class_selection_sha256": _sha256_file(config.class_selection_file),
    }
    return checkpoint_digest(payload)


def _split_rows(
    classes: list[str],
    train_ids: list[str],
    test_ids: list[str],
    hashes: dict[str, str],
    version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[list[str]]]:
    """Create official-test-preserving rows and provenance for excluded train IDs."""
    class_to_id = {name: index for index, name in enumerate(classes)}
    test_hashes: dict[str, list[str]] = defaultdict(list)
    for image_id in test_ids:
        test_hashes[hashes[f"images/{image_id}.jpg"]].append(image_id)
    exclusions: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    train_path_set = {f"images/{image_id}.jpg" for image_id in train_ids}
    test_path_set = {f"images/{image_id}.jpg" for image_id in test_ids}
    if len(train_path_set) != len(train_ids) or len(test_path_set) != len(test_ids):
        raise ValueError("Official split metadata contains duplicate image IDs")
    for split, image_ids in (("train", train_ids), ("test", test_ids)):
        for image_id in image_ids:
            relative_path = f"images/{image_id}.jpg"
            digest = hashes[relative_path]
            class_key = image_id.split("/", 1)[0]
            matching_test = test_hashes.get(digest, []) if split == "train" else []
            if matching_test:
                exclusions.append(
                    {
                        "image_id": image_id,
                        "class_key": class_key,
                        "original_partition": "train",
                        "matching_test_image_ids": ";".join(matching_test),
                        "sha256": digest,
                        "reason": "exact_sha256_match_in_official_test",
                    }
                )
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "relative_path": relative_path,
                    "source_dataset": "food101",
                    "class_id": class_to_id[class_key],
                    "class_key": class_key,
                    "split": split,
                    "checksum": digest,
                    "validation_status": "valid",
                    "dataset_version": version,
                    "capture_group": "",
                }
            )
    all_hashes: dict[str, set[str]] = defaultdict(set)
    duplicates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        all_hashes[row["checksum"]].add(row["split"])
        duplicates[row["checksum"]].append(row["image_id"])
    if any(len(partitions) > 1 for partitions in all_hashes.values()):
        raise ValueError("Clean split still has exact cross-partition duplicates")
    label_risks = [
        image_ids
        for image_ids in duplicates.values()
        if len(image_ids) > 1 and len({image_id.split("/", 1)[0] for image_id in image_ids}) > 1
    ]
    if {row["relative_path"] for row in rows if row["split"] == "train"} - train_path_set:
        raise ValueError("Clean train contains an ID outside official train")
    if {row["relative_path"] for row in rows if row["split"] == "test"} != test_path_set:
        raise ValueError("Clean test differs from official test IDs")
    return rows, exclusions, label_risks


def _split_fingerprint(
    rows: list[dict[str, Any]], exclusions: list[dict[str, str]], config_hash: str
) -> str:
    """Return a deterministic digest of ordered split rows, exclusions and config."""
    payload = {
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
        "exclusions": sorted(exclusions, key=lambda entry: entry["image_id"]),
    }
    return checkpoint_digest(payload)


def _validate_existing(
    output: Path,
    raw_root: Path,
    expected_source: str,
    expected_config: str,
    expected_split: str,
    expected_rows: list[dict[str, Any]],
    classes: list[str],
) -> dict[str, Any]:
    """Validate a prior immutable output and return SKIPPED only on exact evidence match."""
    try:
        metadata = json.loads((output / "dataset_metadata.json").read_text(encoding="utf-8"))
        with (output / "dataset_manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        audit_manifest(rows, raw_root)
        labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
        split_report = json.loads(
            (output / "split_quality_report.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise FileExistsError(
            f"Existing split output is invalid; refusing to overwrite: {exc}"
        ) from exc
    if (
        metadata.get("source_fingerprint") != expected_source
        or metadata.get("config_hash") != expected_config
        or metadata.get("split_fingerprint") != expected_split
        or metadata.get("complete") is not True
        or metadata.get("clean_split_status") != "PASS"
        or split_report.get("clean_split_status") != "PASS"
        or split_report.get("split_fingerprint") != expected_split
    ):
        raise FileExistsError(
            "Existing split output has a different fingerprint/config; refusing overwrite"
        )
    identity_fields = ("image_id", "relative_path", "class_id", "class_key", "split", "checksum")
    existing_rows = [tuple(row[field] for field in identity_fields) for row in rows]
    expected_values = [tuple(str(row[field]) for field in identity_fields) for row in expected_rows]
    if existing_rows != expected_values:
        raise FileExistsError("Existing split manifest differs from the verified partition")
    if (
        labels.get("dataset_version") != metadata.get("dataset_version")
        or [item.get("class_key") for item in labels.get("labels", [])] != classes
    ):
        raise FileExistsError("Existing split labels are invalid")
    return {
        **split_report,
        "state": "SKIPPED",
        "split_fingerprint": metadata["split_fingerprint"],
    }


def build_clean_split(
    config_path: Path,
    baseline_report_path: Path = REPORT_PATH,
    *,
    publish: bool = True,
) -> dict[str, Any]:
    """Verify raw against Phase 2.5, build clean v2 in staging, then publish atomically."""
    config_path = Path(config_path).resolve()
    workspace = config_path.parents[1]
    config = load_pipeline_config(config_path, workspace)
    baseline = _read_report(baseline_report_path)
    raw_root = config.raw_root
    report_root = config.report_root
    output = config.manifest_root
    if (
        output == raw_root
        or output.is_relative_to(raw_root)
        or output.is_relative_to(workspace / "data" / "splits" / "v1")
    ):
        raise ValueError("Split output overlaps protected data")
    official_classes = (raw_root / "meta/classes.txt").read_text(encoding="utf-8").splitlines()
    classes = _load_classes(config, official_classes)
    config_hash = _config_hash(config_path, config)
    output_lock = resource_lock(output, config.lock_timeout_seconds) if publish else nullcontext()
    with output_lock:
        checkpoint_path = Path(baseline_report_path).parent / "food101_checkpoint.json"
        hashes, inventory_sha256, source_fingerprint = _inventory(
            raw_root, baseline, checkpoint_path
        )
        train_ids = (raw_root / "meta/train.txt").read_text(encoding="utf-8").splitlines()
        test_ids = (raw_root / "meta/test.txt").read_text(encoding="utf-8").splitlines()
        if len(train_ids) != baseline.get("train_count") or len(test_ids) != baseline.get(
            "test_count"
        ):
            raise RawBaselineMismatch("Official train/test counts differ from Phase 2.5")
        rows, exclusions, label_risks = _split_rows(
            classes, train_ids, test_ids, hashes, config.split_version
        )
        split_fingerprint = _split_fingerprint(rows, exclusions, config_hash)
        if output.exists():
            existing = _validate_existing(
                output,
                raw_root,
                source_fingerprint,
                config_hash,
                split_fingerprint,
                rows,
                classes,
            )
            existing.update({"source_fingerprint": source_fingerprint, "output": str(output)})
            report_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(report_root / "clean_split_v2_report.json", existing)
            return existing
        stats = {
            "train": sum(row["split"] == "train" for row in rows),
            "test": sum(row["split"] == "test" for row in rows),
        }
        per_class = {
            split: dict(Counter(row["class_key"] for row in rows if row["split"] == split))
            for split in ("train", "test")
        }
        split_report = {
            "schema_version": 1,
            "split_version": config.split_version,
            "raw_integrity_status": "PASS",
            "official_split_leakage_status": "FAIL" if exclusions else "PASS",
            "clean_split_status": "PASS",
            "cross_split_duplicate_hash_count": len({entry["sha256"] for entry in exclusions}),
            "clean_cross_split_duplicate_hash_count": 0,
            "source_fingerprint": source_fingerprint,
            "inventory_sha256": inventory_sha256,
            "config_hash": config_hash,
            "split_fingerprint": split_fingerprint,
            "official_train_count": len(train_ids),
            "official_test_count": len(test_ids),
            "split_counts": stats,
            "per_class_counts": per_class,
            "excluded_train_images": exclusions,
            "duplicate_policy": {
                "version": SPLIT_POLICY_VERSION,
                "preserve_official_test": True,
                "exclude_train_hashes_in_test": True,
                "create_validation": False,
            },
            "label_inconsistency_risk": label_risks,
            "duplicate_findings": [
                {
                    "sha256": digest,
                    "image_ids": sorted(
                        path.removeprefix("images/").removesuffix(".jpg")
                        for path, image_hash in hashes.items()
                        if image_hash == digest
                    ),
                }
                for digest, count in Counter(hashes.values()).items()
                if count > 1
            ],
            "verified_at_utc": datetime.now(UTC).isoformat(),
        }
        if not publish:
            return split_report
        report_root.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        try:
            for name, selected in [("dataset_manifest", rows)] + [
                (split, [row for row in rows if row["split"] == split])
                for split in ("train", "test")
            ]:
                atomic_write_csv(staging / f"{name}.csv", selected, FIELDS)
            labels = {
                "schema_version": "2.0",
                "dataset_version": config.split_version,
                "labels": [
                    {
                        "class_id": index,
                        "class_key": key,
                        "display_name_vi": key,
                        "display_name_en": key.replace("_", " "),
                    }
                    for index, key in enumerate(classes)
                ],
            }
            atomic_write_json(staging / "labels.json", labels)
            metadata = {
                "schema_version": "2.0",
                "dataset_version": config.split_version,
                "root": os.path.relpath(raw_root, output).replace(os.sep, "/"),
                "source_fingerprint": source_fingerprint,
                "inventory_sha256": inventory_sha256,
                "expected_file_count": len(rows),
                "verified_file_count": len(rows),
                "config_hash": config_hash,
                "split_fingerprint": split_fingerprint,
                "split_counts": stats,
                "per_class_counts": per_class,
                "excluded_train_images": exclusions,
                "duplicate_policy": split_report["duplicate_policy"],
                "duplicate_findings": split_report["duplicate_findings"],
                "class_mapping": labels["labels"],
                "clean_split_status": "PASS",
                "official_split_leakage_status": split_report["official_split_leakage_status"],
                "complete": True,
            }
            atomic_write_json(staging / "split_quality_report.json", split_report)
            atomic_write_json(staging / "dataset_metadata.json", metadata)
            with (staging / "dataset_manifest.csv").open(encoding="utf-8", newline="") as stream:
                staged_rows = list(csv.DictReader(stream))
            if audit_manifest(staged_rows, raw_root)["split_counts"] != stats:
                raise ValueError("Staged split validation statistics differ")
            final_hashes, final_inventory, final_source = _inventory(
                raw_root, baseline, checkpoint_path
            )
            if (
                final_inventory != inventory_sha256
                or final_source != source_fingerprint
                or final_hashes != hashes
            ):
                raise RawBaselineMismatch("Raw changed between verification and split publish")
            if output.exists():
                raise FileExistsError("Split target appeared during build; refusing overwrite")
            staging.rename(output)
            atomic_write_json(report_root / "clean_split_v2_report.json", split_report)
            return {**split_report, "state": "PASS", "output": str(output)}
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
