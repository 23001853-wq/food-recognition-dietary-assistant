"""Quality-gated, immutable POC Food-101 manifest transformation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from food_assistant.data_pipeline.atomic import (
    atomic_write_csv,
    atomic_write_json,
    checkpoint_digest,
    resource_lock,
)
from food_assistant.data_pipeline.config import load_pipeline_config
from food_assistant.data_pipeline.splits import (
    REPORT_PATH,
    RawBaselineMismatch,
    _inventory,
    _read_report,
)
from food_assistant.ml.datasets import FIELDS, audit_manifest

QUALITY_POLICY_VERSION = "poc-transform-1"


class TransformQualityError(ValueError):
    """Represent a failed or incomplete Transform quality gate."""


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one source or output file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_class_selection(path: Path, official: list[str]) -> list[str]:
    """Validate the configured class selection while retaining its declared order."""
    try:
        selection = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        classes = selection["classes"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise TransformQualityError(f"Invalid class selection: {exc}") from exc
    if (
        not isinstance(classes, list)
        or not classes
        or len(classes) != len(set(classes))
        or not set(classes).issubset(official)
    ):
        raise TransformQualityError("Class selection is empty, duplicated, or not official")
    return classes


def _read_ids(raw_root: Path, split: str, selected: set[str]) -> list[str]:
    """Read an official split and return unique IDs for selected classes."""
    lines = (raw_root / "meta" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    selected_ids = [line for line in lines if line.split("/", 1)[0] in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise TransformQualityError(f"Official {split} metadata contains duplicate image IDs")
    return selected_ids


def _partition_train(
    train_ids: list[str], hashes: dict[str, str], seed: int, fraction: float
) -> dict[str, str]:
    """Assign train IDs to train/val by deterministic per-class SHA-256 groups."""
    by_class: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for image_id in train_ids:
        class_key = image_id.split("/", 1)[0]
        digest = hashes[f"images/{image_id}.jpg"]
        by_class[class_key][digest].append(image_id)
    assignments: dict[str, str] = {}
    for class_key, grouped_ids in by_class.items():
        digests = sorted(grouped_ids)
        if len(digests) < 2:
            raise TransformQualityError(f"Not enough independent train groups for {class_key}")
        try:
            train_hashes, val_hashes = train_test_split(
                digests,
                test_size=fraction,
                random_state=seed,
                shuffle=True,
            )
        except ValueError as exc:
            message = f"Cannot stratify validation for {class_key}: {exc}"
            raise TransformQualityError(message) from exc
        for digest in train_hashes:
            for image_id in grouped_ids[digest]:
                assignments[image_id] = "train"
        for digest in val_hashes:
            for image_id in grouped_ids[digest]:
                assignments[image_id] = "val"
    return assignments


def _make_rows(
    raw_root: Path,
    classes: list[str],
    train_ids: list[str],
    test_ids: list[str],
    hashes: dict[str, str],
    version: str,
    seed: int,
    fraction: float,
) -> list[dict[str, Any]]:
    """Create rows, preserving official test and grouping equal hashes in train."""
    assignment = _partition_train(train_ids, hashes, seed, fraction)
    class_to_id = {key: index for index, key in enumerate(classes)}
    rows: list[dict[str, Any]] = []
    for split, image_ids in (("train", train_ids), ("test", test_ids)):
        for image_id in image_ids:
            class_key = image_id.split("/", 1)[0]
            relative_path = f"images/{image_id}.jpg"
            digest = hashes[relative_path]
            if split == "train":
                partition = assignment[image_id]
            else:
                partition = "test"
            image_path = raw_root / relative_path
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                oriented = ImageOps.exif_transpose(image)
                oriented.load()
                if oriented.width <= 0 or oriented.height <= 0:
                    raise TransformQualityError(f"Invalid image orientation/dimensions: {image_id}")
            rows.append(
                {
                    "image_id": image_id,
                    "relative_path": relative_path,
                    "source_dataset": "food101",
                    "class_id": class_to_id[class_key],
                    "class_key": class_key,
                    "split": partition,
                    "checksum": digest,
                    "validation_status": "valid",
                    "dataset_version": version,
                    "capture_group": "",
                }
            )
    return rows


def _fingerprint(rows: list[dict[str, Any]], config_hash: str, source: str) -> str:
    """Bind the ordered split rows, configuration, and source evidence."""
    values = sorted(
        (row["image_id"], row["split"], row["class_id"], row["checksum"]) for row in rows
    )
    return checkpoint_digest({"rows": values, "config_hash": config_hash, "source": source})


def _existing_output(
    output: Path, raw_root: Path, expected_fingerprint: str, config_hash: str
) -> dict[str, Any]:
    """Validate an existing version and return a skip result only for exact matching evidence."""
    try:
        metadata = json.loads((output / "dataset_metadata.json").read_text(encoding="utf-8"))
        report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
        with (output / "dataset_manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        audit_manifest(rows, raw_root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        message = f"Existing Transform version is invalid; refusing overwrite: {exc}"
        raise FileExistsError(message) from exc
    if (
        metadata.get("split_fingerprint") != expected_fingerprint
        or metadata.get("config_hash") != config_hash
        or metadata.get("complete") is not True
        or report.get("status") != "PASS"
    ):
        raise FileExistsError("Existing Transform version conflicts with current evidence")
    return {**report, "state": "SKIPPED", "split_fingerprint": expected_fingerprint}


def build_poc_transform(
    config_path: Path,
    baseline_report_path: Path = REPORT_PATH,
    *,
    publish: bool = True,
) -> dict[str, Any]:
    """Verify full raw evidence and build a new immutable POC manifest version."""
    config_path = Path(config_path).resolve()
    workspace = config_path.parents[1]
    config = load_pipeline_config(config_path, workspace)
    if config.dataset_scope != "poc" or config.split_version != "poc-v2":
        raise TransformQualityError("This Transform entry point requires POC split version poc-v2")
    raw_root = config.raw_root
    output = config.manifest_root
    protected = [raw_root, workspace / "data" / "splits" / "v1"]
    if any(output == item or output.is_relative_to(item) for item in protected):
        raise TransformQualityError("Output overlaps raw data or frozen v1")
    baseline = _read_report(baseline_report_path)
    checkpoint_path = Path(baseline_report_path).parent / "food101_checkpoint.json"
    classes = _read_class_selection(
        config.class_selection_file,
        (raw_root / "meta/classes.txt").read_text(encoding="utf-8").splitlines(),
    )
    config_hash = checkpoint_digest(
        {
            "config_sha256": _file_sha256(config_path),
            "classes_sha256": _file_sha256(config.class_selection_file),
            "policy": QUALITY_POLICY_VERSION,
        }
    )
    with resource_lock(output, config.lock_timeout_seconds) if publish else _null_context():
        hashes, inventory_sha256, source_fingerprint = _inventory(
            raw_root, baseline, checkpoint_path
        )
        official_classes = (raw_root / "meta/classes.txt").read_text(encoding="utf-8").splitlines()
        train_ids = _read_ids(raw_root, "train", set(classes))
        test_ids = _read_ids(raw_root, "test", set(classes))
        if baseline.get("overall") not in {"PASS", "FAIL"}:
            raise TransformQualityError("Raw verification baseline is incomplete")
        if len(official_classes) != baseline.get("class_count") or not source_fingerprint:
            raise RawBaselineMismatch("Raw source identity does not match verified baseline")
        expected_train_classes = {image_id.split("/", 1)[0] for image_id in train_ids}
        expected_test_classes = {image_id.split("/", 1)[0] for image_id in test_ids}
        if expected_train_classes != set(classes) or expected_test_classes != set(classes):
            message = "At least one selected class is absent from official metadata"
            raise TransformQualityError(message)
        rows = _make_rows(
            raw_root,
            classes,
            train_ids,
            test_ids,
            hashes,
            config.split_version,
            config.seed,
            config.val_fraction,
        )
        stats = audit_manifest(rows, raw_root)
        test_hashes = {row["checksum"] for row in rows if row["split"] == "test"}
        train_hashes = {row["checksum"] for row in rows if row["split"] in {"train", "val"}}
        if train_hashes & test_hashes:
            raise TransformQualityError("Exact duplicate crosses official test boundary")
        duplicate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            duplicate_rows[row["checksum"]].append(row)
        exact_duplicate_findings = [
            {
                "sha256": digest,
                "image_ids": sorted(row["image_id"] for row in matches),
                "split": matches[0]["split"],
            }
            for digest, matches in duplicate_rows.items()
            if len(matches) > 1
        ]
        fingerprint = _fingerprint(rows, config_hash, source_fingerprint)
        report = {
            "schema_version": "2.0",
            "scope": "poc",
            "status": "PASS",
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "source_fingerprint": source_fingerprint,
            "inventory_sha256": inventory_sha256,
            "config_hash": config_hash,
            "split_fingerprint": fingerprint,
            "policy_version": QUALITY_POLICY_VERSION,
            "metrics": {
                **stats,
                "verified_images": len(rows),
                "expected_images": len(rows),
                "classes": classes,
                "official_test_ids_preserved": len(test_ids),
                "cross_split_duplicate_hashes": 0,
                "duplicate_content_groups": exact_duplicate_findings,
            },
            "errors": [],
            "warnings": [
                "Raw Phase 2.5 overall status is FAIL only for full official split leakage; "
                "POC-selected partitions were independently checked."
            ]
            if baseline.get("overall") == "FAIL"
            else [],
        }
        if exact_duplicate_findings:
            report["warnings"].append(
                "Exact duplicate references within one partition were retained and reported."
            )
        if output.exists():
            existing = _existing_output(output, raw_root, fingerprint, config_hash)
            existing = {
                **report,
                "state": "SKIPPED",
                "output": str(output),
            }
            if publish:
                atomic_write_json(config.report_root / "transform_quality_report.json", existing)
            return existing
        if not publish:
            return {**report, "state": "PREFLIGHT"}
        output.parent.mkdir(parents=True, exist_ok=True)
        config.report_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        try:
            ordered_rows = sorted(rows, key=lambda row: (row["split"], row["image_id"]))
            atomic_write_csv(staging / "dataset_manifest.csv", ordered_rows, FIELDS)
            for split in ("train", "val", "test"):
                atomic_write_csv(
                    staging / f"{split}.csv",
                    [row for row in ordered_rows if row["split"] == split],
                    FIELDS,
                )
            labels = {
                "schema_version": "2.0",
                "dataset_version": config.split_version,
                "labels": [
                    {
                        "class_id": index,
                        "class_key": class_key,
                        "display_name_vi": class_key,
                        "display_name_en": class_key.replace("_", " "),
                    }
                    for index, class_key in enumerate(classes)
                ],
            }
            atomic_write_json(staging / "labels.json", labels)
            metadata = {
                "schema_version": "2.0",
                "dataset_version": config.split_version,
                "root": Path(__import__("os").path.relpath(raw_root, output)).as_posix(),
                "source_fingerprint": source_fingerprint,
                "inventory_sha256": inventory_sha256,
                "config_hash": config_hash,
                "split_fingerprint": fingerprint,
                "split_counts": stats["split_counts"],
                "class_mapping": labels["labels"],
                "quality_policy_version": QUALITY_POLICY_VERSION,
                "complete": True,
            }
            atomic_write_json(staging / "dataset_metadata.json", metadata)
            atomic_write_json(staging / "quality_report.json", report)
            with (staging / "dataset_manifest.csv").open(encoding="utf-8", newline="") as stream:
                staged_rows = list(csv.DictReader(stream))
            if audit_manifest(staged_rows, raw_root) != stats:
                raise TransformQualityError("Staged manifest audit differs from prepublish metrics")
            final_hashes, final_inventory, final_source = _inventory(
                raw_root, baseline, checkpoint_path
            )
            if (
                final_hashes != hashes
                or final_inventory != inventory_sha256
                or final_source != source_fingerprint
            ):
                raise RawBaselineMismatch("Raw source changed while publishing Transform artifacts")
            if output.exists():
                raise FileExistsError("Transform output appeared during publication")
            staging.rename(output)
            atomic_write_json(config.report_root / "transform_quality_report.json", report)
            return {**report, "state": "PASS", "output": str(output)}
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise


class _null_context:
    """Provide a no-op context manager for read-only preflight operation."""

    def __enter__(self) -> None:
        """Enter without acquiring a writer lock."""
        return None

    def __exit__(self, *args: Any) -> None:
        """Exit without suppressing exceptions."""
        return None
