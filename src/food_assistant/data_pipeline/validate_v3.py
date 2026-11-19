"""Read-only full-scope validation of the immutable Food-101 v3 split."""

from __future__ import annotations

import json
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from food_assistant.data_pipeline.atomic import (
    atomic_write_text,
    checkpoint_digest,
    resource_lock,
)
from food_assistant.data_pipeline.config import PipelineConfig, load_pipeline_config
from food_assistant.data_pipeline.split_v3 import (
    V2_CONFIG_PATH,
    _clean_train_rows,
    _fingerprint_v3,
    _make_rows,
    _read_csv,
    _read_json,
    _stratified_assignment,
    _validate_v2_source,
)
from food_assistant.data_pipeline.splits import REPORT_PATH, _config_hash

FULL_V3_REPORT = Path("artifacts/reports/elt/phase_4_full_v3_validation.md")
FULL_V3_RETRY_REPORT = Path("artifacts/reports/elt/phase_4_full_v3_validation_retry.md")
FULL_V3_FINAL_REPORT = Path("artifacts/reports/elt/phase_4_full_v3_validation_final.md")


def classify_scope(actual_classes: list[str], expected_classes: list[str]) -> str:
    """Classify a complete, partial, or incompatible class scope without assuming counts."""
    actual = set(actual_classes)
    expected = set(expected_classes)
    if actual == expected and len(actual_classes) == len(expected_classes):
        return "PASS"
    if actual and actual < expected:
        return "INCOMPLETE"
    return "FAIL"


def _read_json_list(path: Path) -> list[Any]:
    """Read a JSON list artifact and reject a malformed or wrong-shaped file."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read list evidence {path}: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"List evidence has the wrong schema: {path}")
    return value


def inspect_v3_rows(
    rows: list[dict[str, str]],
    classes: list[str],
    raw_root: Path,
    verified_hashes: dict[str, str],
) -> dict[str, Any]:
    """Check manifest semantics and exact/capture-group isolation using trusted hashes."""
    issues: list[str] = []
    ids: set[str] = set()
    mappings = {name: index for index, name in enumerate(classes)}
    split_counts: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "val", "test")
    }
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    hash_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    capture_splits: dict[str, set[str]] = defaultdict(set)
    capture_rows: dict[str, list[str]] = defaultdict(list)
    missing_paths: list[str] = []

    for row in rows:
        image_id = row.get("image_id", "")
        relative = Path(row.get("relative_path", ""))
        split = row.get("split", "")
        class_key = row.get("class_key", "")
        if (
            not image_id
            or image_id in ids
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in row.get("relative_path", "")
            or split not in per_class
            or class_key not in mappings
            or row.get("class_id") != str(mappings.get(class_key, -1))
            or relative.as_posix() != f"images/{image_id}.jpg"
            or row.get("dataset_version") != "v3"
        ):
            issues.append(f"invalid_manifest_row:{image_id}")
            continue
        ids.add(image_id)
        absolute = (Path(raw_root) / relative).resolve()
        if not absolute.is_relative_to(Path(raw_root).resolve()) or not absolute.is_file():
            missing_paths.append(image_id)
        digest = row.get("checksum", "")
        if verified_hashes.get(relative.as_posix()) != digest:
            issues.append(f"checksum_evidence_mismatch:{image_id}")
        hashes_by_split[split].add(digest)
        hash_rows[digest].append(row)
        split_counts[split] += 1
        per_class[split][class_key] += 1
        capture_group = row.get("capture_group", "")
        if capture_group:
            capture_splits[capture_group].add(split)
            capture_rows[capture_group].append(image_id)

    cross_hashes = {
        digest: [row["image_id"] for row in matches]
        for digest, matches in hash_rows.items()
        if len({row["split"] for row in matches}) > 1
    }
    same_split_duplicates = {
        digest: {
            "split": matches[0]["split"],
            "image_ids": sorted(row["image_id"] for row in matches),
        }
        for digest, matches in hash_rows.items()
        if len(matches) > 1 and len({row["split"] for row in matches}) == 1
    }
    capture_leaks = {
        group: sorted(capture_rows[group])
        for group, splits in capture_splits.items()
        if len(splits) > 1
    }
    if missing_paths:
        issues.append(f"missing_paths:{len(missing_paths)}")
    if cross_hashes:
        issues.append(f"cross_partition_exact_duplicates:{len(cross_hashes)}")
    if capture_leaks:
        issues.append(f"cross_partition_capture_groups:{len(capture_leaks)}")
    if set(per_class["train"]) != set(classes) or set(per_class["val"]) != set(classes):
        issues.append("class_missing_from_train_or_validation")
    return {
        "status": "FAIL" if issues else "PASS",
        "errors": issues,
        "image_count": len(rows),
        "class_count": len(set(row.get("class_key") for row in rows)),
        "split_counts": {split: split_counts.get(split, 0) for split in per_class},
        "per_class_counts": {
            split: {name: per_class[split].get(name, 0) for name in classes}
            for split in per_class
        },
        "unique_image_ids": len(ids),
        "missing_image_count": len(missing_paths),
        "cross_partition_exact_hash_count": len(cross_hashes),
        "cross_partition_exact_hashes": cross_hashes,
        "same_split_duplicate_group_count": len(same_split_duplicates),
        "same_split_duplicates": same_split_duplicates,
        "capture_group_count": len(capture_splits),
        "capture_group_status": (
            "CHECKED" if capture_splits else "NOT_EVALUABLE_NO_GROUP_IDS"
        ),
        "cross_partition_capture_group_count": len(capture_leaks),
        "cross_partition_capture_groups": capture_leaks,
    }


def _bundle_status(
    config: PipelineConfig,
    config_path: Path,
    raw_baseline_path: Path,
) -> dict[str, Any]:
    """Validate v3 metadata, policy, lineage, ledger, manifest, and v2 reconciliation."""
    source = _validate_v2_source(config, V2_CONFIG_PATH, raw_baseline_path)
    v3_root = config.manifest_root
    metadata = _read_json(v3_root / "dataset_metadata.json")
    quality = _read_json(v3_root / "quality_report.json")
    labels = _read_json(v3_root / "labels.json")
    exclusions = _read_json_list(v3_root / "excluded_images.json")
    conflicts = _read_json_list(v3_root / "label_conflicts.json")
    rows = _read_csv(v3_root / "dataset_manifest.csv")
    expected_classes = source["class_names"]
    actual_labels = [entry.get("class_key") for entry in labels.get("labels", [])]
    scope_status = classify_scope(actual_labels, expected_classes)
    expected_config_hash = _config_hash(config_path, config)

    candidates, expected_exclusions, expected_conflicts = _clean_train_rows(
        source["train_rows"], source["test_rows"]
    )
    assignments = _stratified_assignment(
        candidates, config.seed, config.val_fraction, expected_classes
    )
    expected_rows = _make_rows(
        candidates, source["test_rows"], assignments, config.split_version
    )
    expected_fingerprint = _fingerprint_v3(
        expected_rows,
        expected_exclusions,
        source["metadata"]["split_fingerprint"],
        expected_config_hash,
    )
    row_metrics = inspect_v3_rows(rows, expected_classes, config.raw_root, source["raw_hashes"])
    expected_policy = {
        "version": 1,
        "same_label_duplicates": "keep lexicographically smallest image_id per SHA-256",
        "cross_label_duplicates": "quarantine every reference; no label inference",
        "test": "preserve source v2 IDs, labels, order and checksums",
        "validation": "stratified after SHA-256 group collapse",
    }
    identity_fields = ("image_id", "split", "class_id", "checksum")
    row_identity = [tuple(row[field] for field in identity_fields) for row in rows]
    expected_identity = [
        tuple(str(row[field]) for field in identity_fields) for row in expected_rows
    ]
    files_match = True
    for split in ("train", "val", "test"):
        split_rows = _read_csv(v3_root / f"{split}.csv")
        selected = [row for row in rows if row["split"] == split]
        if split_rows != selected:
            files_match = False

    metadata_valid = all(
        (
            metadata.get("schema_version") == "3.0",
            metadata.get("manifest_schema_version") == "3.0",
            metadata.get("dataset_version") == "v3",
            metadata.get("complete") is True,
            metadata.get("quality_gates", {}).get("overall") == "PASS",
            metadata.get("source_split_version") == "v2",
            metadata.get("source_split_fingerprint")
            == source["metadata"].get("split_fingerprint"),
            metadata.get("source_fingerprint") == source["source_fingerprint"],
            metadata.get("inventory_sha256") == source["inventory_sha256"],
            metadata.get("config_hash") == expected_config_hash,
            metadata.get("v3_fingerprint") == expected_fingerprint,
            metadata.get("seed") == config.seed,
            metadata.get("validation_fraction") == config.val_fraction,
            metadata.get("deduplication_policy") == expected_policy,
            metadata.get("validation_algorithm", {}).get("name")
            == config.validation_algorithm,
            metadata.get("class_mapping") == labels.get("labels"),
            labels.get("schema_version") == "3.0",
            labels.get("dataset_version") == "v3",
            actual_labels == expected_classes,
            metadata.get("excluded_images") == expected_exclusions == exclusions,
            metadata.get("label_conflict_groups") == expected_conflicts == conflicts,
            quality.get("quality_gates") == metadata.get("quality_gates"),
            quality.get("v3_fingerprint") == expected_fingerprint,
            quality.get("source_fingerprint") == source["source_fingerprint"],
            quality.get("inventory_sha256") == source["inventory_sha256"],
            quality.get("config_hash") == expected_config_hash,
            quality.get("excluded_images") == expected_exclusions,
            quality.get("label_conflict_groups") == expected_conflicts,
            quality.get("deduplication_policy") == expected_policy,
            quality.get("seed") == config.seed,
            quality.get("validation_fraction") == config.val_fraction,
            quality.get("cross_partition_duplicate_hash_count") == 0,
            quality.get("train_duplicate_hash_count") == 0,
            metadata.get("split_counts") == row_metrics["split_counts"],
            metadata.get("per_class_counts") == row_metrics["per_class_counts"],
            row_identity == expected_identity,
            files_match,
        )
    )
    if not metadata_valid:
        row_metrics["errors"].append("v3_metadata_or_lineage_mismatch")
        row_metrics["status"] = "FAIL"

    official_train_ids = (
        config.raw_root / "meta" / "train.txt"
    ).read_text(encoding="utf-8").splitlines()
    official_test_ids = (
        config.raw_root / "meta" / "test.txt"
    ).read_text(encoding="utf-8").splitlines()
    output_ids = {row["image_id"] for row in rows}
    test_ids = [row["image_id"] for row in rows if row["split"] == "test"]
    v2_test_ids = [row["image_id"] for row in source["test_rows"]]
    official_train_excluded_by_v2 = source["metadata"].get("excluded_train_images", [])
    official_train_retained_ids = {
        row["image_id"] for row in source["train_rows"]
    }
    v3_train_val_ids = {
        row["image_id"] for row in rows if row["split"] in {"train", "val"}
    }
    removed_from_v2_train = set(official_train_retained_ids) - v3_train_val_ids
    v3_excluded_ids = {entry["image_id"] for entry in expected_exclusions}
    reconcile = {
        "official_train_ids": len(official_train_ids),
        "official_test_ids": len(official_test_ids),
        "official_train_excluded_by_v2": len(official_train_excluded_by_v2),
        "v2_train_ids": len(source["train_rows"]),
        "v3_train_and_validation_ids": len(v3_train_val_ids),
        "v3_train_ids_excluded_by_policy": len(v3_excluded_ids),
        "v3_test_retained_ids": len(test_ids),
        "v3_test_matches_v2_order_and_ids": test_ids == v2_test_ids,
        "v3_test_missing_from_official": sorted(set(official_test_ids) - set(test_ids)),
        "v3_test_not_in_official": sorted(set(test_ids) - set(official_test_ids)),
        "v2_train_ids_removed_after_v2": len(removed_from_v2_train),
        "v2_train_removals_have_cleanup_provenance": removed_from_v2_train == v3_excluded_ids,
        "v3_image_ids_outside_expected_sources": sorted(
            output_ids - set(official_train_ids) - set(official_test_ids)
        ),
        "official_train_ids_retained_in_v3": len(v3_train_val_ids),
        "official_test_ids_retained_in_v3": len(test_ids),
    }
    reconcile_pass = (
        reconcile["v3_test_matches_v2_order_and_ids"]
        and not reconcile["v3_test_missing_from_official"]
        and not reconcile["v3_test_not_in_official"]
        and reconcile["v2_train_removals_have_cleanup_provenance"]
        and not reconcile["v3_image_ids_outside_expected_sources"]
        and reconcile["v3_train_ids_excluded_by_policy"] == len(expected_exclusions)
    )
    if not reconcile_pass:
        row_metrics["errors"].append("official_metadata_reconciliation_failed")
        row_metrics["status"] = "FAIL"
    if scope_status == "INCOMPLETE":
        non_scope_errors = {
            error
            for error in row_metrics["errors"]
            if error
            not in {
                "class_missing_from_train_or_validation",
                "v3_metadata_or_lineage_mismatch",
            }
        }
        if not non_scope_errors:
            row_metrics["status"] = "INCOMPLETE"
            row_metrics["errors"].append(
                f"scope_incomplete:{len(actual_labels)}_of_{len(expected_classes)}_classes"
            )

    total_expected = sum(row_metrics["split_counts"].values())
    source_evidence = {
        "source_fingerprint": source["source_fingerprint"],
        "inventory_sha256": source["inventory_sha256"],
        "source_split_fingerprint": source["metadata"]["split_fingerprint"],
        "v3_fingerprint": expected_fingerprint,
        "config_hash": expected_config_hash,
        "policy_hash": checkpoint_digest(metadata.get("deduplication_policy", {})),
        "raw_evidence_source": (
            "Reused Phase 2.5 schema-2 checkpoint after current signature, metadata, "
            "inventory and source binding checks."
        ),
        "checkpoint_path": str(source["checkpoint_path"]),
        "v3_quality_report_path": str(v3_root / "quality_report.json"),
        "v3_metadata_path": str(v3_root / "dataset_metadata.json"),
        "phase_2_5_historical_overall": source["baseline"].get("overall"),
        "phase_2_5_historical_cross_split_duplicate_hashes": source["baseline"].get(
            "train_test_duplicate_hashes"
        ),
    }
    return {
        "status": row_metrics["status"],
        "scope_requested": "full Food-101 v3",
        "scope_actual": {
            "classes": row_metrics["class_count"],
            "expected_classes": len(expected_classes),
            "scope_status": scope_status,
            "images": total_expected,
            "expected_images_from_v3_policy": sum(
                metadata.get("split_counts", {}).values()
            ),
        },
        "metrics": row_metrics,
        "official_reconciliation": reconcile,
        "evidence": source_evidence,
        "cleanup_policy": metadata.get("deduplication_policy"),
        "excluded_image_count": len(expected_exclusions),
        "same_label_duplicate_groups_removed": quality.get(
            "same_label_duplicate_group_count"
        ),
        "cross_label_duplicate_groups_quarantined": quality.get(
            "cross_label_duplicate_group_count"
        ),
        "excluded_images": expected_exclusions,
        "label_conflict_groups": expected_conflicts,
        "images_verified_newly": 0,
        "images_skipped_by_verified_evidence": len(source["raw_hashes"]),
        "images_in_v3_not_individually_redecoded": total_expected,
        "decode_evidence_scope": (
            "Phase 2.5 cached full decode/SHA results reused after current file "
            "signatures and source fingerprints matched."
        ),
        "exif_warning_count": None,
        "exif_warning_note": (
            "Historical evidence does not store a dedicated EXIF/TIFF warning counter; "
            "no new full image decode was run."
        ),
        "near_duplicate_status": (
            "NOT_CHECKED; evidence and policy cover exact SHA-256 duplicates only"
        ),
        "test_isolation": (
            "PASS by v3 policy and implementation review; test IDs/order/checksums are "
            "preserved from v2 and validation derives only from train candidates."
        ),
        "publication_result": (
            "PASS+SKIPPED: existing immutable v3 artifact validated; no duplicate copy published."
        ),
        "published_version": str(v3_root),
        "warnings": [
            "Historical Phase 2.5 official train/test leakage FAIL is retained; it is not "
            "applied to the cleaned v3 partition.",
            "No near-duplicate or semantic duplicate audit is claimed.",
            "EXIF/TIFF warning count is unavailable in historical checkpoint evidence.",
            "No capture_group IDs are populated in v3 rows; capture-group independence is "
            "not evaluable.",
            "An initial validator attempt used an object reader for a JSON-list ledger; "
            "the reader was corrected and its FAIL diagnostic was retained separately.",
        ],
    }


def validate_full_v3(
    config_path: Path,
    baseline_report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    """Run read-only full-v3 validation using linked Phase 2.5/2.6/2.7 evidence."""
    resolved_config = Path(config_path).resolve()
    workspace = resolved_config.parents[1]
    config = load_pipeline_config(resolved_config, workspace)
    if config.dataset_scope != "full" or config.split_version != "v3":
        raise ValueError("Full-v3 validation requires the approved full v3 config")
    started = datetime.now(UTC)
    try:
        result = _bundle_status(config, resolved_config, Path(baseline_report_path))
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        message = str(exc)
        incomplete_markers = ("Cannot read", "Cannot load", "missing", "No such file")
        status = "INCOMPLETE" if any(marker in message for marker in incomplete_markers) else "FAIL"
        result = {
            "status": status,
            "scope_requested": "full Food-101 v3",
            "scope_actual": None,
            "metrics": None,
            "errors": [message],
            "warnings": [],
        }
    poc_report_path = workspace / "data/reports/elt/poc_v2/transform_quality_report.json"
    try:
        poc_report = json.loads(poc_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        poc_report = {}
    result["poc_reference"] = {
        "status": poc_report.get("status", "UNKNOWN"),
        "scope": poc_report.get("scope"),
        "source_fingerprint_matches_full_v3": (
            poc_report.get("source_fingerprint") == result.get("evidence", {}).get(
                "source_fingerprint"
            )
        ),
        "report_path": str(poc_report_path),
        "used_as_full_v3_evidence": False,
    }
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={workspace.as_posix()}", "rev-parse", "--short", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", f"safe.directory={workspace.as_posix()}", "status", "--porcelain"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "schema_version": "1.0",
        "run_id": f"full-v3-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "timestamp_utc": started.isoformat(),
        "source_commit": commit,
        "source_dirty": dirty,
        "input_split_version": "v3",
        "input_split_path": str(config.manifest_root),
        "status": result["status"],
        **result,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def write_validation_report(result: dict[str, Any], report_path: Path) -> Path:
    """Atomically write one immutable Markdown report outside dataset versions."""
    markdown_path = Path(report_path).resolve()
    report_root = markdown_path.parents[1]
    if report_root.name != "reports" or report_root.parent.name != "artifacts":
        raise ValueError("Validation report must remain under artifacts/reports")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with resource_lock(markdown_path, timeout_seconds=0):
        if markdown_path.exists():
            raise FileExistsError("Validation report already exists; refusing to overwrite")
        atomic_write_text(markdown_path, _render_markdown(result))
    return markdown_path


def _render_markdown(result: dict[str, Any]) -> str:
    """Render the full-v3 validation evidence as a concise Markdown report."""
    metrics = result.get("metrics") or {}
    counts = metrics.get("split_counts", {})
    actual = result.get("scope_actual") or {}
    evidence = result.get("evidence") or {}
    reconcile = result.get("official_reconciliation") or {}
    poc_reference = result.get("poc_reference") or {}
    lines = [
        "# Phase 4 — Full Validation on cleaned split v3",
        "",
        f"- Run ID: `{result.get('run_id')}`",
        f"- Timestamp UTC: `{result.get('timestamp_utc')}`",
        f"- Commit / dirty: `{result.get('source_commit')}` / `{result.get('source_dirty')}`",
        "- Scope requested / actual: full Food-101 v3 / "
        f"{actual.get('classes', 'unknown')} classes, {actual.get('images', 'unknown')} rows",
        f"- Status: **{result.get('status')}**",
        "- POC status (separate; not used as full evidence): "
        f"{poc_reference.get('status', 'UNKNOWN')}",
        f"- Input split: `{result.get('input_split_path')}` (v3)",
        "",
        "## Counts and reconciliation",
        "",
        "- Train / validation / test: "
        f"{counts.get('train', 0)} / {counts.get('val', 0)} / {counts.get('test', 0)}",
        "- Classes: "
        f"{actual.get('classes', 'unknown')} (expected "
        f"{actual.get('expected_classes', 'unknown')})",
        "- Official train IDs removed by v2: "
        f"{reconcile.get('official_train_excluded_by_v2', 'unknown')}",
        "- Additional v3 train exclusions: "
        f"{reconcile.get('v3_train_ids_excluded_by_policy', 'unknown')}",
        "- Official test retained: "
        f"{reconcile.get('v3_test_retained_ids', 'unknown')}; order/IDs match: "
        f"{reconcile.get('v3_test_matches_v2_order_and_ids', 'unknown')}",
        f"- Exclusion ledger entries: {result.get('excluded_image_count', 'unknown')}",
        "- Official train retained in v3: "
        f"{reconcile.get('official_train_ids_retained_in_v3', 'unknown')}; test retained: "
        f"{reconcile.get('official_test_ids_retained_in_v3', 'unknown')}",
        "- Same-label groups removed: "
        f"{result.get('same_label_duplicate_groups_removed', 'unknown')}; cross-label "
        f"groups quarantined: {result.get('cross_label_duplicate_groups_quarantined', 'unknown')}",
        "",
        "## Quality and evidence",
        "",
        "- Unique IDs: "
        f"{metrics.get('unique_image_ids', 'unknown')}; missing: "
        f"{metrics.get('missing_image_count', 'unknown')}",
        "- Exact cross-partition hash groups: "
        f"{metrics.get('cross_partition_exact_hash_count', 'unknown')}",
        "- Same-split exact duplicate groups: "
        f"{metrics.get('same_split_duplicate_group_count', 'unknown')}",
        "- Capture groups: "
        f"{metrics.get('capture_group_count', 'unknown')}; cross-partition: "
        f"{metrics.get('cross_partition_capture_group_count', 'unknown')}",
        f"- Capture-group audit: {metrics.get('capture_group_status', 'unknown')}",
        f"- Source fingerprint: `{evidence.get('source_fingerprint')}`",
        f"- Inventory SHA-256: `{evidence.get('inventory_sha256')}`",
        f"- v2 fingerprint: `{evidence.get('source_split_fingerprint')}`",
        f"- v3 fingerprint: `{evidence.get('v3_fingerprint')}`",
        f"- Evidence reuse: {result.get('decode_evidence_scope')}",
        f"- Near duplicate audit: {result.get('near_duplicate_status')}",
        f"- Test isolation: {result.get('test_isolation')}",
        f"- Publication: {result.get('publication_result')}",
        f"- Existing validated version: `{result.get('published_version')}`",
        f"- Raw checkpoint: `{evidence.get('checkpoint_path')}`",
        "- v3 metadata/report: "
        f"`{evidence.get('v3_metadata_path')}` / `{evidence.get('v3_quality_report_path')}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in result.get("warnings", []))
    lines.extend(["", "## Errors", ""])
    errors = metrics.get("errors", result.get("errors", []))
    lines.extend(f"- {error}" for error in errors)
    if not errors:
        lines.append("None.")
    per_class = metrics.get("per_class_counts", {})
    if per_class:
        lines.extend(
            [
                "",
                "## Per-class distribution",
                "",
                "| Class | Train | Validation | Test |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {class_key} | {per_class['train'][class_key]} | "
            f"{per_class['val'][class_key]} | {per_class['test'][class_key]} |"
            for class_key in per_class["train"]
        )
    exclusions = result.get("excluded_images", [])
    if exclusions:
        lines.extend(
            [
                "",
                "## Cleanup provenance",
                "",
                "| Image ID | Class | Reason | Representative/matches |",
                "|---|---|---|---|",
            ]
        )
        lines.extend(
            "| {image_id} | {class_key} | {reason} | {matches} |".format(
                **entry,
                matches=entry.get("representative_image_id")
                or ", ".join(entry.get("matching_image_ids", [])),
            )
            for entry in exclusions
        )
    return "\n".join(lines) + "\n"
