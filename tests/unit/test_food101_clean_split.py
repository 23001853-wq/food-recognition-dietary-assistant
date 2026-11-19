"""Synthetic tests for immutable Food-101 clean split generation."""

import csv
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest
import yaml
from PIL import Image

from food_assistant.data_pipeline.atomic import checkpoint_digest
from food_assistant.data_pipeline.splits import (
    RawBaselineMismatch,
    _split_fingerprint,
    build_clean_split,
)
from food_assistant.ml.datasets import FIELDS, ManifestDataset


def _build_split_in_process(config_path, report_path, result_queue):
    """Publish a synthetic split from a competing operating-system process."""
    try:
        result_queue.put(build_clean_split(config_path, report_path)["state"])
    except BaseException as exc:
        result_queue.put(f"{type(exc).__name__}: {exc}")


def _make_fixture(tmp_path):
    """Create a tiny official-like dataset and a matching Phase 2.5 fingerprint report."""
    workspace = tmp_path
    raw = workspace / "data/raw/food-101"
    metadata = raw / "meta"
    metadata.mkdir(parents=True)
    classes = ["burger", "pizza"]
    train_ids = ["pizza/p1", "pizza/p3", "burger/b1", "burger/b2", "pizza/p2"]
    test_ids = ["pizza/t1", "pizza/t2"]
    (metadata / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (metadata / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (metadata / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    image_root = raw / "images"
    for image_id, color in (
        ("pizza/p1", (200, 20, 20)),
        ("pizza/p3", (200, 20, 20)),
        ("pizza/t1", (200, 20, 20)),
        ("pizza/t2", (20, 20, 200)),
        ("burger/b1", (20, 200, 20)),
        ("burger/b2", (20, 200, 20)),
        ("pizza/p2", (20, 200, 20)),
    ):
        target = image_root / f"{image_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(target, format="JPEG")
    marker = {"dataset": "food101", "download_sha256": "a" * 64}
    (raw / "download_metadata.json").write_text(json.dumps(marker), encoding="utf-8")
    paths = sorted(path.relative_to(raw).as_posix() for path in image_root.rglob("*.jpg"))
    hashes = {
        relative: hashlib.sha256((raw / relative).read_bytes()).hexdigest() for relative in paths
    }
    inventory = hashlib.sha256()
    for relative, digest in sorted(hashes.items()):
        inventory.update(f"{relative}:{digest}\n".encode())
    metadata_paths = [f"meta/{name}.txt" for name in ("classes", "train", "test")]
    metadata_hashes = {
        relative: hashlib.sha256((raw / relative).read_bytes()).hexdigest()
        for relative in metadata_paths
    }
    source = checkpoint_digest(
        {
            "inventory_sha256": inventory.hexdigest(),
            "metadata_sha256": metadata_hashes,
            "source_sha256": marker["download_sha256"],
        }
    )
    report = {
        "mode": "full",
        "filesystem_paths": paths,
        "actual_image_count": len(paths),
        "class_count": len(classes),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "metadata_sha256": metadata_hashes,
        "inventory_sha256": inventory.hexdigest(),
        "source_fingerprint": source,
    }
    report_path = workspace / "baseline.json"
    fingerprint_paths = sorted(paths + metadata_paths + ["download_metadata.json"])
    signatures = {}
    files = {}
    for relative in fingerprint_paths:
        stat = (raw / relative).stat()
        signature = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        signatures[relative] = signature
        if relative in hashes:
            files[relative] = {
                "signature": signature,
                "result": {"valid": True, "sha256": hashes[relative]},
            }
    filesystem_fingerprint = checkpoint_digest(signatures)
    report["filesystem_paths"] = fingerprint_paths
    report["filesystem_fingerprint"] = filesystem_fingerprint
    checkpoint_payload = {
        "schema_version": 2,
        "dataset_root": str(raw.resolve()),
        "input_hash": filesystem_fingerprint,
        "expected_files": sorted(paths),
        "files": files,
        "complete": True,
    }
    checkpoint_path = workspace / "food101_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {"payload": checkpoint_payload, "sha256": checkpoint_digest(checkpoint_payload)}
        ),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    class_path = workspace / "classes_full.yaml"
    class_path.write_text(yaml.safe_dump({"version": "test", "classes": classes}), encoding="utf-8")
    config_document = {
        "schema_version": 1,
        "paths": {
            "raw_root": str(raw),
            "staging_root": str(workspace / "staging"),
            "archive": str(workspace / "archive.tar.gz"),
            "manifest_root": str(workspace / "data/splits/v2"),
            "report_root": str(workspace / "data/reports/v2"),
        },
        "approval_file": str(workspace / "approval.json"),
        "class_selection_file": str(class_path),
        "dataset_scope": "full",
        "split_version": "v2",
        "seed": 42,
        "val_fraction": 0.2,
        "download": {"policy": "disabled", "retries": 0, "backoff_seconds": 0},
        "quality": {"require_full_integrity": True, "reject_cross_split_duplicates": True},
        "split_policy": {
            "preserve_official_test": True,
            "exclude_train_hashes_in_test": True,
            "create_validation": False,
            "reject_cross_partition_duplicates": True,
        },
        "locks": {"timeout_seconds": 30},
        "fixtures": {"enabled": False, "root": str(workspace / "fixtures")},
    }
    config_path = workspace / "food101_elt_full_v2.yaml"
    config_path.write_text(yaml.safe_dump(config_document), encoding="utf-8")
    return raw, config_path, report_path, report


def test_clean_split_excludes_all_train_leaks_and_keeps_official_test(tmp_path):
    """Exclude train hashes matching either test ID and retain train-only duplicate groups."""
    raw, config_path, report_path, baseline = _make_fixture(tmp_path)

    result = build_clean_split(config_path, report_path)

    assert result["split_counts"] == {"train": 3, "test": 2}
    assert [item["image_id"] for item in result["excluded_train_images"]] == [
        "pizza/p1",
        "pizza/p3",
    ]
    assert all(
        item["matching_test_image_ids"] == "pizza/t1"
        and item["reason"] == "exact_sha256_match_in_official_test"
        and item["sha256"]
        for item in result["excluded_train_images"]
    )
    assert result["label_inconsistency_risk"] == [["burger/b1", "burger/b2", "pizza/p2"]]
    assert result["official_split_leakage_status"] == "FAIL"
    dataset = ManifestDataset(tmp_path / "data/splits/v2", "test")
    assert {row["image_id"] for row in dataset.rows} == {"pizza/t1", "pizza/t2"}
    train = ManifestDataset(tmp_path / "data/splits/v2", "train")
    assert {row["image_id"] for row in train.rows} == {"burger/b1", "burger/b2", "pizza/p2"}
    assert train.root == raw
    assert baseline["source_fingerprint"] == result["source_fingerprint"]
    with (tmp_path / "data/splits/v2/dataset_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        manifest_rows = list(csv.DictReader(stream))
    metadata = json.loads(
        (tmp_path / "data/splits/v2/dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert (
        _split_fingerprint(
            manifest_rows, metadata["excluded_train_images"], metadata["config_hash"]
        )
        == metadata["split_fingerprint"]
    )


def test_raw_metadata_change_blocks_publish(tmp_path):
    """Refuse to publish when official source metadata no longer matches its fingerprint."""
    raw, config_path, report_path, _ = _make_fixture(tmp_path)
    (raw / "meta/train.txt").write_text("pizza/p1\n", encoding="utf-8")

    with pytest.raises(RawBaselineMismatch, match="Raw file signatures differ"):
        build_clean_split(config_path, report_path)

    assert not (tmp_path / "data/splits/v2").exists()


def test_invalid_checkpoint_fails_closed_without_success_metadata(tmp_path):
    """Reject corrupted Phase 2.5 evidence and never create a completion marker."""
    _, config_path, report_path, _ = _make_fixture(tmp_path)
    checkpoint_path = tmp_path / "food101_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RawBaselineMismatch, match="incomplete or does not bind"):
        build_clean_split(config_path, report_path)

    assert not (tmp_path / "data/splits/v2").exists()
    assert not list(tmp_path.rglob("dataset_metadata.json"))


def test_raw_integrity_gate_failure_does_not_create_success_metadata(tmp_path):
    """Keep source integrity failures separate and do not publish complete metadata."""
    _, config_path, report_path, report = _make_fixture(tmp_path)
    report["criteria"] = {"image_inventory": {"status": "FAIL"}}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RawBaselineMismatch, match="raw-integrity criteria"):
        build_clean_split(config_path, report_path)

    assert not (tmp_path / "data/splits/v2").exists()
    assert not list(tmp_path.rglob("dataset_metadata.json"))


def test_existing_different_config_output_is_not_overwritten(tmp_path):
    """Reject an immutable existing split when its config fingerprint differs."""
    _, config_path, report_path, _ = _make_fixture(tmp_path)
    build_clean_split(config_path, report_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["seed"] += 1
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    manifest_path = tmp_path / "data/splits/v2/dataset_manifest.csv"
    previous = manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="different fingerprint/config"):
        build_clean_split(config_path, report_path)

    assert manifest_path.read_bytes() == previous


def test_matching_existing_output_is_skipped_idempotently(tmp_path):
    """Validate an identical published output and return SKIPPED without rewriting it."""
    _, config_path, report_path, _ = _make_fixture(tmp_path)
    first = build_clean_split(config_path, report_path)
    manifest_path = tmp_path / "data/splits/v2/dataset_manifest.csv"
    previous = manifest_path.read_bytes()

    second = build_clean_split(config_path, report_path)

    assert second["state"] == "SKIPPED"
    assert second["split_fingerprint"] == first["split_fingerprint"]
    assert manifest_path.read_bytes() == previous


def test_preflight_does_not_create_output_or_report_directories(tmp_path):
    """Keep read-only preflight free of split output, lock and report directory writes."""
    _, config_path, report_path, _ = _make_fixture(tmp_path)

    result = build_clean_split(config_path, report_path, publish=False)

    assert result["clean_split_status"] == "PASS"
    assert not (tmp_path / "data/splits/v2").exists()
    assert not (tmp_path / "data/reports/v2").exists()
    assert not list(tmp_path.rglob("*.lock"))


def test_publish_failure_keeps_target_absent_and_cleans_staging(tmp_path, monkeypatch):
    """Clean staging after a simulated failure at atomic directory publication."""
    _, config_path, report_path, _ = _make_fixture(tmp_path)
    original_rename = Path.rename

    def fail_split_publish(path, target):
        """Raise only for the v2 staging rename and preserve every other rename."""
        if ".v2.staging-" in path.name:
            raise OSError("simulated publish interruption")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_split_publish)
    with pytest.raises(OSError, match="simulated publish interruption"):
        build_clean_split(config_path, report_path)

    assert not (tmp_path / "data/splits/v2").exists()
    assert not list((tmp_path / "data/splits").glob(".v2.staging-*"))


def test_two_processes_publish_one_identical_split_safely(tmp_path):
    """Serialize competing split publishers and validate a single immutable result."""
    _, config_path, report_path, _ = _make_fixture(tmp_path)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_build_split_in_process,
            args=(config_path, report_path, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(40)
        if process.is_alive():
            process.terminate()
            process.join()

    states = sorted(result_queue.get(timeout=2) for _ in processes)
    assert states == ["PASS", "SKIPPED"]
    assert all(process.exitcode == 0 for process in processes)
    assert (tmp_path / "data/splits/v2/dataset_metadata.json").is_file()


def test_manifest_dataset_keeps_legacy_v1_absolute_root_semantics(tmp_path):
    """Continue reading the frozen v1 schema with its existing absolute root field."""
    raw = tmp_path / "raw"
    image_path = raw / "images/pizza/image.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (12, 34, 56)).save(image_path, format="JPEG")
    manifest_dir = tmp_path / "v1"
    manifest_dir.mkdir()
    version = "poc-v1"
    metadata = {
        "dataset_version": version,
        "root": str(raw.resolve()),
    }
    labels = {
        "schema_version": "1.0",
        "dataset_version": version,
        "labels": [{"class_id": 0, "class_key": "pizza"}],
    }
    checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
    row = {
        "image_id": "pizza/image",
        "relative_path": "images/pizza/image.jpg",
        "source_dataset": "food101",
        "class_id": 0,
        "class_key": "pizza",
        "split": "train",
        "checksum": checksum,
        "validation_status": "valid",
        "dataset_version": version,
        "capture_group": "",
    }
    (manifest_dir / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (manifest_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    with (manifest_dir / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    dataset = ManifestDataset(manifest_dir, "train")

    assert len(dataset) == 1
    assert dataset.root == raw.resolve()
    assert dataset.rows[0]["dataset_version"] == version
