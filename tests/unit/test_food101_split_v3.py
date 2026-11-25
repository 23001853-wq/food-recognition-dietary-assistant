"""Synthetic verification for Food-101 v3 cleaning, validation and publishing."""

import csv
import hashlib
import json
import multiprocessing
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml
from PIL import Image

from food_assistant.data_pipeline.atomic import checkpoint_digest
from food_assistant.data_pipeline.split_v3 import (
    _clean_train_rows,
    _stratified_assignment,
    build_training_split_v3,
)
from food_assistant.data_pipeline.splits import build_clean_split
from food_assistant.ml.datasets import ManifestDataset


def _write_image(path, color):
    """Write a deterministic synthetic JPEG at the requested relative path."""
    target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(target, format="JPEG")


def _make_v3_fixture(workspace):
    """Create full-contract-shaped raw/v2 evidence with duplicate and conflict groups."""
    raw = workspace / "data/raw/food-101"
    metadata_root = raw / "meta"
    metadata_root.mkdir(parents=True)
    classes = ["alpha", "beta"]
    train_ids = [
        *[f"alpha/a{index:02d}" for index in range(12)],
        "alpha/a_same_1",
        "alpha/a_same_2",
        "alpha/a_same_3",
        "alpha/a_conflict",
        *[f"beta/b{index:02d}" for index in range(12)],
        "beta/b_multi_1",
        "beta/b_multi_2",
        "beta/b_multi_3",
        "beta/b_conflict",
    ]
    test_ids = [f"{class_key}/test{index}" for class_key in classes for index in range(4)]
    (metadata_root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (metadata_root / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (metadata_root / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    colors = {}
    for index, image_id in enumerate(train_ids):
        colors[image_id] = (
            (index * 37 + 11) % 255,
            (index * 53 + 23) % 255,
            (index * 71 + 41) % 255,
        )
    for index, image_id in enumerate(test_ids):
        colors[image_id] = (
            (index * 43 + 17) % 255,
            (index * 61 + 29) % 255,
            (index * 79 + 47) % 255,
        )
    colors.update(
        {
            "alpha/a_same_1": (14, 44, 74),
            "alpha/a_same_2": (14, 44, 74),
            "alpha/a_same_3": (14, 44, 74),
            "beta/b_multi_1": (15, 45, 75),
            "beta/b_multi_2": (15, 45, 75),
            "beta/b_multi_3": (15, 45, 75),
            "alpha/a_conflict": (16, 46, 76),
            "beta/b_conflict": (16, 46, 76),
        }
    )
    for image_id, color in colors.items():
        _write_image(raw / "images" / f"{image_id}.jpg", color)
    marker = {"dataset": "food101", "download_sha256": "b" * 64}
    (raw / "download_metadata.json").write_text(json.dumps(marker), encoding="utf-8")
    paths = sorted(path.relative_to(raw).as_posix() for path in (raw / "images").rglob("*.jpg"))
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
    source_fingerprint = checkpoint_digest(
        {
            "inventory_sha256": inventory.hexdigest(),
            "metadata_sha256": metadata_hashes,
            "source_sha256": marker["download_sha256"],
        }
    )
    signature_paths = sorted(paths + metadata_paths + ["download_metadata.json"])
    signatures = {}
    files = {}
    for relative in signature_paths:
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
    baseline = {
        "mode": "full",
        "filesystem_paths": signature_paths,
        "filesystem_fingerprint": filesystem_fingerprint,
        "actual_image_count": len(paths),
        "class_count": len(classes),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "metadata_sha256": metadata_hashes,
        "inventory_sha256": inventory.hexdigest(),
        "source_fingerprint": source_fingerprint,
        "criteria": {},
    }
    baseline_path = workspace / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    checkpoint_payload = {
        "schema_version": 2,
        "dataset_root": str(raw.resolve()),
        "input_hash": filesystem_fingerprint,
        "expected_files": paths,
        "files": files,
        "complete": True,
    }
    (workspace / "food101_checkpoint.json").write_text(
        json.dumps(
            {"payload": checkpoint_payload, "sha256": checkpoint_digest(checkpoint_payload)}
        ),
        encoding="utf-8",
    )
    classes_path = workspace / "classes_full.yaml"
    classes_path.write_text(
        yaml.safe_dump({"version": "fixture", "classes": classes}), encoding="utf-8"
    )
    v2_config_path = workspace / "food101_elt_full_v2.yaml"
    v2_config = _config_document(
        workspace,
        raw,
        classes_path,
        version="v2",
        output="data/splits/v2",
        report="data/reports/v2",
    )
    v2_config["val_fraction"] = 0.2
    v2_config["split_policy"] = {
        "preserve_official_test": True,
        "exclude_train_hashes_in_test": True,
        "create_validation": False,
        "reject_cross_partition_duplicates": True,
    }
    v2_config_path.write_text(yaml.safe_dump(v2_config), encoding="utf-8")
    build_clean_split(v2_config_path, baseline_path)
    v3_config_path = workspace / "food101_elt_full_v3.yaml"
    v3_config = _config_document(
        workspace,
        raw,
        classes_path,
        version="v3",
        output="data/splits/v3",
        report="data/reports/v3",
    )
    v3_config["source_manifest_root"] = str(workspace / "data/splits/v2")
    v3_config["seed"] = 42
    v3_config["val_fraction"] = 0.1
    v3_config["split_policy"] = {
        "preserve_official_test": True,
        "exclude_train_hashes_in_test": False,
        "create_validation": True,
        "reject_cross_partition_duplicates": True,
        "deduplicate_train": True,
        "same_label_duplicate_policy": "keep_lexicographic_min",
        "cross_label_duplicate_policy": "quarantine_all",
        "validation_algorithm": "stratified_after_sha256_group_collapse",
    }
    v3_config_path.write_text(yaml.safe_dump(v3_config), encoding="utf-8")
    return raw, v2_config_path, v3_config_path, baseline_path


def _config_document(workspace, raw, classes_path, version, output, report):
    """Build a validated synthetic ELT config for one split version."""
    return {
        "schema_version": 1,
        "paths": {
            "raw_root": str(raw),
            "staging_root": str(workspace / "staging"),
            "archive": str(workspace / "archive.tar.gz"),
            "manifest_root": str(workspace / output),
            "report_root": str(workspace / report),
        },
        "approval_file": str(workspace / "approval.json"),
        "class_selection_file": str(classes_path),
        "dataset_scope": "full",
        "split_version": version,
        "seed": 42,
        "val_fraction": 0.2,
        "download": {"policy": "disabled", "retries": 0, "backoff_seconds": 0},
        "quality": {"require_full_integrity": True, "reject_cross_split_duplicates": True},
        "locks": {"timeout_seconds": 30},
        "fixtures": {"enabled": False, "root": str(workspace / "fixtures")},
    }


def _publish_v3_process(config_path, v2_config_path, baseline_path, results):
    """Publish one synthetic v3 split from a competing process."""
    try:
        outcome = build_training_split_v3(config_path, v2_config_path, baseline_path)
        results.put(outcome["state"])
    except BaseException as exc:
        results.put(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def test_v3_deduplicates_same_label_and_quarantines_all_cross_label_refs(tmp_path):
    """Keep deterministic representatives, remove every conflict reference and record provenance."""
    raw, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    result = build_training_split_v3(v3_config, v2_config, baseline)

    assert result["same_label_duplicate_group_count"] == 2
    assert result["cross_label_duplicate_group_count"] == 1
    assert len(result["excluded_images"]) == 6
    excluded = {entry["image_id"]: entry for entry in result["excluded_images"]}
    assert excluded["alpha/a_same_2"]["representative_image_id"] == "alpha/a_same_1"
    assert excluded["alpha/a_same_3"]["representative_image_id"] == "alpha/a_same_1"
    assert excluded["beta/b_multi_2"]["representative_image_id"] == "beta/b_multi_1"
    assert excluded["beta/b_multi_3"]["representative_image_id"] == "beta/b_multi_1"
    assert excluded["alpha/a_conflict"]["status"] == "LABEL_CONFLICT_QUARANTINED"
    assert excluded["beta/b_conflict"]["status"] == "LABEL_CONFLICT_QUARANTINED"
    output = tmp_path / "data/splits/v3"
    reader = ManifestDataset(output, "train")
    assert reader.root == raw.resolve()
    assert {row["class_key"] for row in reader.rows} == {"alpha", "beta"}
    assert (raw / "images/alpha/a_same_2.jpg").is_file()
    assert (tmp_path / "data/splits/v2/dataset_manifest.csv").is_file()


def test_v3_validation_is_deterministic_stratified_and_preserves_test(tmp_path):
    """Create repeatable class-stratified validation and retain every v2 test ID."""
    raw, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    result = build_training_split_v3(v3_config, v2_config, baseline)
    output = tmp_path / "data/splits/v3"
    with (output / "dataset_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (tmp_path / "data/splits/v2/test.csv").open(encoding="utf-8", newline="") as stream:
        v2_test = list(csv.DictReader(stream))
    v3_test = [row for row in rows if row["split"] == "test"]
    assert [row["image_id"] for row in v3_test] == [row["image_id"] for row in v2_test]
    assert [(row["class_id"], row["class_key"], row["checksum"]) for row in v3_test] == [
        (row["class_id"], row["class_key"], row["checksum"]) for row in v2_test
    ]
    assert result["split_counts"]["test"] == len(v2_test)
    assert (
        0
        < result["split_counts"]["val"]
        < result["after_dedup_and_quarantine_counts"]["train_candidates"]
    )
    assert set(result["per_class_counts"]["train"]) == {"alpha", "beta"}
    assert set(result["per_class_counts"]["val"]) == {"alpha", "beta"}
    assert result["quality_gates"]["overall"] == "PASS"
    assert result["cross_partition_duplicate_hash_count"] == 0
    assert result["train_duplicate_hash_count"] == 0
    assert raw.is_dir()


def test_v3_hash_partitions_have_no_cross_partition_overlap(tmp_path):
    """Validate every manifest checksum occurs in only one partition."""
    _, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    build_training_split_v3(v3_config, v2_config, baseline)
    rows = list(
        csv.DictReader((tmp_path / "data/splits/v3/dataset_manifest.csv").open(encoding="utf-8"))
    )
    partitions = defaultdict(set)
    train_hashes = Counter()
    ids = []
    for row in rows:
        partitions[row["checksum"]].add(row["split"])
        if row["split"] == "train":
            train_hashes[row["checksum"]] += 1
        ids.append(row["image_id"])
    assert all(len(splits) == 1 for splits in partitions.values())
    assert all(count == 1 for count in train_hashes.values())
    assert len(ids) == len(set(ids))


def test_v3_stratified_assignment_is_seeded_and_group_unique():
    """Use deterministic stratification after collapsing each SHA-256 group."""
    candidates = [
        {
            "image_id": f"{class_key}/{index}",
            "class_key": class_key,
            "checksum": f"{class_key}-{index}",
        }
        for class_key in ("alpha", "beta")
        for index in range(20)
    ]
    first = _stratified_assignment(candidates, 42, 0.1, ["alpha", "beta"])
    repeat = _stratified_assignment(candidates, 42, 0.1, ["alpha", "beta"])
    other_seed = _stratified_assignment(candidates, 43, 0.1, ["alpha", "beta"])
    assert first == repeat
    assert first != other_seed
    assert Counter(first.values()) == {"train": 36, "val": 4}


def test_v3_cross_label_duplicate_group_with_many_refs_quarantines_every_image():
    """Quarantine all members of any multi-image cross-label content group."""
    digest = "f" * 64
    rows = [
        {"image_id": "alpha/one", "class_key": "alpha", "checksum": digest},
        {"image_id": "beta/two", "class_key": "beta", "checksum": digest},
        {"image_id": "gamma/three", "class_key": "gamma", "checksum": digest},
    ]
    candidates, exclusions, conflicts = _clean_train_rows(rows, [])
    assert candidates == []
    assert {row["image_id"] for row in exclusions} == {"alpha/one", "beta/two", "gamma/three"}
    assert conflicts[0]["classes"] == ["alpha", "beta", "gamma"]
    assert conflicts[0]["status"] == "LABEL_CONFLICT_QUARANTINED"


def test_v3_preflight_does_not_write_lock_or_split_output(tmp_path):
    """Keep read-only v3 preflight free of staging, lock and report writes."""
    _, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    previous_v2 = (tmp_path / "data/splits/v2/dataset_manifest.csv").read_bytes()
    result = build_training_split_v3(v3_config, v2_config, baseline, publish=False)
    assert result["quality_gates"]["overall"] == "PASS"
    assert not (tmp_path / "data/splits/v3").exists()
    assert not (tmp_path / "data/reports/v3").exists()
    assert not list((tmp_path / "data/splits").glob(".v3.lock"))
    assert (tmp_path / "data/splits/v2/dataset_manifest.csv").read_bytes() == previous_v2


def test_v3_raw_fingerprint_mismatch_blocks_without_success_metadata(tmp_path):
    """Refuse v3 when raw input no longer matches the signed Phase 2.5 evidence."""
    raw, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    target = raw / "meta/classes.txt"
    target.write_text("alpha\nbeta\nchanged\n", encoding="utf-8")
    with pytest.raises(Exception, match="Phase 2.5 class/image counts do not match"):
        build_training_split_v3(v3_config, v2_config, baseline)
    assert not (tmp_path / "data/splits/v3").exists()
    assert not (tmp_path / "data/splits/v3/dataset_metadata.json").exists()


def test_v3_output_conflict_does_not_overwrite_existing_manifest(tmp_path):
    """Refuse a v3 target whose existing fingerprint differs from current config."""
    _, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    build_training_split_v3(v3_config, v2_config, baseline)
    manifest = tmp_path / "data/splits/v3/dataset_manifest.csv"
    previous = manifest.read_bytes()
    config = yaml.safe_load(v3_config.read_text(encoding="utf-8"))
    config["seed"] = 43
    v3_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicts"):
        build_training_split_v3(v3_config, v2_config, baseline)
    assert manifest.read_bytes() == previous


def test_v3_publish_failure_cleans_stage_and_preserves_v2(tmp_path, monkeypatch):
    """Keep v2 intact and remove only the new staging tree after publish interruption."""
    _, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    v2_manifest = tmp_path / "data/splits/v2/dataset_manifest.csv"
    previous = v2_manifest.read_bytes()
    original_rename = Path.rename

    def fail_v3_rename(path, target):
        """Simulate interruption at the final v3 directory rename."""
        if ".v3.staging-" in path.name:
            raise OSError("simulated v3 publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_v3_rename)
    with pytest.raises(OSError, match="simulated v3 publish failure"):
        build_training_split_v3(v3_config, v2_config, baseline)
    assert not (tmp_path / "data/splits/v3").exists()
    assert not list((tmp_path / "data/splits").glob(".v3.staging-*"))
    assert v2_manifest.read_bytes() == previous
    monkeypatch.undo()
    recovered = build_training_split_v3(v3_config, v2_config, baseline)
    assert recovered["state"] == "PASS"
    assert not list((tmp_path / "data/splits").glob(".v3.staging-*"))


def test_v3_two_process_publish_is_serialized(tmp_path):
    """Serialize concurrent v3 publishers and preserve one valid immutable result."""
    _, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_publish_v3_process,
            args=(v3_config, v2_config, baseline, output),
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
    assert sorted(output.get(timeout=2) for _ in processes) == ["PASS", "SKIPPED"]
    assert all(process.exitcode == 0 for process in processes)
    assert (tmp_path / "data/splits/v3/dataset_metadata.json").is_file()


def test_v3_two_process_different_config_conflicts_without_overwrite(tmp_path):
    """Serialize distinct v3 identities and preserve the first published manifest bytes."""
    _, v2_config, v3_config, baseline = _make_v3_fixture(tmp_path)
    conflicting_config = tmp_path / "food101_elt_full_v3_conflict.yaml"
    config_value = yaml.safe_load(v3_config.read_text(encoding="utf-8"))
    config_value["seed"] = 43
    conflicting_config.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_publish_v3_process,
            args=(config_path, v2_config, baseline, results),
        )
        for config_path in (v3_config, conflicting_config)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(40)
        if process.is_alive():
            process.terminate()
            process.join()
    outcomes = [results.get(timeout=2) for _ in processes]
    assert sum(result == "PASS" for result in outcomes) == 1
    assert sum(result.startswith("FileExistsError:") for result in outcomes) == 1
    assert all(process.exitcode == 0 for process in processes)
    published_manifest = tmp_path / "data/splits/v3/dataset_manifest.csv"
    published_before = published_manifest.read_bytes()
    metadata = json.loads(
        (tmp_path / "data/splits/v3/dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["complete"] is True
    ManifestDataset(tmp_path / "data/splits/v3", "train")
    assert published_manifest.read_bytes() == published_before
