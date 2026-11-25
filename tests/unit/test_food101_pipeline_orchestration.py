"""Synthetic tests for the resumable Food-101 orchestration DAG."""

import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml
from PIL import Image

from food_assistant.data_pipeline.atomic import resource_lock
from food_assistant.data_pipeline.pipeline import run_pipeline
from food_assistant.ml import food101_verifier
from food_assistant.ml.datasets import FIELDS


def _config(tmp_path: Path, *, download_policy: str = "disabled") -> Path:
    """Create a minimal valid side-effect-free POC config under a temporary workspace."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 1,
        "paths": {
            "raw_root": "data/raw/food-101",
            "staging_root": "data/staging/food101",
            "archive": "data/raw/food-101.tar.gz",
            "manifest_root": "data/splits/poc-v2",
            "report_root": "data/reports/elt/poc_v2",
        },
        "approval_file": "configs/approval.json",
        "class_selection_file": "configs/classes.yaml",
        "dataset_scope": "poc",
        "split_version": "poc-v2",
        "seed": 42,
        "val_fraction": 0.2,
        "download": {
            "policy": download_policy,
            "retries": 1,
            "backoff_seconds": 0,
        },
        "quality": {"require_full_integrity": True, "reject_cross_split_duplicates": True},
        "locks": {"timeout_seconds": 0.1},
        "fixtures": {"enabled": False, "root": "tests/fixtures"},
    }
    path = config_dir / "elt.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _handlers(events: list[str], *, quality: str = "PASS", fail_transform: bool = False):
    """Build deterministic stage doubles that expose orchestration order."""
    handlers = {}
    for name in ("inspect", "extract", "load", "quality", "transform", "validate"):

        def execute(*, config_path, config, report_dir, current=name):
            events.append(current)
            if current == "inspect" and (config.raw_root / "images").is_dir():
                return {
                    "status": "PASS",
                    "raw_verified": True,
                    "source_fingerprint": "synthetic-source",
                }
            if current == "quality":
                return {"status": quality, "reason": f"synthetic {quality}"}
            if current == "transform" and fail_transform:
                raise RuntimeError("synthetic interruption")
            return {
                "status": "SKIPPED" if current in {"extract", "load"} else "PASS",
                "skip_reason": "verified synthetic source"
                if current in {"extract", "load"}
                else None,
                "artifact_fingerprints": {"fixture": "abc"},
            }

        handlers[name] = execute
    return handlers


def test_synthetic_end_to_end_runs_dag_in_order_and_writes_reports(tmp_path):
    """Exercise a synthetic E-L-T path without network or Food-101 files."""
    config = _config(tmp_path)
    events = []
    result = run_pipeline(
        config,
        stage="all",
        run_id="synthetic-e2e",
        workspace=tmp_path,
        stage_handlers=_handlers(events),
    )
    assert events == ["inspect", "extract", "load", "quality", "transform", "validate"]
    assert result["status"] == "PASS"
    assert (tmp_path / "artifacts/reports/elt/synthetic-e2e/run_report.json").is_file()
    assert (tmp_path / "artifacts/reports/elt/synthetic-e2e/run_report.md").is_file()


def test_supported_poc_config_runs_real_quality_transform_validate_and_resume(
    tmp_path, monkeypatch, capsys
):
    """Run real ELT quality, POC Transform, ManifestDataset validation, and idempotent resume."""
    config_path = _config(tmp_path)
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_value["paths"]["manifest_root"] = "data/splits/synthetic-poc-v2"
    config_value["paths"]["report_root"] = "data/reports/elt/synthetic-poc-v2"
    config_value["class_selection_file"] = "configs/classes_poc.yaml"
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    (tmp_path / "configs/classes_poc.yaml").write_text("classes:\n  - pizza\n", encoding="utf-8")

    raw_root = tmp_path / config_value["paths"]["raw_root"]
    image_specs = {
        "train": [("pizza/train_a", (210, 20, 30)), ("pizza/train_b", (20, 210, 30))],
        "test": [("pizza/test_a", (20, 30, 210))],
    }
    image_results = {}
    for split_name, values in image_specs.items():
        (raw_root / "meta" / f"{split_name}.txt").parent.mkdir(parents=True, exist_ok=True)
        (raw_root / "meta" / f"{split_name}.txt").write_text(
            "\n".join(image_id for image_id, _ in values) + "\n", encoding="utf-8"
        )
        for image_id, color in values:
            image_path = raw_root / "images" / f"{image_id}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 12), color).save(image_path, format="JPEG")
            relative = image_path.relative_to(raw_root).as_posix()
            image_results[relative] = {"result": food101_verifier._inspect_image(image_path)}
    (raw_root / "meta" / "classes.txt").write_text("pizza\n", encoding="utf-8")
    (raw_root / "meta" / "train.json").write_text(
        json.dumps({"pizza": ["pizza/train_a", "pizza/train_b"]}), encoding="utf-8"
    )
    (raw_root / "meta" / "test.json").write_text(
        json.dumps({"pizza": ["pizza/test_a"]}), encoding="utf-8"
    )
    (raw_root / "meta" / "labels.txt").write_text("Pizza\n", encoding="utf-8")
    (raw_root / "license_agreement.txt").write_text("synthetic license\n", encoding="utf-8")
    monkeypatch.setattr(food101_verifier, "EXPECTED_CLASSES", 1)
    monkeypatch.setattr(food101_verifier, "EXPECTED_IMAGES", 3)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TRAIN_PER_CLASS", 2)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TEST_PER_CLASS", 1)
    (raw_root / "download_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "food101",
                "image_count": 3,
                "download_sha256": "a" * 64,
                "source_url": "https://example.invalid/food101",
                "archive_url": "https://example.invalid/food101.tar.gz",
                "inventory_sha256": food101_verifier._content_inventory_hash(image_results),
                "integrity_status": "PASS",
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    phase25_report_dir = tmp_path / "data/reports/elt/v2/phase_2_5_verification_retry"
    baseline = food101_verifier.audit_food101(
        raw_root,
        phase25_report_dir,
        reject_cross_split_duplicates=False,
    )
    assert baseline["overall"] == "PASS"
    from scripts.run_food101_elt import main as cli_main

    monkeypatch.chdir(tmp_path)

    def invoke_cli(*arguments):
        """Invoke the real CLI entry point and parse its JSON result."""
        monkeypatch.setattr(sys, "argv", ["run_food101_elt.py", *arguments])
        exit_code = cli_main()
        output = capsys.readouterr().out
        return exit_code, json.loads(output)

    before_dry_run = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    dry_exit, dry_run = invoke_cli("--config", "configs/elt.yaml", "--stage", "all", "--dry-run")
    after_dry_run = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert dry_exit == 0
    assert dry_run["status"] == "DRY_RUN"
    assert dry_run["plan"]["transform"]["status"] == "RUN"
    assert before_dry_run == after_dry_run

    first_exit, first = invoke_cli(
        "--config",
        "configs/elt.yaml",
        "--stage",
        "all",
        "--run-id",
        "real-synthetic-poc",
    )
    assert first_exit == 0
    assert first["status"] == "PASS"
    assert first["stages"]["quality"]["raw_integrity_status"] == "PASS"
    assert first["stages"]["transform"]["status"] == "PASS"
    assert first["stages"]["validate"]["status"] == "PASS"
    assert first["stages"]["validate"]["split_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    manifest_path = tmp_path / config_value["paths"]["manifest_root"] / "dataset_manifest.csv"
    manifest_before = manifest_path.read_bytes()

    second_exit, second = invoke_cli(
        "--config",
        "configs/elt.yaml",
        "--stage",
        "all",
        "--resume",
        "--run-id",
        "real-synthetic-poc",
    )
    assert second_exit == 0
    assert second["status"] == "PASS"
    assert second["stages"]["transform"]["status"] == "SKIPPED"
    assert (
        second["stages"]["transform"]["split_fingerprint"]
        == first["stages"]["transform"]["split_fingerprint"]
    )
    assert manifest_path.read_bytes() == manifest_before

    original_config = config_path.read_bytes()
    config_value["seed"] = 43
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    conflict_exit, conflict = invoke_cli(
        "--config",
        "configs/elt.yaml",
        "--stage",
        "all",
        "--run-id",
        "changed-config-conflict",
    )
    assert conflict_exit == 1
    assert conflict["status"] == "FAIL"
    assert conflict["stages"]["transform"]["error_type"] == "FileExistsError"
    assert conflict["stages"]["validate"]["status"] == "BLOCKED"
    assert manifest_path.read_bytes() == manifest_before
    config_path.write_bytes(original_config)

    published_quality_report = manifest_path.parent / "quality_report.json"
    published_quality_report.write_text("{broken", encoding="utf-8")
    corrupt_report = published_quality_report.read_bytes()
    resumed_exit, resumed_corrupt = invoke_cli(
        "--config",
        "configs/elt.yaml",
        "--stage",
        "all",
        "--resume",
        "--run-id",
        "real-synthetic-poc",
    )
    assert resumed_exit == 1
    assert resumed_corrupt["status"] == "FAIL"
    assert resumed_corrupt["stages"]["transform"]["error_type"] == "FileExistsError"
    assert published_quality_report.read_bytes() == corrupt_report
    assert manifest_path.read_bytes() == manifest_before


def test_resume_recovers_after_transform_publish_before_run_state_update(
    tmp_path, monkeypatch
):
    """Recover a real POC Transform published immediately before a hard process stop."""
    config_path = _config(tmp_path)
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_value["paths"]["manifest_root"] = "data/splits/synthetic-crash-v2"
    config_value["paths"]["report_root"] = "data/reports/elt/synthetic-crash-v2"
    config_value["class_selection_file"] = "configs/classes_poc.yaml"
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    (tmp_path / "configs/classes_poc.yaml").write_text("classes:\n  - pizza\n", encoding="utf-8")

    raw_root = tmp_path / config_value["paths"]["raw_root"]
    image_records = {}
    split_ids = {"train": ("train_a", "train_b"), "test": ("test_a",)}
    colors = ((210, 30, 20), (20, 210, 30), (30, 20, 210))
    image_index = 0
    for split_name, image_names in split_ids.items():
        metadata_path = raw_root / "meta" / f"{split_name}.txt"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            "".join(f"pizza/{image_name}\n" for image_name in image_names),
            encoding="utf-8",
        )
        for image_name in image_names:
            image_path = raw_root / "images" / "pizza" / f"{image_name}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 12), colors[image_index]).save(image_path, format="JPEG")
            relative_path = image_path.relative_to(raw_root).as_posix()
            image_records[relative_path] = {
                "result": food101_verifier._inspect_image(image_path)
            }
            image_index += 1
    metadata_dir = raw_root / "meta"
    (metadata_dir / "classes.txt").write_text("pizza\n", encoding="utf-8")
    (metadata_dir / "train.json").write_text(
        json.dumps({"pizza": ["pizza/train_a", "pizza/train_b"]}), encoding="utf-8"
    )
    (metadata_dir / "test.json").write_text(
        json.dumps({"pizza": ["pizza/test_a"]}), encoding="utf-8"
    )
    (metadata_dir / "labels.txt").write_text("Pizza\n", encoding="utf-8")
    (raw_root / "license_agreement.txt").write_text("synthetic license\n", encoding="utf-8")
    (raw_root / "download_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "food101",
                "image_count": len(image_records),
                "download_sha256": "a" * 64,
                "source_url": "https://example.invalid/food101",
                "archive_url": "https://example.invalid/food101.tar.gz",
                "inventory_sha256": food101_verifier._content_inventory_hash(image_records),
                "integrity_status": "PASS",
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(food101_verifier, "EXPECTED_CLASSES", 1)
    monkeypatch.setattr(food101_verifier, "EXPECTED_IMAGES", 3)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TRAIN_PER_CLASS", 2)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TEST_PER_CLASS", 1)
    baseline_path = tmp_path / "data/reports/elt/v2/phase_2_5_verification_retry"
    baseline = food101_verifier.audit_food101(
        raw_root,
        baseline_path,
        reject_cross_split_duplicates=False,
    )
    assert baseline["overall"] == "PASS"

    child_code = r'''
import json
import sys
from pathlib import Path
from food_assistant.data_pipeline import pipeline
from food_assistant.ml import food101_verifier

food101_verifier.EXPECTED_CLASSES = 1
food101_verifier.EXPECTED_IMAGES = 3
food101_verifier.EXPECTED_TRAIN_PER_CLASS = 2
food101_verifier.EXPECTED_TEST_PER_CLASS = 1
original_write_json = pipeline.atomic_write_json

def pause_after_transform_publish(path, value):
    """Stop before persisting the post-publication Transform stage state."""
    state = value if isinstance(value, dict) else {}
    transform = state.get("stages", {}).get("transform", {})
    if Path(path).name == "run_state.json" and transform.get("status") == "PASS":
        print("TRANSFORM_PUBLISHED_STATE_PENDING", flush=True)
        sys.stdin.readline()
    original_write_json(path, value)

pipeline.atomic_write_json = pause_after_transform_publish
result = pipeline.run_pipeline(
    Path(sys.argv[1]),
    stage="all",
    run_id=sys.argv[2],
    workspace=Path(sys.argv[3]),
)
print("RUN_COMPLETED " + json.dumps(result), flush=True)
'''
    repository_root = Path(__file__).resolve().parents[2]
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(repository_root),
            str(repository_root / "src"),
            child_environment.get("PYTHONPATH", ""),
        )
        if part
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            str(config_path),
            "post-publish-crash",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=child_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    signal_queue = queue.Queue()

    def read_child_signal():
        """Read the process synchronization signal without blocking pytest forever."""
        signal_queue.put(child.stdout.readline())

    signal_reader = threading.Thread(target=read_child_signal, daemon=True)
    signal_reader.start()
    try:
        try:
            signal = signal_queue.get(timeout=60)
        except queue.Empty:
            pytest.fail("Child did not reach the post-publication synchronization point")
        if signal.strip() != "TRANSFORM_PUBLISHED_STATE_PENDING":
            child.wait(timeout=15)
            child_error = child.stderr.read()
            pytest.fail(
                "Child exited before publish synchronization: "
                f"stdout={signal!r}; stderr={child_error}"
            )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=15)
        signal_reader.join(timeout=5)
    child.stderr.read()
    assert child.returncode != 0

    state_path = tmp_path / "artifacts/reports/elt/post-publish-crash/run_state.json"
    stale_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert stale_state["stages"]["transform"]["status"] == "RUNNING"
    output_path = tmp_path / config_value["paths"]["manifest_root"]
    assert (output_path / "dataset_manifest.csv").is_file()
    published_before = {
        path.relative_to(output_path).as_posix(): path.read_bytes()
        for path in output_path.rglob("*")
        if path.is_file()
    }

    monkeypatch.setattr(food101_verifier, "EXPECTED_CLASSES", 1)
    monkeypatch.setattr(food101_verifier, "EXPECTED_IMAGES", 3)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TRAIN_PER_CLASS", 2)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TEST_PER_CLASS", 1)
    recovered = run_pipeline(
        config_path,
        stage="all",
        resume=True,
        run_id="post-publish-crash",
        workspace=tmp_path,
    )

    assert recovered["status"] == "PASS"
    assert recovered["stages"]["transform"]["status"] == "SKIPPED"
    assert recovered["stages"]["transform"]["attempts"] == 2
    assert recovered["stages"]["validate"]["status"] == "PASS"
    published_after = {
        path.relative_to(output_path).as_posix(): path.read_bytes()
        for path in output_path.rglob("*")
        if path.is_file()
    }
    assert published_after == published_before


def test_validate_stage_loads_real_manifestdataset_contract(tmp_path):
    """Validate a tiny real manifest with ManifestDataset, not a mocked reader."""
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_root = tmp_path / config["paths"]["raw_root"]
    manifest_root = tmp_path / config["paths"]["manifest_root"]
    manifest_root.mkdir(parents=True)
    rows = []
    for index, split_name in enumerate(("train", "val", "test")):
        relative_path = f"images/class_a/sample-{index}.jpg"
        image_path = raw_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color=(index * 50, 20, 30)).save(image_path)
        rows.append(
            {
                "image_id": f"class_a/sample-{index}",
                "relative_path": relative_path,
                "source_dataset": "food101",
                "class_id": 0,
                "class_key": "class_a",
                "split": split_name,
                "checksum": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "validation_status": "valid",
                "dataset_version": "poc-v2",
                "capture_group": "",
            }
        )
    with (manifest_root / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (manifest_root / "dataset_metadata.json").write_text(
        json.dumps({"dataset_version": "poc-v2", "root": str(raw_root)}),
        encoding="utf-8",
    )
    (manifest_root / "labels.json").write_text(
        json.dumps(
            {"dataset_version": "poc-v2", "labels": [{"class_id": 0, "class_key": "class_a"}]}
        ),
        encoding="utf-8",
    )
    handlers = _handlers([])
    handlers.pop("validate")
    result = run_pipeline(
        config_path,
        stage="validate",
        run_id="manifest-reader",
        workspace=tmp_path,
        stage_handlers=handlers,
    )
    assert result["status"] == "PASS"
    assert result["stages"]["validate"]["split_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }


def test_existing_raw_skips_extract_and_load_without_network(tmp_path):
    """Skip acquisition when raw directory is present while retaining quality as a prerequisite."""
    config = _config(tmp_path)
    (tmp_path / "data/raw/food-101/images").mkdir(parents=True)
    (tmp_path / "data/raw/food-101/meta").mkdir(parents=True)
    plan = run_pipeline(config, stage="all", dry_run=True, workspace=tmp_path)
    assert plan["plan"]["extract"]["status"] == "RUN"
    assert plan["plan"]["load"]["status"] == "RUN"
    events = []
    handlers = _handlers(events)
    handlers.pop("extract")
    handlers.pop("load")
    result = run_pipeline(
        config,
        stage="all",
        run_id="raw-present",
        workspace=tmp_path,
        stage_handlers=handlers,
    )
    assert result["stages"]["extract"]["status"] == "SKIPPED"
    assert result["stages"]["load"]["status"] == "SKIPPED"


def test_missing_raw_with_download_disabled_blocks_offline(tmp_path):
    """Refuse missing raw in offline mode without invoking network code."""
    config = _config(tmp_path)
    result = run_pipeline(config, stage="all", run_id="missing-raw", workspace=tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["stages"]["extract"]["status"] == "BLOCKED"
    assert result["stages"]["load"]["status"] == "BLOCKED"
    assert result["stages"]["transform"]["status"] == "BLOCKED"


def test_dry_run_does_not_mutate_filesystem(tmp_path):
    """Ensure dry-run does not create logs, reports, lock files, or staging output."""
    config = _config(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    result = run_pipeline(config, stage="all", dry_run=True, workspace=tmp_path)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert result["status"] == "DRY_RUN"
    assert before == after


def test_dry_run_marks_unsupported_transform_config_blocked(tmp_path):
    """Expose a stale scope/version pair in the plan before any state is written."""
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["split_version"] = "v2"
    config.write_text(yaml.safe_dump(value), encoding="utf-8")
    (tmp_path / "data/raw/food-101/images").mkdir(parents=True)
    (tmp_path / "data/raw/food-101/meta").mkdir(parents=True)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    plan = run_pipeline(config, stage="all", dry_run=True, workspace=tmp_path)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert plan["plan"]["transform"]["status"] == "BLOCKED"
    assert before == after


@pytest.mark.parametrize("quality", ["FAIL", "INCOMPLETE"])
def test_quality_failure_blocks_transform_and_downstream(tmp_path, quality):
    """Verify quality failures prevent Transform and Validate handlers from running."""
    config = _config(tmp_path)
    events = []
    result = run_pipeline(
        config,
        stage="all",
        run_id=f"quality-{quality.lower()}",
        workspace=tmp_path,
        stage_handlers=_handlers(events, quality=quality),
    )
    assert "transform" not in events
    assert "validate" not in events
    assert result["status"] == quality
    assert result["stages"]["transform"]["status"] == "BLOCKED"


def test_unknown_quality_status_fails_closed(tmp_path):
    """Prevent an unrecognized quality result from authorizing Transform."""
    config = _config(tmp_path)
    events = []
    result = run_pipeline(
        config,
        stage="all",
        run_id="quality-unknown",
        workspace=tmp_path,
        stage_handlers=_handlers(events, quality="WARN"),
    )
    assert result["status"] == "INCOMPLETE"
    assert "transform" not in events


def test_resume_retries_interrupted_transform_and_preserves_stage_history(tmp_path):
    """Rerun the interrupted stage after validating the prior state and config hash."""
    config = _config(tmp_path)
    events = []
    first = run_pipeline(
        config,
        stage="transform",
        run_id="resume-case",
        workspace=tmp_path,
        stage_handlers=_handlers(events, fail_transform=True),
    )
    assert first["status"] == "FAIL"
    events.clear()
    second = run_pipeline(
        config,
        stage="transform",
        resume=True,
        run_id="resume-case",
        workspace=tmp_path,
        stage_handlers=_handlers(events),
    )
    assert second["status"] == "PASS"
    assert second["stages"]["transform"]["attempts"] == 2
    assert "transform" in events


def test_corrupt_resume_state_is_rejected(tmp_path):
    """Fail closed rather than trusting malformed checkpoint JSON."""
    config = _config(tmp_path)
    run_dir = tmp_path / "artifacts/reports/elt/corrupt-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_state.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        run_pipeline(config, stage="all", resume=True, run_id="corrupt-run", workspace=tmp_path)


def test_resume_discovers_latest_matching_run_without_mutating_other_runs(tmp_path):
    """Support the documented --resume form while selecting only matching config state."""
    config = _config(tmp_path)
    handlers = _handlers([])
    initial = run_pipeline(
        config,
        stage="inspect",
        run_id="auto-resume",
        workspace=tmp_path,
        stage_handlers=handlers,
    )
    resumed = run_pipeline(
        config,
        stage="inspect",
        resume=True,
        workspace=tmp_path,
        stage_handlers=_handlers([]),
    )
    assert resumed["run_id"] == initial["run_id"]


def test_runner_lock_conflict_is_reported(tmp_path):
    """Reject a concurrent owner of the same run-state resource."""
    config = _config(tmp_path)
    run_dir = tmp_path / "artifacts/reports/elt/locked-run"
    run_dir.mkdir(parents=True)
    resource = run_dir / "run_state.json"
    with resource_lock(resource, timeout_seconds=0):
        with pytest.raises(RuntimeError, match="locked"):
            run_pipeline(
                config,
                stage="all",
                run_id="locked-run",
                workspace=tmp_path,
                stage_handlers=_handlers([]),
            )


def test_report_state_is_valid_json_and_no_sensitive_config_values(tmp_path):
    """Persist versioned stage evidence without copying arbitrary config secrets."""
    config = _config(tmp_path)
    result = run_pipeline(
        config,
        stage="inspect",
        run_id="state-schema",
        workspace=tmp_path,
        stage_handlers=_handlers([]),
    )
    state = json.loads(
        (tmp_path / "artifacts/reports/elt/state-schema/run_state.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "PASS"
    assert state["schema_version"] == 1
    assert state["stages"]["inspect"]["attempts"] == 1
    assert state["stages"]["inspect"]["duration_seconds"] >= 0
