"""Focused unit tests for resumable Food-101 auditing and official label mapping."""

import json
import multiprocessing

import pytest
from PIL import Image

from food_assistant.ml import food101_verifier
from food_assistant.ml.datasets import OfficialFood101Dataset


def _configure_small_contract(monkeypatch):
    """Use an official-shaped two-image fixture as a tiny contract in unit tests."""
    monkeypatch.setattr(food101_verifier, "EXPECTED_CLASSES", 1)
    monkeypatch.setattr(food101_verifier, "EXPECTED_IMAGES", 2)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TRAIN_PER_CLASS", 1)
    monkeypatch.setattr(food101_verifier, "EXPECTED_TEST_PER_CLASS", 1)


def _refresh_fixture_marker(root):
    """Refresh the temporary fixture's Phase 2 marker after deliberate image edits."""
    marker_path = root / "download_metadata.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    rows = {
        relative: {"result": food101_verifier._inspect_image(root / relative)}
        for relative in ("images/pizza/train_image.jpg", "images/pizza/test_image.jpg")
    }
    marker["inventory_sha256"] = food101_verifier._content_inventory_hash(rows)
    food101_verifier.atomic_write_json(marker_path, marker)


def _publish_metadata_worker(root, report_path, metadata_path, report_root, start, results):
    """Race one independent process for the configured metadata publication lock."""
    from food_assistant.ml import food101_verifier as verifier

    verifier.EXPECTED_CLASSES = 1
    verifier.EXPECTED_IMAGES = 2
    verifier.EXPECTED_TRAIN_PER_CLASS = 1
    verifier.EXPECTED_TEST_PER_CLASS = 1
    start.wait(20)
    try:
        result = verifier.publish_metadata_v2(root, report_path, metadata_path, report_root)
        results.put(("ok", result["state"]))
    except Exception as exc:
        results.put(("error", str(exc)))


def _make_small_food101(root):
    """Create a two-image official-shaped fixture without external dataset files."""
    (root / "images" / "pizza").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "classes.txt").write_text("pizza\n", encoding="utf-8")
    (root / "meta" / "train.txt").write_text("pizza/train_image\n", encoding="utf-8")
    (root / "meta" / "test.txt").write_text("pizza/test_image\n", encoding="utf-8")
    (root / "meta" / "train.json").write_text(
        json.dumps({"pizza": ["pizza/train_image"]}), encoding="utf-8"
    )
    (root / "meta" / "test.json").write_text(
        json.dumps({"pizza": ["pizza/test_image"]}), encoding="utf-8"
    )
    (root / "meta" / "labels.txt").write_text("Pizza\n", encoding="utf-8")
    Image.new("RGB", (12, 8), (255, 0, 0)).save(root / "images" / "pizza" / "train_image.jpg")
    Image.new("RGB", (12, 8), (0, 255, 0)).save(root / "images" / "pizza" / "test_image.jpg")
    (root / "license_agreement.txt").write_text("fixture license", encoding="utf-8")
    image_results = {
        f"images/pizza/{name}_image.jpg": {
            "result": food101_verifier._inspect_image(root / "images" / "pizza" / f"{name}_image.jpg")
        }
        for name in ("train", "test")
    }
    (root / "download_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "food101",
                "image_count": 2,
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


def test_full_audit_rejects_stale_source_fingerprint(tmp_path, monkeypatch):
    root, report_dir = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    calls = []
    original = food101_verifier._inspect_image

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(food101_verifier, "_inspect_image", counted)
    first = food101_verifier.audit_food101(root, report_dir, progress_every=1)
    assert first["overall"] == "PASS"
    assert first["valid_images"] == 2
    assert len(calls) == 2

    second = food101_verifier.audit_food101(root, report_dir, progress_every=1)
    assert second["valid_images"] == 2
    assert len(calls) == 2

    changed = root / "images" / "pizza" / "train_image.jpg"
    Image.new("RGB", (13, 9), (0, 0, 255)).save(changed)
    third = food101_verifier.audit_food101(root, report_dir, progress_every=1)
    assert third["valid_images"] == 2
    assert len(calls) == 4


def test_full_audit_rejects_missing_image_from_configured_class_count(tmp_path, monkeypatch):
    root, report_dir = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    (root / "images" / "pizza" / "test_image.jpg").unlink()

    result = food101_verifier.audit_food101(root, report_dir, progress_every=1)

    assert result["overall"] == "FAIL"
    assert result["criteria"]["image_inventory"]["status"] == "FAIL"
    assert result["criteria"]["image_inventory"]["details"]["actual_jpg"] == 1
    assert result["criteria"]["image_inventory"]["details"]["missing_count"] == 1


def test_corrupt_checkpoint_is_safely_ignored(tmp_path, monkeypatch):
    root, report_dir = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    report_dir.mkdir()
    (report_dir / "food101_checkpoint.json").write_text("{broken", encoding="utf-8")
    result = food101_verifier.audit_food101(root, report_dir)
    assert result["valid_images"] == 2
    checkpoint = json.loads((report_dir / "food101_checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["payload"]["files"]) == 2


def test_checkpoint_with_incomplete_inventory_is_rejected(tmp_path, monkeypatch):
    """Reject incomplete checkpoint payloads and rebuild evidence from fixture images."""
    root, report_dir = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    report_dir.mkdir()
    (report_dir / "food101_checkpoint.json").write_text(
        json.dumps({"payload": {"schema_version": 2, "complete": False}, "sha256": "bad"}),
        encoding="utf-8",
    )

    result = food101_verifier.audit_food101(root, report_dir)

    assert result["overall"] == "PASS"
    stored = json.loads((report_dir / "food101_checkpoint.json").read_text(encoding="utf-8"))
    assert stored["payload"]["complete"] is True
    assert len(stored["payload"]["expected_files"]) == 2


def test_audit_rejects_wrong_class_count(tmp_path, monkeypatch):
    """Fail the full dataset contract when metadata class count is wrong."""
    root, report_dir = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    monkeypatch.setattr(food101_verifier, "EXPECTED_CLASSES", 2)

    result = food101_verifier.audit_food101(root, report_dir)

    assert result["overall"] == "FAIL"
    assert result["criteria"]["class_labels_and_directories"]["status"] == "FAIL"


def test_audit_rejects_corrupted_image(tmp_path, monkeypatch):
    """Report decode failure even when the metadata lists the expected file."""
    root, report_dir = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    (root / "images" / "pizza" / "train_image.jpg").write_bytes(b"broken image")

    result = food101_verifier.audit_food101(root, report_dir)

    assert result["overall"] == "FAIL"
    assert result["bad_image_count"] == 1
    assert result["criteria"]["image_decoding_and_sha256"]["status"] == "FAIL"


def test_cross_split_duplicates_follow_configured_quality_policy(tmp_path, monkeypatch):
    """Fail duplicates under reject policy and report them as warnings when allowed."""
    root = tmp_path / "food-101"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    (root / "images" / "pizza" / "test_image.jpg").write_bytes(
        (root / "images" / "pizza" / "train_image.jpg").read_bytes()
    )
    _refresh_fixture_marker(root)

    rejected = food101_verifier.audit_food101(
        root, tmp_path / "reports-reject", reject_cross_split_duplicates=True
    )
    allowed = food101_verifier.audit_food101(
        root, tmp_path / "reports-allow", reject_cross_split_duplicates=False
    )

    assert rejected["overall"] == "FAIL"
    assert rejected["criteria"]["train_test_duplicate_content"]["status"] == "FAIL"
    assert allowed["overall"] == "PASS"
    assert allowed["criteria"]["train_test_duplicate_content"]["status"] == "WARN"
    assert allowed["train_test_duplicate_hashes"] == 1


def test_metadata_v2_is_published_and_accepted_by_current_verifier(tmp_path, monkeypatch):
    """Create schema-compatible v2 evidence only from a passing full fixture audit."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify-before"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    result = food101_verifier.audit_food101(root, report_dir)
    report_path = report_dir / "food101_report.json"
    metadata_path = report_root / "food101_metadata_v2.json"

    assert result["overall"] == "PASS"
    published = food101_verifier.publish_metadata_v2(
        root, report_path, metadata_path, report_root
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verified = food101_verifier.verify_metadata_v2(
        root, metadata_path, report_root / "verify-after"
    )
    skipped = food101_verifier.publish_metadata_v2(
        root, report_path, metadata_path, report_root
    )

    assert published["state"] == "PASS"
    assert metadata["schema_version"] == food101_verifier.RAW_METADATA_SCHEMA_VERSION
    assert metadata["complete"] is True
    assert verified["overall"] == "PASS"
    assert verified["criteria"]["download_completion_marker"]["status"] == "PASS"
    assert skipped["state"] == "SKIPPED"


def test_metadata_is_not_created_when_raw_changes_after_audit(tmp_path, monkeypatch):
    """Reject stale evidence at publish time and leave the new metadata path absent."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    food101_verifier.audit_food101(root, report_dir)
    metadata_path = report_root / "food101_metadata_v2.json"
    Image.new("RGB", (15, 15), (80, 10, 20)).save(
        root / "images" / "pizza" / "train_image.jpg"
    )

    with pytest.raises(ValueError, match="changed after verification"):
        food101_verifier.publish_metadata_v2(
            root, report_dir / "food101_report.json", metadata_path, report_root
        )

    assert not metadata_path.exists()


def test_metadata_is_not_created_when_verification_fails(tmp_path, monkeypatch):
    """Refuse metadata publication for a failed quality report and preserve no success file."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    (root / "images" / "pizza" / "test_image.jpg").write_bytes(
        (root / "images" / "pizza" / "train_image.jpg").read_bytes()
    )
    _refresh_fixture_marker(root)
    result = food101_verifier.audit_food101(root, report_dir)
    metadata_path = report_root / "food101_metadata_v2.json"

    assert result["overall"] == "FAIL"
    with pytest.raises(ValueError, match="passing full verification"):
        food101_verifier.publish_metadata_v2(
            root, report_dir / "food101_report.json", metadata_path, report_root
        )
    assert not metadata_path.exists()


@pytest.mark.parametrize("corrupt_value", ["{broken", json.dumps({"schema_version": 99})])
def test_current_verifier_rejects_corrupt_or_unsupported_metadata_v2(
    tmp_path, monkeypatch, corrupt_value
):
    """Reject unreadable or unsupported metadata rather than accepting its completion claim."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    metadata_path = report_root / "food101_metadata_v2.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(corrupt_value, encoding="utf-8")

    result = food101_verifier.verify_metadata_v2(
        root, metadata_path, report_root / "verify"
    )

    assert result["overall"] == "FAIL"
    assert result["criteria"]["download_completion_marker"]["status"] == "FAIL"


def test_current_verifier_rejects_metadata_inventory_mismatch(tmp_path, monkeypatch):
    """Reject a syntactically valid metadata record whose inventory digest was altered."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify-before"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    food101_verifier.audit_food101(root, report_dir)
    metadata_path = report_root / "food101_metadata_v2.json"
    food101_verifier.publish_metadata_v2(
        root, report_dir / "food101_report.json", metadata_path, report_root
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["inventory_sha256"] = "0" * 64
    food101_verifier.atomic_write_json(metadata_path, metadata)

    result = food101_verifier.verify_metadata_v2(
        root, metadata_path, report_root / "verify-after"
    )

    assert result["overall"] == "FAIL"
    assert result["criteria"]["download_completion_marker"]["status"] == "FAIL"


def test_unknown_source_is_blocked_without_metadata_publication(tmp_path, monkeypatch):
    """Block a structurally sound fixture when no local provenance identifies its source."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    (root / "download_metadata.json").unlink()

    result = food101_verifier.audit_food101(root, report_dir)
    metadata_path = report_root / "food101_metadata_v2.json"
    with pytest.raises(ValueError, match="passing full verification"):
        food101_verifier.publish_metadata_v2(
            root, report_dir / "food101_report.json", metadata_path, report_root
        )

    assert result["overall"] == "BLOCKED"
    assert result["criteria"]["download_completion_marker"]["status"] == "BLOCKED"
    assert not metadata_path.exists()


def test_interrupted_audit_cannot_publish_complete_metadata(tmp_path, monkeypatch):
    """Leave no passing report or metadata when verification stops mid-image scan."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)

    def interrupted(path):
        """Simulate an interruption while a fixture image is being checked."""
        raise OSError("simulated verification interruption")

    monkeypatch.setattr(food101_verifier, "_inspect_image", interrupted)
    with pytest.raises(OSError, match="interruption"):
        food101_verifier.audit_food101(root, report_dir, progress_every=1)
    metadata_path = report_root / "food101_metadata_v2.json"
    with pytest.raises((OSError, ValueError)):
        food101_verifier.publish_metadata_v2(
            root, report_dir / "food101_report.json", metadata_path, report_root
        )

    assert not metadata_path.exists()


def test_two_processes_cannot_publish_different_metadata(tmp_path, monkeypatch):
    """Serialize independent metadata writers and preserve one consistent artifact."""
    root, report_root = tmp_path / "food-101", tmp_path / "reports"
    report_dir = report_root / "verify"
    _make_small_food101(root)
    _configure_small_contract(monkeypatch)
    food101_verifier.audit_food101(root, report_dir)
    report_path = report_dir / "food101_report.json"
    metadata_path = report_root / "food101_metadata_v2.json"
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [
        context.Process(
            target=_publish_metadata_worker,
            args=(root, report_path, metadata_path, report_root, start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        outcomes = [results.get(timeout=30) for _ in processes]
    finally:
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join()

    states = [value for kind, value in outcomes if kind == "ok"]
    assert "PASS" in states
    assert all(state in {"PASS", "SKIPPED"} for state in states)
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["complete"] is True


def test_official_dataset_uses_classes_file_order(tmp_path):
    root = tmp_path / "food-101"
    (root / "images" / "zebra_cake").mkdir(parents=True)
    (root / "images" / "apple_pie").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "classes.txt").write_text("zebra_cake\napple_pie\n", encoding="utf-8")
    (root / "meta" / "train.txt").write_text("apple_pie/image\n", encoding="utf-8")
    Image.new("RGB", (4, 5)).save(root / "images" / "apple_pie" / "image.jpg")
    dataset = OfficialFood101Dataset(root, "train")
    image, class_id = dataset[0]
    assert image.shape == (5, 4, 3)
    assert class_id == 1
    with pytest.raises(ValueError, match="split"):
        OfficialFood101Dataset(root, "validation")


def test_audit_lock_rejects_parallel_run(tmp_path):
    lock_path = tmp_path / "food101_audit.lock"
    with food101_verifier.AuditLock(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with food101_verifier.AuditLock(lock_path):
                pass
    assert not lock_path.exists()
