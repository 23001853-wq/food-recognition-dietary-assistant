"""Failure-mode tests for atomic publication, safe extraction, and resource locks."""

import csv
import io
import json
import multiprocessing
import tarfile

import pytest
from PIL import Image

from food_assistant.data_pipeline import atomic
from food_assistant.data_pipeline.load import extract_archive_staged
from food_assistant.ml.datasets import prepare_food101
from scripts import download_food101


def _archive(path, members):
    """Create a small local tar fixture from (name, type, value) tuples."""
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, value in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                body = value.encode("utf-8")
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = value
                archive.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = value
                archive.addfile(info)
            elif kind == "device":
                info.type = tarfile.CHRTYPE
                archive.addfile(info)


def _hold_lock(resource, ready, release):
    """Hold a resource lock in a separate process until released by its parent."""
    with atomic.resource_lock(resource, timeout_seconds=1):
        ready.set()
        release.wait(5)


@pytest.mark.parametrize(
    "member",
    [
        ("../escape.txt", "file", "x"),
        ("/absolute.txt", "file", "x"),
        ("C:/absolute.txt", "file", "x"),
        ("nested/./dot.txt", "file", "x"),
        ("nested/../escape.txt", "file", "x"),
        ("nested/link", "symlink", "../escape.txt"),
        ("nested/link", "hardlink", "nested/data"),
        ("nested/device", "device", ""),
    ],
)
def test_unsafe_archive_members_are_rejected_and_stage_removed(tmp_path, member):
    """Reject traversal, absolute locations, links, and special archive entries."""
    archive = tmp_path / "fixture.tar.gz"
    target = tmp_path / "raw"
    _archive(archive, [member])

    with pytest.raises(ValueError):
        extract_archive_staged(archive, target, lambda staging: True)

    assert not target.exists()
    assert list(tmp_path.glob(".raw.staging-*")) == []


def test_duplicate_archive_path_is_rejected(tmp_path):
    """Reject repeated normalized archive paths before any destination publish."""
    archive = tmp_path / "duplicate.tar.gz"
    _archive(archive, [("image.jpg", "file", "one"), ("image.jpg", "file", "two")])

    with pytest.raises(ValueError, match="Duplicate"):
        extract_archive_staged(archive, tmp_path / "raw", lambda staging: True)


def test_case_insensitive_windows_path_collision_is_rejected(tmp_path):
    """Reject names that alias on the Windows filesystem used for publication."""
    archive = tmp_path / "case-collision.tar.gz"
    _archive(archive, [("images/food.jpg", "file", "one"), ("images/FOOD.jpg", "file", "two")])

    with pytest.raises(ValueError, match="Duplicate"):
        extract_archive_staged(archive, tmp_path / "raw", lambda staging: True)


def test_existing_raw_directory_is_never_opened_for_overwrite(tmp_path):
    """Reject an existing raw path before invoking the archive validator or writer."""
    archive = tmp_path / "fixture.tar.gz"
    target = tmp_path / "raw"
    target.mkdir()
    good_image = target / "good.jpg"
    good_image.write_bytes(b"known-good")
    _archive(archive, [("image.jpg", "file", "new-data")])

    validator_called = False

    def reject(staging):
        """Record whether validation was reached for a protected raw target."""
        nonlocal validator_called
        validator_called = True
        return False

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        extract_archive_staged(archive, target, reject)

    assert validator_called is False
    assert good_image.read_bytes() == b"known-good"
    assert not (target / "image.jpg").exists()


def test_mid_archive_failure_keeps_existing_raw_and_cleans_new_stage(tmp_path):
    """Remove a partially written stage after a later unsafe member is encountered."""
    archive = tmp_path / "partial.tar.gz"
    existing_raw = tmp_path / "raw"
    new_target = tmp_path / "raw-new"
    existing_raw.mkdir()
    good_image = existing_raw / "good.jpg"
    good_image.write_bytes(b"known-good")
    _archive(archive, [("first.bin", "file", "written first"), ("../escape.bin", "file", "bad")])

    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_archive_staged(archive, new_target, lambda staging: True)

    assert good_image.read_bytes() == b"known-good"
    assert not new_target.exists()
    assert list(tmp_path.glob(".raw-new.staging-*")) == []


def test_validator_failure_cleans_new_staging_directory(tmp_path):
    """Discard a half-extracted stage when its contract check fails."""
    archive = tmp_path / "fixture.tar.gz"
    target = tmp_path / "raw"
    _archive(archive, [("first.bin", "file", "first"), ("second.bin", "file", "second")])

    with pytest.raises(ValueError, match="failed its validation"):
        extract_archive_staged(archive, target, lambda staging: False)

    assert not target.exists()
    assert list(tmp_path.glob(".raw.staging-*")) == []


def test_atomic_json_replace_failure_preserves_previous_report(tmp_path, monkeypatch):
    """Leave the old complete report intact if atomic replacement fails."""
    report = tmp_path / "report.json"
    report.write_text('{"complete": true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        atomic.atomic_write_json(report, {"complete": False})

    assert report.read_text(encoding="utf-8") == '{"complete": true}\n'
    assert list(tmp_path.glob("*.pending")) == []


def test_atomic_writer_retries_transient_windows_replace_error(tmp_path, monkeypatch):
    """Retry transient Windows file-share conflicts and publish the complete new file."""
    target = tmp_path / "report.txt"
    target.write_text("old", encoding="utf-8")
    original_replace = atomic.os.replace
    attempts = 0

    def replace_once_fails(source, destination):
        """Simulate one antivirus/file-share denial before a successful replacement."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient file-share conflict")
        original_replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", replace_once_fails)
    atomic.atomic_write_text(target, "new")

    assert attempts == 2
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_text_and_csv_writers_publish_complete_files(tmp_path):
    """Publish complete text and CSV outputs using the sibling-temp mechanism."""
    atomic.atomic_write_text(tmp_path / "report.txt", "complete\n")
    atomic.atomic_write_csv(tmp_path / "rows.csv", [{"id": 1}], ["id"])

    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == "complete\n"
    with (tmp_path / "rows.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"id": "1"}]


def test_process_lock_rejects_second_writer(tmp_path):
    """Ensure an independent process cannot acquire the same resource lock."""
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_lock, args=(tmp_path / "manifest", ready, release))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(RuntimeError, match="Resource is locked"):
            with atomic.resource_lock(tmp_path / "manifest", timeout_seconds=0):
                pytest.fail("A competing process acquired an already held lock")
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_directory_is_published_only_after_validation(tmp_path):
    """Publish a fully validated staged directory and leave no staging residue."""
    archive = tmp_path / "valid.tar.gz"
    target = tmp_path / "raw"
    _archive(archive, [("meta/classes.txt", "file", "one\n"), ("image.jpg", "file", "fixture")])

    published = extract_archive_staged(
        archive,
        target,
        lambda staging: (staging / "meta/classes.txt").is_file()
        and (staging / "image.jpg").read_text(encoding="utf-8") == "fixture",
    )

    assert published == target
    assert (target / "image.jpg").read_text(encoding="utf-8") == "fixture"
    assert list(tmp_path.glob(".raw.staging-*")) == []


def test_food101_loader_publishes_complete_marker_only_after_fixture_validation(
    tmp_path, monkeypatch
):
    """Publish a tiny contract fixture with complete provenance only after all checks pass."""
    monkeypatch.setattr(download_food101, "EXPECTED_CLASSES", 1)
    monkeypatch.setattr(download_food101, "EXPECTED_IMAGES", 2)
    monkeypatch.setattr(download_food101, "EXPECTED_TRAIN_PER_CLASS", 1)
    monkeypatch.setattr(download_food101, "EXPECTED_TEST_PER_CLASS", 1)
    image_buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (42, 10, 8)).save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    second_image_buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (8, 10, 42)).save(second_image_buffer, format="JPEG")
    second_image_bytes = second_image_buffer.getvalue()
    archive_path = tmp_path / "food101_fixture.tar.gz"
    members = [
        ("food-101/meta/classes.txt", b"pizza\n"),
        ("food-101/meta/train.txt", b"pizza/train_image\n"),
        ("food-101/meta/test.txt", b"pizza/test_image\n"),
        ("food-101/license_agreement.txt", b"fixture terms"),
        ("food-101/images/pizza/train_image.jpg", image_bytes),
        ("food-101/images/pizza/test_image.jpg", second_image_bytes),
    ]
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, body in members:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))

    dataset = download_food101.safe_extract(
        archive_path,
        tmp_path / "raw",
        {"dataset": "food101", "download_sha256": "a" * 64},
    )

    marker = json.loads((dataset / "download_metadata.json").read_text(encoding="utf-8"))
    assert marker["complete"] is True
    assert marker["schema_version"] == 1
    assert marker["integrity_status"] == "PASS"
    assert marker["image_count"] == 2
    assert len(marker["inventory_sha256"]) == 64


def test_prepare_manifest_publishes_new_version_and_refuses_existing_target(tmp_path):
    """Build in staging, then refuse to change any already-published manifest path."""
    root = tmp_path / "food-101"
    (root / "meta").mkdir(parents=True)
    (root / "images" / "pizza").mkdir(parents=True)
    (root / "meta" / "classes.txt").write_text("pizza\n", encoding="utf-8")
    train_names = [f"pizza/train_{index}" for index in range(5)]
    test_names = ["pizza/test_0", "pizza/test_1"]
    (root / "meta" / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (root / "meta" / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    for index, name in enumerate(train_names + test_names):
        Image.new("RGB", (8, 8), (index * 20, index * 20, index * 20)).save(
            root / "images" / f"{name}.jpg"
        )
    output = tmp_path / "splits" / "v2"

    stats = prepare_food101(root, output, ["pizza"], version="v2", seed=11, val_fraction=0.2)

    assert stats["image_count"] == len(train_names) + len(test_names)
    assert (output / "dataset_manifest.csv").is_file()
    sentinel = output / "preserve.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_food101(root, output, ["pizza"], version="v2", seed=11, val_fraction=0.2)
    assert sentinel.read_text(encoding="utf-8") == "keep"
