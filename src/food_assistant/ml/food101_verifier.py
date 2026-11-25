"""Read-only structural and image integrity audit for the official Food-101 release."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from food_assistant.data_pipeline.atomic import (
    atomic_write_json,
    checkpoint_digest,
    resource_lock,
)

EXPECTED_CLASSES = 101
EXPECTED_IMAGES = 101_000
EXPECTED_TRAIN_PER_CLASS = 750
EXPECTED_TEST_PER_CLASS = 250
CHECKPOINT_VERSION = 2
RAW_METADATA_SCHEMA_VERSION = 1
VERIFICATION_POLICY_VERSION = 1


def _read_lines(path: Path) -> list[str]:
    """Read non-empty UTF-8 lines from a Food-101 metadata file."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fingerprint(path: Path) -> dict[str, int]:
    """Return a cheap file signature used to invalidate stale image checkpoints."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one existing file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_inventory_hash(image_results: dict[str, Any]) -> str:
    """Hash the ordered relative paths and verified image digests as one inventory."""
    inventory = hashlib.sha256()
    for relative_path, entry in sorted(image_results.items()):
        image_digest = entry.get("result", {}).get("sha256")
        if image_digest:
            inventory.update(f"{relative_path}:{image_digest}\n".encode())
    return inventory.hexdigest()


def _source_fingerprint(
    inventory_sha256: str, metadata_sha256: dict[str, str], source_sha256: str
) -> str:
    """Bind verified image inventory, official metadata, and source archive identity."""
    payload = {
        "inventory_sha256": inventory_sha256,
        "metadata_sha256": metadata_sha256,
        "source_sha256": source_sha256,
    }
    return checkpoint_digest(payload)


def _filesystem_input_hash(dataset_root: Path, relative_paths: list[str]) -> str:
    """Fingerprint metadata and image file signatures for verification-to-publish checks."""
    paths = sorted(set(relative_paths))
    signatures = {
        relative: _fingerprint(dataset_root / Path(relative))
        for relative in paths
        if (dataset_root / Path(relative)).is_file()
    }
    return checkpoint_digest(signatures)


def _save_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file to avoid partial checkpoints."""
    atomic_write_json(path, value)


def _load_checkpoint(
    path: Path, dataset_root: Path, expected_files: list[str], input_hash: str
) -> dict[str, Any]:
    """Load only a self-consistent versioned checkpoint for the exact expected inventory."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = value["payload"]
        if (
            not isinstance(value, dict)
            or not isinstance(payload, dict)
            or value.get("sha256") != checkpoint_digest(payload)
            or payload.get("schema_version") != CHECKPOINT_VERSION
            or payload.get("dataset_root") != str(dataset_root.resolve())
            or payload.get("expected_files") != expected_files
            or payload.get("input_hash") != input_hash
            or not isinstance(payload.get("files"), dict)
            or payload.get("complete") is not True
            or not set(payload["files"]).issubset(expected_files)
            or set(payload["files"]) != set(expected_files)
            or any(not entry.get("result", {}).get("valid") for entry in payload["files"].values())
        ):
            raise ValueError("Checkpoint contract mismatch")
        return payload
    except (OSError, json.JSONDecodeError, AttributeError):
        return {"files": {}}
    except (KeyError, TypeError, ValueError):
        return {"files": {}}


def _checkpoint_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap checkpoint payload with a deterministic integrity digest."""
    return {"payload": payload, "sha256": checkpoint_digest(payload)}


class AuditLock:
    """Prevent concurrent verifier runs using an exclusive, process-owned lock file."""

    def __init__(self, path: Path):
        """Create a lock manager for the specified lock file."""
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> AuditLock:
        """Acquire the lock or fail with the owner information."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = "unknown"
            try:
                owner = self.path.read_text(encoding="utf-8")
                os.kill(int(owner), 0)
            except ProcessLookupError:
                self.path.unlink(missing_ok=True)
                try:
                    self._descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError as retry_exc:
                    raise RuntimeError(
                        "Food-101 audit lock was reacquired by another process"
                    ) from retry_exc
            except (OSError, ValueError):
                raise RuntimeError(
                    f"Food-101 audit already running or stale lock exists: {owner}"
                ) from exc
            else:
                raise RuntimeError(f"Food-101 audit already running with PID {owner}") from exc
        os.write(self._descriptor, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Release the lock owned by this process."""
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _inspect_image(path: Path) -> dict[str, Any]:
    """Decode an image and compute its SHA-256 without modifying it."""
    try:
        if path.stat().st_size <= 0:
            raise ValueError("empty file")
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
        if image_format != "JPEG":
            raise ValueError(f"unexpected image format: {image_format}")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions: {width}x{height}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"valid": True, "sha256": digest.hexdigest(), "width": width, "height": height}
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


def audit_food101(
    dataset_root: Path,
    report_dir: Path,
    mode: str = "full",
    progress_every: int = 500,
    reject_cross_split_duplicates: bool = True,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Audit Food-101 and atomically save evidence under a report-scoped process lock."""
    with AuditLock(Path(report_dir) / "food101_audit.lock"):
        return _audit_food101(
            dataset_root,
            report_dir,
            mode,
            progress_every,
            reject_cross_split_duplicates,
            metadata_path,
        )


def _audit_food101(
    dataset_root: Path,
    report_dir: Path,
    mode: str = "full",
    progress_every: int = 500,
    reject_cross_split_duplicates: bool = True,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Audit official Food-101 structure, metadata and image integrity, then save reports."""
    dataset_root, report_dir = Path(dataset_root), Path(report_dir)
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    report_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("food101_audit")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(report_dir / "food101_audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    started = time.time()
    issues: list[dict[str, str]] = []
    criteria: dict[str, dict[str, Any]] = {}

    def criterion(name: str, passed: bool | None | str, details: Any) -> None:
        """Record a named pass/fail criterion in the final report."""
        status = (
            passed
            if isinstance(passed, str)
            else "NOT_RUN"
            if passed is None
            else "PASS"
            if passed
            else "FAIL"
        )
        criteria[name] = {"status": status, "details": details}

    required = [dataset_root / "images", dataset_root / "meta"]
    criterion("dataset_root", dataset_root.is_dir(), str(dataset_root.resolve()))
    criterion("images_directory", required[0].is_dir(), str(required[0]))
    criterion("metadata_directory", required[1].is_dir(), str(required[1]))
    meta_files = {
        name: dataset_root / "meta" / name for name in ("classes.txt", "train.txt", "test.txt")
    }
    missing_meta = [str(path) for path in meta_files.values() if not path.is_file()]
    criterion("required_metadata_files", not missing_meta, {"missing": missing_meta})
    if missing_meta or not dataset_root.is_dir():
        result = {"mode": mode, "criteria": criteria, "overall": "FAIL", "issues": issues}
        _write_reports(report_dir, result)
        return result

    classes = _read_lines(meta_files["classes.txt"])
    class_duplicates = sorted(key for key, count in Counter(classes).items() if count > 1)
    class_dirs = sorted(path.name for path in (dataset_root / "images").iterdir() if path.is_dir())
    criterion(
        "class_labels_and_directories",
        len(classes) == EXPECTED_CLASSES
        and len(set(classes)) == EXPECTED_CLASSES
        and class_dirs == sorted(classes),
        {
            "metadata_classes": len(classes),
            "unique_labels": len(set(classes)),
            "image_directories": len(class_dirs),
            "duplicate_labels": class_duplicates,
            "directories_without_metadata": sorted(set(class_dirs) - set(classes)),
            "metadata_without_directories": sorted(set(classes) - set(class_dirs)),
        },
    )
    train_names, test_names = (
        _read_lines(meta_files["train.txt"]),
        _read_lines(meta_files["test.txt"]),
    )
    train_set, test_set = set(train_names), set(test_names)
    path_overlap = sorted(train_set & test_set)
    repeated_train_names = sorted(name for name, count in Counter(train_names).items() if count > 1)
    repeated_test_names = sorted(name for name, count in Counter(test_names).items() if count > 1)
    train_counts, test_counts = (
        Counter(name.split("/", 1)[0] for name in train_names),
        Counter(name.split("/", 1)[0] for name in test_names),
    )
    all_names = train_set | test_set
    invalid_names = sorted(
        name
        for name in all_names
        if len(name.split("/")) != 2
        or name.split("/", 1)[0] not in set(classes)
        or "\\" in name
        or ":" in name
        or ".." in name.split("/")
    )
    split_counts_valid = (
        len(train_names) == EXPECTED_CLASSES * EXPECTED_TRAIN_PER_CLASS
        and len(test_names) == EXPECTED_CLASSES * EXPECTED_TEST_PER_CLASS
        and not path_overlap
        and not repeated_train_names
        and not repeated_test_names
        and not invalid_names
        and all(
            train_counts[key] == EXPECTED_TRAIN_PER_CLASS
            and test_counts[key] == EXPECTED_TEST_PER_CLASS
            for key in classes
        )
    )
    criterion(
        "official_split_metadata",
        split_counts_valid,
        {
            "train_entries": len(train_names),
            "test_entries": len(test_names),
            "train_per_class": dict(train_counts),
            "test_per_class": dict(test_counts),
            "overlap_paths": path_overlap,
            "repeated_train_entries": repeated_train_names,
            "repeated_test_entries": repeated_test_names,
            "invalid_entries": invalid_names,
        },
    )

    split_json_paths = {
        split: dataset_root / "meta" / f"{split}.json" for split in ("train", "test")
    }
    existing_split_json = [path.is_file() for path in split_json_paths.values()]
    if any(existing_split_json):
        split_json_valid = all(existing_split_json)
        split_json_details: dict[str, Any] = {"present": existing_split_json}
        if split_json_valid:
            try:
                split_json_values = {
                    split: json.loads(path.read_text(encoding="utf-8"))
                    for split, path in split_json_paths.items()
                }
                json_names: dict[str, list[str]] = {}
                for split, value in split_json_values.items():
                    if not isinstance(value, dict) or set(value) != set(classes):
                        split_json_valid = False
                        continue
                    names = [
                        name
                        for values in value.values()
                        if isinstance(values, list)
                        for name in values
                    ]
                    if any(not isinstance(values, list) for values in value.values()):
                        split_json_valid = False
                    json_names[split] = names
                if set(json_names) == {"train", "test"}:
                    split_json_valid = split_json_valid and (
                        set(json_names["train"]) == set(train_names)
                        and set(json_names["test"]) == set(test_names)
                        and len(json_names["train"]) == len(set(json_names["train"]))
                        and len(json_names["test"]) == len(set(json_names["test"]))
                    )
                    split_json_details.update(
                        {
                            "train_entries": len(json_names["train"]),
                            "test_entries": len(json_names["test"]),
                            "train_ids_match_text": set(json_names["train"]) == set(train_names),
                            "test_ids_match_text": set(json_names["test"]) == set(test_names),
                        }
                    )
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                split_json_valid = False
                split_json_details["error"] = f"{type(exc).__name__}: {exc}"
        criterion("official_json_split_metadata", split_json_valid, split_json_details)
    else:
        criterion("official_json_split_metadata", "NOT_RUN", "train.json/test.json are absent")

    labels_path = dataset_root / "meta" / "labels.txt"
    if labels_path.is_file():
        labels = _read_lines(labels_path)
        criterion(
            "official_display_labels",
            len(labels) == len(classes) and len(set(labels)) == len(labels),
            {"label_count": len(labels), "class_count": len(classes)},
        )
    else:
        criterion("official_display_labels", "NOT_RUN", "labels.txt is absent")

    expected = {f"images/{name}.jpg" for name in all_names}
    image_files = [path for path in (dataset_root / "images").rglob("*") if path.is_file()]
    actual_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in image_files
        if path.suffix.lower() == ".jpg"
    }
    non_jpg_files = sorted(
        path.relative_to(dataset_root).as_posix()
        for path in image_files
        if path.suffix.lower() != ".jpg"
    )
    missing_paths = sorted(expected - actual_paths)
    unlisted_paths = sorted(actual_paths - expected)
    per_class_files = {
        key: sum(1 for path in actual_paths if path.startswith(f"images/{key}/")) for key in classes
    }
    total_actual = len(actual_paths)
    expected_images_per_class = EXPECTED_TRAIN_PER_CLASS + EXPECTED_TEST_PER_CLASS
    structure_ok = (
        total_actual == EXPECTED_IMAGES
        and all(count == expected_images_per_class for count in per_class_files.values())
        and not missing_paths
        and not unlisted_paths
        and not non_jpg_files
    )
    criterion(
        "image_inventory",
        structure_ok,
        {
            "actual_jpg": total_actual,
            "expected_jpg": EXPECTED_IMAGES,
            "per_class": per_class_files,
            "missing_count": len(missing_paths),
            "missing_paths": missing_paths,
            "unlisted_count": len(unlisted_paths),
            "unlisted_paths": unlisted_paths,
            "non_jpg_files": non_jpg_files,
        },
    )

    marker_path = (
        Path(metadata_path)
        if metadata_path is not None
        else dataset_root / "download_metadata.json"
    )
    try:
        marker = (
            json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
        )
        marker_sha256_valid = (
            isinstance(marker, dict)
            and marker.get("dataset") == "food101"
            and marker.get("image_count") == EXPECTED_IMAGES
            and isinstance(marker.get("download_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", marker["download_sha256"]) is not None
        )
        marker_origin_identified = bool(
            isinstance(marker, dict) and (marker.get("archive_url") or marker.get("source_url"))
        )
        checksum_info = {
            "local_sha256": marker.get("download_sha256") if isinstance(marker, dict) else None,
            "official_sha256": marker.get("official_sha256") if isinstance(marker, dict) else None,
            "official_checksum_verified": bool(marker and marker.get("official_checksum_verified")),
            "note": "A local SHA-256 is not an independently verified official checksum.",
        }
    except (OSError, json.JSONDecodeError) as exc:
        marker, marker_sha256_valid, marker_origin_identified, checksum_info = (
            None,
            False,
            False,
            {"error": f"{type(exc).__name__}: {exc}"},
        )
    archive_path = dataset_root.parent / "food-101.tar.gz.part"
    archive_digest = None
    archive_matches = None
    official_matches = None
    if archive_path.is_file() and mode == "full":
        logger.info("Recomputing SHA-256 for retained archive: %s", archive_path)
        digest = hashlib.sha256()
        with archive_path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        archive_digest = digest.hexdigest()
        archive_matches = bool(marker and marker.get("download_sha256") == archive_digest)
        official_sha256 = marker.get("official_sha256") if isinstance(marker, dict) else None
        official_matches = archive_digest == official_sha256 if official_sha256 else None
        checksum_info.update(
            {
                "archive_present": True,
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256_recomputed": archive_digest,
                "archive_matches_download_metadata": archive_matches,
                "archive_matches_official": official_matches,
                "official_checksum_verified": official_matches is True,
            }
        )
        logger.info("Archive SHA-256 check finished; matches download marker: %s", archive_matches)
    else:
        checksum_info["archive_present"] = archive_path.is_file()
        if archive_path.is_file():
            checksum_info.update(
                {"archive_bytes": archive_path.stat().st_size, "archive_recomputed": False}
            )
    checksum_info["archive_matches_download_metadata"] = archive_matches
    checksum_info["archive_matches_official"] = official_matches
    license_path = dataset_root / "license_agreement.txt"
    criterion("license_file_present", license_path.is_file(), str(license_path))

    image_results: dict[str, Any] = {}
    checkpoint_path = report_dir / "food101_checkpoint.json"
    expected_paths = sorted(expected & actual_paths)
    fingerprint_metadata_paths = [
        path.relative_to(dataset_root).as_posix()
        for path in [
            *meta_files.values(),
            dataset_root / "license_agreement.txt",
            split_json_paths["train"],
            split_json_paths["test"],
            labels_path,
        ]
        if path.is_file()
    ]
    filesystem_paths = sorted(set(expected_paths + fingerprint_metadata_paths))
    metadata_sha256 = (
        {
            relative: _sha256_file(dataset_root / Path(relative))
            for relative in fingerprint_metadata_paths
        }
        if mode == "full"
        else {}
    )
    image_signatures: dict[str, dict[str, int]] = {}
    input_hash = ""
    if mode == "full":
        source_signatures = {
            relative: _fingerprint(dataset_root / Path(relative)) for relative in filesystem_paths
        }
        image_signatures = {relative: source_signatures[relative] for relative in expected_paths}
        input_hash = checkpoint_digest(source_signatures)
        checkpoint = _load_checkpoint(checkpoint_path, dataset_root, expected_paths, input_hash)
        old_results = checkpoint.get("files", {})
        files: dict[str, Any] = {}
        done = 0
        for relative in sorted(expected & actual_paths):
            path = dataset_root / Path(relative)
            signature = image_signatures[relative]
            previous = old_results.get(relative, {})
            if not isinstance(previous, dict):
                previous = {}
            previous_result = previous.get("result", {})
            if not isinstance(previous_result, dict):
                previous_result = {}
            if previous.get("signature") == signature and previous_result.get("valid"):
                result = previous_result
            else:
                result = _inspect_image(path)
            files[relative] = {"signature": signature, "result": result}
            done += 1
            if done % max(progress_every, 1) == 0:
                logger.info(
                    "Checked %s/%s images (%s valid, %s errors)",
                    done,
                    len(expected & actual_paths),
                    sum(entry["result"]["valid"] for entry in files.values()),
                    sum(not entry["result"]["valid"] for entry in files.values()),
                )
                checkpoint_payload = {
                    "schema_version": CHECKPOINT_VERSION,
                    "dataset_root": str(dataset_root.resolve()),
                    "input_hash": input_hash,
                    "expected_files": expected_paths,
                    "files": files,
                    "complete": False,
                }
                _save_json_atomic(checkpoint_path, _checkpoint_envelope(checkpoint_payload))
        checkpoint_payload = {
            "schema_version": CHECKPOINT_VERSION,
            "dataset_root": str(dataset_root.resolve()),
            "input_hash": input_hash,
            "expected_files": expected_paths,
            "files": files,
            "complete": len(files) == len(expected_paths)
            and all(entry.get("result", {}).get("valid") for entry in files.values()),
        }
        _save_json_atomic(checkpoint_path, _checkpoint_envelope(checkpoint_payload))
        image_results = files
    valid_count = sum(entry["result"]["valid"] for entry in image_results.values())
    bad_images = [
        {"path": path, **entry["result"]}
        for path, entry in image_results.items()
        if not entry["result"]["valid"]
    ]
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for path, entry in image_results.items():
        digest = entry["result"].get("sha256")
        if digest:
            duplicate_groups[digest].append(path)
    duplicates = [paths for paths in duplicate_groups.values() if len(paths) > 1]
    if mode == "full":
        train_hashes = {
            image_results[path]["result"].get("sha256")
            for path in image_results
            if path.removeprefix("images/").removesuffix(".jpg") in train_set
        }
        test_hashes = {
            image_results[path]["result"].get("sha256")
            for path in image_results
            if path.removeprefix("images/").removesuffix(".jpg") in test_set
        }
        hash_leaks = sorted(digest for digest in train_hashes & test_hashes if digest)
        train_image_paths = {f"images/{name}.jpg" for name in train_set}
        test_image_paths = {f"images/{name}.jpg" for name in test_set}
        cross_split_duplicate_groups = [
            paths
            for paths in duplicates
            if set(paths) & train_image_paths and set(paths) & test_image_paths
        ]
        criterion(
            "train_test_duplicate_content",
            "FAIL"
            if hash_leaks and reject_cross_split_duplicates
            else "WARN"
            if hash_leaks
            else True,
            {
                "matching_sha256_count": len(hash_leaks),
                "sha256": hash_leaks,
                "duplicate_path_groups": cross_split_duplicate_groups,
                "policy": "reject_cross_split_duplicates",
                "policy_version": VERIFICATION_POLICY_VERSION,
                "policy_enabled": reject_cross_split_duplicates,
            },
        )
    else:
        hash_leaks = []
        cross_split_duplicate_groups = []
        criterion("train_test_duplicate_content", None, "Not checked in quick mode.")
    if mode == "full":
        criterion(
            "image_decoding_and_sha256",
            valid_count == len(expected & actual_paths),
            {
                "checked": len(image_results),
                "valid": valid_count,
                "errors": len(bad_images),
                "bad_images": bad_images,
                "duplicate_content_groups": len(duplicates),
                "duplicate_groups": duplicates,
                "train_test_duplicate_hashes": len(hash_leaks),
                "train_test_duplicate_sha256": hash_leaks,
            },
        )
    else:
        criterion(
            "image_decoding_and_sha256",
            None,
            "Not run in quick mode; run --mode full for Pillow decode and content hashes.",
        )

    inventory_sha256 = _content_inventory_hash(image_results) if mode == "full" else None
    source_sha256 = marker.get("download_sha256", "") if isinstance(marker, dict) else ""
    source_fingerprint = (
        _source_fingerprint(inventory_sha256, metadata_sha256, source_sha256)
        if inventory_sha256 is not None
        else None
    )
    marker_is_phase2 = (
        isinstance(marker, dict)
        and marker.get("schema_version") == RAW_METADATA_SCHEMA_VERSION
        and marker.get("complete") is True
        and marker.get("integrity_status") == "PASS"
        and isinstance(marker.get("inventory_sha256"), str)
    )
    if metadata_path is not None and not marker_sha256_valid:
        marker_status: bool | str = "FAIL"
        marker_reason = "Metadata v2 is corrupt or has an invalid dataset identity/digest."
    elif not marker_origin_identified:
        marker_status = "BLOCKED"
        marker_reason = "The dataset source cannot be identified from local metadata."
    elif not marker_sha256_valid:
        marker_status = "FAIL"
        marker_reason = "Source metadata has an invalid dataset identity or digest."
    elif archive_matches is False or official_matches is False:
        marker_status = "FAIL"
        marker_reason = "Retained archive digest does not match its recorded source metadata."
    elif mode != "full":
        marker_status = "INCOMPLETE"
        marker_reason = "A full image and source inventory is required."
    elif metadata_path is not None:
        required_metadata_matches = (
            marker.get("schema_version") == RAW_METADATA_SCHEMA_VERSION
            and marker.get("complete") is True
            and marker.get("integrity_status") == "PASS"
            and marker.get("image_count") == valid_count == EXPECTED_IMAGES
            and marker.get("verified_file_count") == valid_count
            and marker.get("expected_file_count") == EXPECTED_IMAGES
            and marker.get("class_count") == len(classes) == EXPECTED_CLASSES
            and marker.get("inventory_sha256") == inventory_sha256
            and marker.get("source_fingerprint") == source_fingerprint
            and marker.get("filesystem_fingerprint") == input_hash
            and marker.get("metadata_sha256") == metadata_sha256
            and marker.get("verification_state") == "PASS"
            and marker.get("verification_policy")
            == {
                "version": VERIFICATION_POLICY_VERSION,
                "reject_cross_split_duplicates": reject_cross_split_duplicates,
            }
        )
        marker_status = required_metadata_matches
        marker_reason = "External metadata v2 agrees with the current full verification evidence."
    elif marker_is_phase2:
        marker_status = marker.get("inventory_sha256") == inventory_sha256
        marker_reason = "Phase 2 marker matches the independently verified image inventory."
    elif archive_matches is True:
        marker_status = True
        marker_reason = (
            "Legacy origin metadata is bound to the retained archive; "
            "v2 evidence was freshly recomputed."
        )
    else:
        marker_status = "INCOMPLETE"
        marker_reason = (
            "Legacy metadata lacks v2 evidence and no matching retained archive was verified."
        )
    criterion(
        "download_completion_marker",
        marker_status,
        {
            "path": str(marker_path),
            "marker_claims": marker if isinstance(marker, dict) else None,
            "reason": marker_reason,
            "checksum": checksum_info,
        },
    )

    image_issue_count = len(bad_images)
    has_failure = any(value["status"] == "FAIL" for value in criteria.values())
    has_blocker = any(value["status"] == "BLOCKED" for value in criteria.values())
    has_incomplete = any(value["status"] == "INCOMPLETE" for value in criteria.values())
    overall = (
        "FAIL"
        if has_failure or image_issue_count > 0
        else "BLOCKED"
        if has_blocker
        else "INCOMPLETE"
        if has_incomplete or mode != "full"
        else "PASS"
    )
    result = {
        "dataset_root": str(dataset_root.resolve()),
        "mode": mode,
        "started_at_epoch": started,
        "duration_seconds": round(time.time() - started, 2),
        "actual_image_count": total_actual,
        "class_count": len(classes),
        "train_count": len(train_names),
        "test_count": len(test_names),
        "valid_images": valid_count if mode == "full" else None,
        "bad_image_count": image_issue_count if mode == "full" else None,
        "missing_image_count": len(missing_paths),
        "unlisted_image_count": len(unlisted_paths),
        "duplicate_content_groups": len(duplicates) if mode == "full" else None,
        "train_test_duplicate_hashes": len(hash_leaks) if mode == "full" else None,
        "expected_file_count": EXPECTED_IMAGES,
        "verified_file_count": valid_count if mode == "full" else None,
        "inventory_sha256": inventory_sha256,
        "source_fingerprint": source_fingerprint,
        "filesystem_fingerprint": input_hash if mode == "full" else None,
        "metadata_sha256": metadata_sha256 if mode == "full" else None,
        "filesystem_paths": filesystem_paths,
        "verification_policy": {
            "version": VERIFICATION_POLICY_VERSION,
            "reject_cross_split_duplicates": reject_cross_split_duplicates,
        },
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "issues": bad_images,
        "warnings": [
            {
                "type": "duplicate_content",
                "message": "Exact duplicates may affect split independence and test evaluation.",
                "duplicate_content_groups": len(duplicates),
                "train_test_duplicate_hashes": len(hash_leaks),
                "train_test_duplicate_path_groups": cross_split_duplicate_groups,
            }
        ]
        if duplicates
        else [],
        "criteria": criteria,
        "overall": overall,
        "data_loader": {
            "status": "NOT_RUN",
            "scope": (
                "Run --dataloader after full audit; this verifier does not mutate "
                "the frozen POC manifest."
            ),
        },
    }
    logger.info(
        "Audit finished: %s (%s valid images, %s image errors)",
        overall,
        result["valid_images"],
        result["bad_image_count"],
    )
    _write_reports(report_dir, result)
    return result


def _validate_passing_report(
    dataset_root: Path, report_path: Path, report_root: Path
) -> dict[str, Any]:
    """Validate a passing report and confirm its raw inputs stayed unchanged."""
    root = Path(dataset_root).resolve()
    report_file, allowed_root = Path(report_path).resolve(), Path(report_root).resolve()
    if not report_file.is_relative_to(allowed_root):
        raise ValueError("Verification report must be inside the configured report root")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("overall") != "PASS"
        or report.get("mode") != "full"
        or report.get("dataset_root") != str(root)
        or report.get("verified_file_count") != EXPECTED_IMAGES
        or report.get("expected_file_count") != EXPECTED_IMAGES
        or not report.get("source_fingerprint")
        or not report.get("filesystem_fingerprint")
    ):
        raise ValueError("Metadata v2 requires a complete, passing full verification report")
    failed_criteria = {
        name: value
        for name, value in report.get("criteria", {}).items()
        if value.get("status") not in {"PASS", "WARN", "NOT_RUN"}
    }
    if failed_criteria:
        raise ValueError(f"Verification report has non-passing criteria: {sorted(failed_criteria)}")
    filesystem_paths = report.get("filesystem_paths")
    if (
        not isinstance(filesystem_paths, list)
        or _filesystem_input_hash(root, filesystem_paths) != report["filesystem_fingerprint"]
    ):
        raise ValueError("Raw filesystem changed after verification")

    metadata_paths = [path for path in filesystem_paths if not path.startswith("images/")]
    metadata_sha256 = {path: _sha256_file(root / Path(path)) for path in metadata_paths}
    if metadata_sha256 != report.get("metadata_sha256"):
        raise ValueError("Official metadata changed after verification")
    train_names = _read_lines(root / "meta" / "train.txt")
    test_names = _read_lines(root / "meta" / "test.txt")
    image_paths = sorted({f"images/{name}.jpg" for name in train_names + test_names})
    actual_image_files = [path for path in (root / "images").rglob("*") if path.is_file()]
    actual_image_paths = {
        path.relative_to(root).as_posix()
        for path in actual_image_files
        if path.suffix.lower() == ".jpg"
    }
    class_names = _read_lines(root / "meta" / "classes.txt")
    class_directories = sorted(path.name for path in (root / "images").iterdir() if path.is_dir())
    if actual_image_paths != set(image_paths) or len(actual_image_files) != len(image_paths):
        raise ValueError("Raw image inventory changed after verification")
    if class_directories != sorted(class_names):
        raise ValueError("Raw class directories changed after verification")
    image_results: dict[str, Any] = {}
    for relative in image_paths:
        image_results[relative] = {"result": _inspect_image(root / Path(relative))}
    if any(not entry["result"].get("valid") for entry in image_results.values()):
        raise ValueError("An image failed revalidation before metadata publication")
    inventory_sha256 = _content_inventory_hash(image_results)
    marker_details = report["criteria"]["download_completion_marker"]["details"]
    source_marker = marker_details.get("marker_claims") or {}
    source_sha256 = source_marker.get("download_sha256", "")
    source_fingerprint = _source_fingerprint(inventory_sha256, metadata_sha256, source_sha256)
    if inventory_sha256 != report.get("inventory_sha256") or source_fingerprint != report.get(
        "source_fingerprint"
    ):
        raise ValueError("Raw content fingerprint changed after verification")
    archive_digest = marker_details.get("checksum", {}).get("archive_sha256_recomputed")
    archive_path = root.parent / "food-101.tar.gz.part"
    if archive_digest and (
        not archive_path.is_file() or _sha256_file(archive_path) != archive_digest
    ):
        raise ValueError("Retained source archive changed after verification")
    return report


def publish_metadata_v2(
    dataset_root: Path,
    report_path: Path,
    metadata_path: Path,
    report_root: Path,
) -> dict[str, Any]:
    """Atomically publish Phase 2 schema-compatible metadata only from passing evidence."""
    root, target, allowed_root = (
        Path(dataset_root).resolve(),
        Path(metadata_path).resolve(),
        Path(report_root).resolve(),
    )
    if not target.is_relative_to(allowed_root) or target.is_relative_to(root):
        raise ValueError("Metadata v2 must be a new file under the configured report root")
    with resource_lock(target, timeout_seconds=0):
        report = _validate_passing_report(root, report_path, allowed_root)
        marker = report["criteria"]["download_completion_marker"]["details"]["marker_claims"]
        metadata = {
            "schema_version": RAW_METADATA_SCHEMA_VERSION,
            "dataset": "food101",
            "source_url": marker.get("source_url"),
            "archive_url": marker.get("archive_url"),
            "official_sha256": marker.get("official_sha256"),
            "download_sha256": marker["download_sha256"],
            "official_checksum_verified": marker.get("official_checksum_verified", False),
            "image_count": EXPECTED_IMAGES,
            "expected_file_count": EXPECTED_IMAGES,
            "verified_file_count": report["verified_file_count"],
            "class_count": report["class_count"],
            "inventory_sha256": report["inventory_sha256"],
            "source_fingerprint": report["source_fingerprint"],
            "filesystem_fingerprint": report["filesystem_fingerprint"],
            "metadata_sha256": report["metadata_sha256"],
            "integrity_status": "PASS",
            "verification_state": "PASS",
            "verification_policy": report["verification_policy"],
            "duplicate_cross_split_hash_count": report["train_test_duplicate_hashes"],
            "verified_at_utc": report["verified_at_utc"],
            "complete": True,
        }
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FileExistsError(
                    "Existing metadata v2 is corrupt; refusing to replace it"
                ) from exc
            existing_fingerprint = dict(existing) if isinstance(existing, dict) else {}
            expected_fingerprint = dict(metadata)
            existing_fingerprint.pop("verified_at_utc", None)
            expected_fingerprint.pop("verified_at_utc", None)
            if existing_fingerprint == expected_fingerprint:
                return {
                    "state": "SKIPPED",
                    "path": str(target),
                    "source_fingerprint": report["source_fingerprint"],
                }
            raise FileExistsError(
                "Existing metadata v2 has different evidence; refusing to replace it"
            )
        atomic_write_json(target, metadata)
        return {
            "state": "PASS",
            "path": str(target),
            "source_fingerprint": report["source_fingerprint"],
        }


def verify_metadata_v2(
    dataset_root: Path,
    metadata_path: Path,
    report_dir: Path,
    reject_cross_split_duplicates: bool = True,
) -> dict[str, Any]:
    """Run the current full verifier against an external Phase 2 schema metadata file."""
    return audit_food101(
        dataset_root,
        report_dir,
        mode="full",
        reject_cross_split_duplicates=reject_cross_split_duplicates,
        metadata_path=metadata_path,
    )


def _write_reports(report_dir: Path, result: dict[str, Any]) -> None:
    """Persist JSON and human-readable Markdown reports without touching source data."""
    _save_json_atomic(report_dir / "food101_report.json", result)
    lines = [
        "# Food-101 verification report",
        "",
        f"- Overall: **{result.get('overall', 'FAIL')}**",
        f"- Mode: {result.get('mode', 'unknown')}",
        f"- Root: `{result.get('dataset_root', '')}`",
        f"- Images on disk: {result.get('actual_image_count', 0)}",
        f"- Classes: {result.get('class_count', 0)}",
        f"- Train/test entries: {result.get('train_count', 0)} / {result.get('test_count', 0)}",
        f"- Valid / bad images: {result.get('valid_images', 'not checked')} / "
        f"{result.get('bad_image_count', 'not checked')}",
        f"- Missing / unlisted images: {result.get('missing_image_count', 0)} / "
        f"{result.get('unlisted_image_count', 0)}",
        f"- Duplicate content groups: {result.get('duplicate_content_groups', 'not checked')}",
        f"- Train/test duplicate hashes: "
        f"{result.get('train_test_duplicate_hashes', 'not checked')}",
        f"- DataLoader status: {result.get('data_loader', {}).get('status', 'not run')}",
        "",
        "## Warnings",
        "",
    ]
    for item in result.get("warnings", []):
        lines.append(
            f"- {item['message']} ({item['duplicate_content_groups']} duplicate groups; "
            f"{item['train_test_duplicate_hashes']} cross-split hashes)."
        )
    if not result.get("warnings"):
        lines.append("None reported.")
    lines.extend(
        [
            "",
            "## Criteria",
            "",
            "| Criterion | Status | Details |",
            "|---|---|---|",
        ]
    )
    for name, value in result.get("criteria", {}).items():
        lines.append(
            f"| {name} | {value['status']} | "
            f"`{json.dumps(value['details'], ensure_ascii=False)[:1000]}` |"
        )
    lines.extend(["", "## Image issues", ""])
    for item in result.get("issues", []):
        lines.append(f"- `{item['path']}`: {item.get('error', 'unknown error')}")
    if not result.get("issues"):
        lines.append("None reported.")
    from food_assistant.data_pipeline.atomic import atomic_write_text

    atomic_write_text(report_dir / "food101_report.md", "\n".join(lines) + "\n")
