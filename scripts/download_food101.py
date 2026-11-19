"""Resume the official archive; safe extraction and academic approval are mandatory."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from food_assistant.data_pipeline.atomic import atomic_write_json
from food_assistant.data_pipeline.config import load_pipeline_config
from food_assistant.data_pipeline.extract import (
    RetryPolicy,
    download_parallel,
    download_sequential,
)
from food_assistant.data_pipeline.load import extract_archive_staged
from food_assistant.data_pipeline.splits import _inventory, _read_report
from food_assistant.ml.food101_verifier import (
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    EXPECTED_TEST_PER_CLASS,
    EXPECTED_TRAIN_PER_CLASS,
)


def download(url, target, **options):
    """Download and verify an archive through the shared safe Extract implementation."""
    return download_sequential(url, target, **options)


def verify_existing_raw(raw_root: Path, report_path: Path | None = None) -> dict[str, Any]:
    """Verify an existing raw tree against its Phase 2.5 checkpoint without writes."""
    raw_root = Path(raw_root).resolve()
    evidence_path = Path(
        report_path or "data/reports/elt/v2/phase_2_5_verification_retry/food101_report.json"
    ).resolve()
    baseline = _read_report(evidence_path)
    hashes, inventory_sha256, source_fingerprint = _inventory(
        raw_root,
        baseline,
        evidence_path.parent / "food101_checkpoint.json",
    )
    return {
        "state": "SKIPPED",
        "reason": "existing raw matches full Phase 2.5 source/inventory evidence",
        "image_count": len(hashes),
        "inventory_sha256": inventory_sha256,
        "source_fingerprint": source_fingerprint,
    }


def safe_extract(archive, output, metadata=None):
    """Safely stage the official top-level food-101 directory and publish it once."""
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset") != "food101"
        or not isinstance(metadata.get("download_sha256"), str)
        or len(metadata["download_sha256"]) != 64
    ):
        raise ValueError("Validated source metadata is required before raw publication")
    output = Path(output)
    dataset = output / "food-101"

    def validate(staging):
        """Check the archive's required Food-101 structure before publication."""
        root = staging
        image_paths = sorted((root / "images").glob("*/*.jpg"))
        image_files = [path for path in (root / "images").rglob("*") if path.is_file()]
        image_count = len(image_paths)
        class_names = (root / "meta" / "classes.txt").read_text(encoding="utf-8").splitlines()
        train_names = (root / "meta" / "train.txt").read_text(encoding="utf-8").splitlines()
        test_names = (root / "meta" / "test.txt").read_text(encoding="utf-8").splitlines()
        train_counts = Counter(name.split("/", 1)[0] for name in train_names)
        test_counts = Counter(name.split("/", 1)[0] for name in test_names)
        class_directories = sorted(
            path.name for path in (root / "images").iterdir() if path.is_dir()
        )
        valid_names = all(
            len(name.split("/")) == 2
            and name.split("/", 1)[0] in class_names
            and "\\" not in name
            and ":" not in name
            and all(part not in {"", ".", ".."} for part in name.split("/"))
            for name in train_names + test_names
        )
        expected_paths = {f"images/{name}.jpg" for name in set(train_names) | set(test_names)}
        actual_paths = {path.relative_to(root).as_posix() for path in image_paths}
        inventory = hashlib.sha256()
        integrity_valid = True
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                image_hash = hashlib.sha256()
                with image_path.open("rb") as image_file:
                    for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                        image_hash.update(chunk)
                image_digest = image_hash.hexdigest()
                inventory.update(
                    f"{image_path.relative_to(root).as_posix()}:{image_digest}\n".encode()
                )
            except (OSError, UnidentifiedImageError):
                integrity_valid = False
                break
        valid = (
            all(
                (root / "meta" / name).is_file()
                for name in ("classes.txt", "train.txt", "test.txt")
            )
            and (root / "images").is_dir()
            and (root / "license_agreement.txt").is_file()
            and len(class_names) == EXPECTED_CLASSES
            and len(set(class_names)) == EXPECTED_CLASSES
            and class_directories == sorted(class_names)
            and len(train_names) == EXPECTED_CLASSES * EXPECTED_TRAIN_PER_CLASS
            and len(test_names) == EXPECTED_CLASSES * EXPECTED_TEST_PER_CLASS
            and valid_names
            and len(set(train_names)) == len(train_names)
            and len(set(test_names)) == len(test_names)
            and not (set(train_names) & set(test_names))
            and actual_paths == expected_paths
            and len(image_files) == image_count
            and all(train_counts[name] == EXPECTED_TRAIN_PER_CLASS for name in class_names)
            and all(test_counts[name] == EXPECTED_TEST_PER_CLASS for name in class_names)
            and image_count == EXPECTED_IMAGES
            and integrity_valid
        )
        if valid and metadata is not None:
            atomic_write_json(
                root / "download_metadata.json",
                {
                    **metadata,
                    "schema_version": 1,
                    "image_count": image_count,
                    "inventory_sha256": inventory.hexdigest(),
                    "integrity_status": "PASS",
                    "complete": True,
                },
            )
        return valid

    return extract_archive_staged(archive, dataset, validate, strip_prefix="food-101")


def _load_approval(path: Path) -> dict[str, Any]:
    """Read and validate the configured academic source approval record."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read dataset approval: {type(error).__name__}") from None
    if (
        not isinstance(value, dict)
        or value.get("academic_approval") != "confirmed_by_user"
        or not isinstance(value.get("archive_url"), str)
        or not value["archive_url"].startswith("https://")
    ):
        raise PermissionError("BLOCKED: valid academic approval and HTTPS archive URL required")
    official_sha256 = value.get("official_sha256")
    if official_sha256 is not None and (
        not isinstance(official_sha256, str)
        or len(official_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in official_sha256)
    ):
        raise ValueError("approval official_sha256 must be null or a 64-digit SHA-256")
    return value


def run_download_cli(mode: str = "sequential", argv: list[str] | None = None) -> dict[str, Any]:
    """Run the compatible downloader CLI with explicit policy and academic approval gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/food101_elt.yaml"))
    parser.add_argument(
        "--download", action="store_true", help="Explicitly authorize network access"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print a plan without locks or writes"
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config, Path.cwd())
    output_root = Path(args.output or config.raw_root.parent).resolve()
    if (output_root / "food-101").resolve() != config.raw_root.resolve():
        parser.error("--output must publish exactly to the configured raw_root/food-101")
    if args.dry_run:
        if args.download:
            raise ValueError("--dry-run cannot be combined with --download")
        return {
            "state": "INCOMPLETE",
            "mode": mode,
            "raw_root": str(config.raw_root),
            "archive_target": str(config.archive),
            "download_policy": config.download_policy,
            "network_requested": False,
            "writes_performed": False,
        }
    if config.raw_root.exists():
        result = verify_existing_raw(config.raw_root)
        return result
    if not args.download:
        raise PermissionError("BLOCKED: pass --download after reviewing this extraction plan")
    if config.download_policy != "approved_only":
        raise PermissionError("BLOCKED: ELT config download.policy is disabled")
    approval = _load_approval(args.approval or config.approval_file)
    archive_target = config.archive.resolve()
    if archive_target.parent != output_root:
        raise ValueError("Configured archive must be a sibling of the raw dataset directory")
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    retry_policy = RetryPolicy(
        max_attempts=config.retries + 1,
        base_delay_seconds=config.backoff_seconds,
    )
    download_function = download_sequential
    if mode == "parallel":
        download_function = download_parallel
    elif mode != "sequential":
        raise ValueError(f"Unsupported downloader mode: {mode}")
    result = download_function(
        approval["archive_url"],
        archive_target,
        approval=approval,
        official_sha256=approval.get("official_sha256"),
        retry_policy=retry_policy,
        lock_timeout_seconds=config.lock_timeout_seconds,
        **({"workers": args.workers} if mode == "parallel" else {}),
    )
    archive_metadata = {
        "dataset": "food101",
        "source_url": approval.get("source_url"),
        "academic_approval": approval["academic_approval"],
        "academic_approval_date": approval.get("academic_approval_date"),
        "citation": approval.get("citation"),
        "use": approval.get("use"),
        "rights_status": approval.get("rights_status"),
        "download_sha256": result["local_sha256"],
        "local_sha256": result["local_sha256"],
        "official_sha256": approval.get("official_sha256"),
        "official_checksum_verified": result["official_checksum_verified"],
    }
    safe_extract(archive_target, output_root, archive_metadata)
    return {**result, "raw_root": str(config.raw_root), "state": "PASS"}


def main() -> None:
    """Run sequential Extract without downloading unless config and CLI both allow it."""
    result = run_download_cli("sequential")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
