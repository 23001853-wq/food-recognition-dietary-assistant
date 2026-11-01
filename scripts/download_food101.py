"""Resume the official archive; safe extraction and academic approval are mandatory."""

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


def download(url, target):
    offset = target.stat().st_size if target.exists() else 0
    request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Server returned an inconsistent resume range")
        total = int(response.headers.get("Content-Length", 0)) + (offset if append else 0)
        done = offset if append else 0
        last = time.monotonic()
        with target.open("ab" if append else "wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"Downloaded {done / 2**20:.0f}/{total / 2**20:.0f} MiB", flush=True)
                    last = time.monotonic()
        if total and done != total:
            raise RuntimeError("Incomplete download; rerun to resume")


def safe_extract(archive, output):
    resolved = output.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            destination = (resolved / member.name).resolve()
            if not destination.is_relative_to(resolved) or member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError("Unsupported archive member type")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--approval", type=Path, default=Path("configs/dataset_approval.json"))
    args = parser.parse_args()
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    if approval["academic_approval"] != "confirmed_by_user":
        parser.error("BLOCKED: academic approval is not confirmed")
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "food-101/download_metadata.json"
    if marker.exists():
        json.loads(marker.read_text(encoding="utf-8"))
        root = marker.parent
        if (root / "meta/train.txt").exists() and len(
            list((root / "images").glob("*/*.jpg"))
        ) == 101000:
            print("Existing dataset has expected count; run verify_dataset.py for full audit")
            return
    archive = args.output / "food-101.tar.gz.part"
    download(approval["archive_url"], archive)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    if approval.get("official_sha256") and digest != approval["official_sha256"]:
        raise RuntimeError("Official checksum mismatch")
    safe_extract(archive, args.output)
    images = list((args.output / "food-101/images").glob("*/*.jpg"))
    if len(images) != 101000:
        raise RuntimeError(f"Unexpected image count: {len(images)}")
    marker.write_text(
        json.dumps(
            {
                **approval,
                "download_sha256": digest,
                "image_count": len(images),
                "official_checksum_verified": bool(approval.get("official_sha256")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Downloaded and extracted {len(images)} images; local SHA256 {digest}")


if __name__ == "__main__":
    main()
