"""Bounded HTTP Range downloader for the approved official archive; resumable chunks."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    existing = Path("data/raw/food-101")
    if (existing / "download_metadata.json").exists() and (existing / "meta/train.txt").exists():
        if len(list((existing / "images").glob("*/*.jpg"))) == 101000:
            print("Dataset already extracted with expected count; run verify_dataset.py for audit")
            return
    approval = json.loads(Path("configs/dataset_approval.json").read_text())
    if approval["academic_approval"] != "confirmed_by_user":
        raise SystemExit("Academic approval required")
    url = approval["archive_url"]
    with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=30) as response:
        total = int(response.headers["Content-Range"].split("/")[-1])
    chunk_size = 32 * 1024 * 1024
    folder = Path("data/raw/download_chunks")
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(index):
        start, end = index * chunk_size, min(total - 1, (index + 1) * chunk_size - 1)
        target = folder / f"{index:04d}.part"
        expected = end - start + 1
        for attempt in range(6):
            offset = target.stat().st_size if target.exists() else 0
            if offset == expected:
                return target
            try:
                with urlopen(
                    Request(url, headers={"Range": f"bytes={start + offset}-{end}"}), timeout=30
                ) as response:
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {start + offset}-"):
                        raise RuntimeError("Inconsistent range response")
                    with target.open("ab") as output:
                        while data := response.read(1024 * 1024):
                            output.write(data)
                if target.stat().st_size == expected:
                    print(
                        f"Chunk {index + 1}/{(total + chunk_size - 1) // chunk_size} complete",
                        flush=True,
                    )
                    return target
            except (OSError, TimeoutError) as exc:
                print(f"Chunk {index}: retry {attempt + 1} ({type(exc).__name__})", flush=True)
                time.sleep(min(attempt + 1, 5))
        raise RuntimeError(f"Chunk {index} incomplete; rerun to resume")

    count = (total + chunk_size - 1) // chunk_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(fetch, range(count)))
    # Preserve the original sequential partial download until a complete archive exists.
    combined = Path("data/raw/food-101.parallel.complete.tar.gz")
    import shutil

    with combined.open("wb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output)
    if combined.stat().st_size != total:
        raise RuntimeError("Combined archive length mismatch")
    from download_food101 import safe_extract

    safe_extract(combined, Path("data/raw"))
    import hashlib

    metadata = {
        **approval,
        "image_count": len(list(Path("data/raw/food-101/images").glob("*/*.jpg"))),
        "download_sha256": hashlib.file_digest(combined.open("rb"), "sha256").hexdigest(),
        "official_checksum_verified": False,
    }
    if metadata["image_count"] != 101000:
        raise RuntimeError("Unexpected Food-101 count")
    Path("data/raw/food-101/download_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Complete: 101000 images extracted", flush=True)


if __name__ == "__main__":
    main()
