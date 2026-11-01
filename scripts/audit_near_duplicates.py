"""Flag cross-split dHash neighbors for manual review; never alter official test."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from food_assistant.ml.datasets import read_json, write_json


def dhash(path):
    with Image.open(path) as image:
        values = np.asarray(image.convert("L").resize((9, 8)))
    return int.from_bytes(np.packbits(values[:, 1:] > values[:, :-1]).tobytes(), "big")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--distance", type=int, default=4)
    args = parser.parse_args()
    root = Path(read_json(args.manifest_dir / "dataset_metadata.json")["root"])
    with (args.manifest_dir / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Multi-index buckets: a <=4-bit neighbor shares at least one of 8 byte segments.
    buckets, pairs = {}, []
    for index, row in enumerate(rows):
        digest = dhash(root / row["relative_path"])
        candidates = set()
        for byte in range(8):
            candidates.update(buckets.get((byte, (digest >> (8 * byte)) & 255), []))
        for prior_index, prior_hash in candidates:
            prior = rows[prior_index]
            distance = (digest ^ prior_hash).bit_count()
            if row["split"] != prior["split"] and distance <= args.distance:
                pairs.append(
                    {"a": prior["relative_path"], "b": row["relative_path"], "distance": distance}
                )
        for byte in range(8):
            buckets.setdefault((byte, (digest >> (8 * byte)) & 255), []).append((index, digest))
    write_json(args.manifest_dir / "near_duplicate_review.json", pairs)
    print(f"{len(pairs)} cross-split candidates require manual review; no files changed")


if __name__ == "__main__":
    main()
