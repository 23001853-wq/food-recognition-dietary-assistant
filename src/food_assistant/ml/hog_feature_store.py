"""Atomic disk-backed HOG feature extraction for bounded-memory preparation."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from food_assistant.data_pipeline.atomic import atomic_write_json, checkpoint_digest, resource_lock
from food_assistant.ml.hog_features import extract_hog


def build_feature_store(dataset, config, output, *, manifest_dir, split, batch_size=64):
    """Extract ordered HOG rows to immutable memmaps with a completion digest.

    Dataset iteration remains deterministic and each image is decoded only while
    its bounded batch is being written. A failed staging write is never published.
    """
    if batch_size < 1 or len(dataset) < 1:
        raise ValueError("Feature store requires positive batch size and non-empty data")
    image_ids = [row["image_id"] for row in dataset.rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Feature store input contains duplicate image IDs")
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    first, _ = dataset[0]
    feature_count = len(extract_hog(first, config))
    manifest_digest = hashlib.sha256(
        (Path(manifest_dir) / "dataset_manifest.csv").read_bytes()
    ).hexdigest()
    identity = {
        "schema_version": "1.0",
        "split": split,
        "rows": len(dataset),
        "feature_count": feature_count,
        "dtype": "float32",
        "batch_size": batch_size,
        "manifest_sha256": manifest_digest,
        "image_ids_sha256": hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest(),
        "labels_sha256": hashlib.sha256(
            json.dumps(dataset.labels, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "hog_config_sha256": checkpoint_digest(config["hog"]),
    }
    with resource_lock(target, timeout_seconds=0):
        if target.exists():
            metadata = _validate_store(target)
            if metadata.get("identity") != identity:
                raise FileExistsError("HOG feature store exists with incompatible metadata")
            return target
        for stale in target.parent.glob(f".{target.name}.*.pending"):
            if stale.is_dir() and stale.resolve().parent == target.parent:
                shutil.rmtree(stale)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.", suffix=".pending", dir=target.parent
            )
        )
        matrix = None
        labels = None
        try:
            matrix = np.lib.format.open_memmap(
                staging / "features.npy", mode="w+", dtype=np.float32,
                shape=(len(dataset), feature_count),
            )
            labels = np.lib.format.open_memmap(
                staging / "labels.npy", mode="w+", dtype=np.int64, shape=(len(dataset),)
            )
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                for index in range(start, stop):
                    image, label = dataset[index]
                    feature = extract_hog(image, config)
                    if feature.shape != (feature_count,):
                        raise ValueError("HOG feature dimension changed during extraction")
                    matrix[index] = feature
                    labels[index] = label
            matrix.flush()
            labels.flush()
            matrix._mmap.close()
            labels._mmap.close()
            matrix = None
            labels = None
            content = {
                "features_sha256": _sha256(staging / "features.npy"),
                "labels_sha256": _sha256(staging / "labels.npy"),
                "identity": identity,
                "complete": True,
            }
            atomic_write_json(staging / "metadata.json", content)
            _validate_store(staging)
            staging.rename(target)
        except Exception:
            for array in (matrix, labels):
                if array is not None and getattr(array, "_mmap", None) is not None:
                    array._mmap.close()
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return target


def open_feature_store(path):
    """Validate a completed HOG feature store and return read-only memmaps."""
    _validate_store(Path(path))
    return (
        np.load(Path(path) / "features.npy", mmap_mode="r"),
        np.load(Path(path) / "labels.npy", mmap_mode="r"),
    )


def _validate_store(path):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    identity = metadata.get("identity", {})
    if metadata.get("complete") is not True:
        raise ValueError("HOG feature store is incomplete")
    if _sha256(path / "features.npy") != metadata.get("features_sha256"):
        raise ValueError("HOG feature matrix checksum mismatch")
    if _sha256(path / "labels.npy") != metadata.get("labels_sha256"):
        raise ValueError("HOG label vector checksum mismatch")
    features = np.load(path / "features.npy", mmap_mode="r")
    labels = np.load(path / "labels.npy", mmap_mode="r")
    if features.shape != (identity.get("rows"), identity.get("feature_count")):
        raise ValueError("HOG feature matrix shape mismatch")
    if (
        features.dtype != np.float32
        or labels.dtype != np.int64
        or labels.shape != (identity["rows"],)
    ):
        raise ValueError("HOG feature store dtype/label contract mismatch")
    return metadata


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
