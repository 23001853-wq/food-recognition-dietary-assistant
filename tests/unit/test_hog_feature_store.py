"""Synthetic memmap feature-store tests; no Food-101 data is used."""

import numpy as np
import pytest

from food_assistant.ml.hog_feature_store import build_feature_store, open_feature_store


class SyntheticDataset:
    """Small ordered dataset stub exposing the interface consumed by the store."""

    def __init__(self, size=5):
        self.rows = [{"image_id": f"id-{index}"} for index in range(size)]
        self.labels = {"dataset_version": "fixture", "labels": []}

    def __len__(self):
        """Return synthetic row count."""
        return len(self.rows)

    def __getitem__(self, index):
        """Return row-keyed placeholder image and aligned label."""
        return np.full((2, 2, 3), index, dtype=np.uint8), index % 2


def test_store_batches_rows_and_reopens_as_memmap(tmp_path, monkeypatch):
    """Write complete fixed-dimension rows and preserve input ordering."""
    import food_assistant.ml.hog_feature_store as store

    calls = []

    def fake_hog(image, config):
        calls.append(int(image[0, 0, 0]))
        return np.full((8100,), float(image[0, 0, 0]), dtype=np.float32)

    monkeypatch.setattr(store, "extract_hog", fake_hog)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "dataset_manifest.csv").write_text("image_id\n", encoding="utf-8")
    output = build_feature_store(
        SyntheticDataset(), {"hog": {"orientations": 9}}, tmp_path / "store",
        manifest_dir=manifest_dir, split="train", batch_size=2,
    )
    features, labels = open_feature_store(output)
    assert isinstance(features, np.memmap)
    assert features.shape == (5, 8100)
    assert features.dtype == np.float32
    assert labels.tolist() == [0, 1, 0, 1, 0]
    assert features[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert calls == [0, 0, 1, 2, 3, 4]


def test_incomplete_extraction_cleans_staging_and_allows_retry(tmp_path, monkeypatch):
    """Never publish a completion marker after interrupted feature extraction."""
    import food_assistant.ml.hog_feature_store as store

    count = 0

    def interrupted(image, config):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("synthetic extraction interruption")
        return np.zeros((8100,), dtype=np.float32)

    monkeypatch.setattr(store, "extract_hog", interrupted)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "dataset_manifest.csv").write_text("image_id\n", encoding="utf-8")
    target = tmp_path / "store"
    with pytest.raises(RuntimeError, match="interruption"):
        build_feature_store(
            SyntheticDataset(), {"hog": {}}, target, manifest_dir=manifest_dir, split="train"
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".store.*.pending"))


def test_store_rejects_duplicate_ids_and_corrupted_reuse(tmp_path, monkeypatch):
    """Reject ambiguous row alignment and checksum-invalid cache reuse."""
    import food_assistant.ml.hog_feature_store as store

    monkeypatch.setattr(
        store,
        "extract_hog",
        lambda image, config: np.full((8100,), image[0, 0, 0], dtype=np.float32),
    )
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "dataset_manifest.csv").write_text("image_id\n", encoding="utf-8")
    duplicate = SyntheticDataset(2)
    duplicate.rows[1]["image_id"] = duplicate.rows[0]["image_id"]
    with pytest.raises(ValueError, match="duplicate image IDs"):
        build_feature_store(
            duplicate, {"hog": {}}, tmp_path / "duplicate",
            manifest_dir=manifest_dir, split="train",
        )
    target = build_feature_store(
        SyntheticDataset(2), {"hog": {}}, tmp_path / "store",
        manifest_dir=manifest_dir, split="train",
    )
    with (target / "features.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        build_feature_store(
            SyntheticDataset(2), {"hog": {}}, target,
            manifest_dir=manifest_dir, split="train",
        )


def test_store_cleans_only_its_own_stale_staging_directory(tmp_path, monkeypatch):
    """Recover an interrupted prior staging attempt under the store lock."""
    import food_assistant.ml.hog_feature_store as store

    monkeypatch.setattr(
        store,
        "extract_hog",
        lambda image, config: np.full((8100,), image[0, 0, 0], dtype=np.float32),
    )
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "dataset_manifest.csv").write_text("image_id\n", encoding="utf-8")
    stale = tmp_path / ".features.crash.pending"
    stale.mkdir()
    (stale / "partial.npy").write_bytes(b"incomplete")
    target = build_feature_store(
        SyntheticDataset(2), {"hog": {}}, tmp_path / "features",
        manifest_dir=manifest_dir, split="train",
    )
    assert target.is_dir()
    assert not stale.exists()
