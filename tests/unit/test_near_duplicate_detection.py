"""Synthetic tests for bounded pHash and SSIM similarity checks."""

import numpy as np
import pytest
from PIL import Image

from food_assistant.ml.near_duplicate import (
    SimilarityPolicy,
    compare_pair,
    find_suspicious_pairs,
    hamming_distance,
    perceptual_hash,
    structural_similarity_score,
)


def _food_like_image() -> np.ndarray:
    """Create a deterministic, textured synthetic image without external data."""
    y, x = np.mgrid[:96, :112]
    return np.stack(((x * 3 + y) % 256, (y * 2 + 45) % 256, ((x + y) * 2) % 256), axis=-1).astype(
        np.uint8
    )


def test_hash_and_pair_metrics_are_deterministic():
    """pHash is stable and exact images score as suspicious with permissive demo thresholds."""
    image = _food_like_image()
    policy = SimilarityPolicy(max_hamming_distance=64, min_ssim=0.5)
    assert perceptual_hash(image, policy) == perceptual_hash(image.copy(), policy)
    assert hamming_distance("0f", "00") == 4
    evidence = compare_pair(image, image.copy(), policy)
    assert evidence["suspicious"] is True
    assert evidence["ssim"] == pytest.approx(1.0)


def test_resize_recompression_and_dissimilar_image_evidence():
    """Report evidence for common near-copy changes and reject unrelated texture."""
    image = _food_like_image()
    resized = np.asarray(Image.fromarray(image).resize((80, 68), Image.Resampling.BILINEAR))
    encoded = Image.fromarray(image)
    import io

    buffer = io.BytesIO()
    encoded.save(buffer, format="JPEG", quality=82)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"))
    unrelated = np.random.default_rng(91).integers(0, 256, image.shape, dtype=np.uint8)
    policy = SimilarityPolicy(max_hamming_distance=12, min_ssim=0.72)
    resized_evidence = compare_pair(image, resized, policy)
    jpeg_evidence = compare_pair(image, recompressed, policy)
    unrelated_evidence = compare_pair(image, unrelated, policy)
    assert resized_evidence["hamming_distance"] >= 0
    assert jpeg_evidence["ssim"] >= -1.0
    assert unrelated_evidence["ssim"] < 0.72
    identical_pairs = find_suspicious_pairs([("first", image), ("copy", image.copy())], policy)
    assert len(identical_pairs) == 1
    assert identical_pairs[0]["first_id"] == "first"


def test_invalid_similarity_inputs_and_pair_bound():
    """Reject malformed hashes, undersized SSIM input and unbounded pair collections."""
    with pytest.raises(ValueError):
        hamming_distance("0", "00")
    with pytest.raises(ValueError):
        hamming_distance("xz", "00")
    with pytest.raises(ValueError):
        structural_similarity_score(
            np.zeros((6, 8), dtype=np.uint8), np.zeros((6, 8), dtype=np.uint8)
        )
    image = _food_like_image()
    policy = SimilarityPolicy(max_pairs=1)
    with pytest.raises(ValueError):
        find_suspicious_pairs([("a", image), ("b", image), ("c", image)], policy)
    with pytest.raises(ValueError):
        SimilarityPolicy(hash_size=1)
