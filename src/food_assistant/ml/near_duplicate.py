"""Small-batch, explicitly scoped perceptual image similarity utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class SimilarityPolicy:
    """Validated thresholds for research/demo perceptual similarity comparisons."""

    hash_size: int = 8
    max_hamming_distance: int = 8
    min_ssim: float = 0.90
    max_pairs: int = 10_000

    def __post_init__(self) -> None:
        """Reject invalid policy values at construction time."""
        if (
            isinstance(self.hash_size, bool)
            or not isinstance(self.hash_size, int)
            or self.hash_size < 2
        ):
            raise ValueError("hash_size must be an integer >= 2")
        if (
            isinstance(self.max_hamming_distance, bool)
            or not isinstance(self.max_hamming_distance, int)
            or not 0 <= self.max_hamming_distance <= self.hash_size * self.hash_size
        ):
            raise ValueError("max_hamming_distance is outside the hash bit range")
        if not np.isfinite(self.min_ssim) or not -1.0 <= self.min_ssim <= 1.0:
            raise ValueError("min_ssim must be finite and in [-1,1]")
        if (
            isinstance(self.max_pairs, bool)
            or not isinstance(self.max_pairs, int)
            or self.max_pairs < 1
        ):
            raise ValueError("max_pairs must be a positive integer")


def _gray_u8(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert supported image input to grayscale uint8 with explicit RGB semantics."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a PIL Image or numpy array")
    if image.dtype != np.uint8 or image.ndim not in {2, 3} or min(image.shape[:2]) < 1:
        raise ValueError("array image must be non-empty uint8 HxW or HxWxC")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.shape[2] == 1:
        return image[..., 0].copy()
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError("array image must have 1, 3, or 4 channels")


def perceptual_hash(image: Image.Image | np.ndarray, policy: SimilarityPolicy) -> str:
    """Compute a deterministic DCT pHash encoded as fixed-width lowercase hex."""
    gray = _gray_u8(image)
    side = policy.hash_size * 4
    resized = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[: policy.hash_size, : policy.hash_size].copy()
    median = float(np.median(low.reshape(-1)[1:]))
    bits = low > median
    bit_string = "".join("1" if value else "0" for value in bits.reshape(-1))
    return f"{int(bit_string, 2):0{(len(bit_string) + 3) // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    """Return bit Hamming distance for equal-width hexadecimal hashes."""
    if not first or len(first) != len(second):
        raise ValueError("hashes must have the same non-zero width")
    try:
        left = int(first, 16)
        right = int(second, 16)
    except ValueError as error:
        raise ValueError("hashes must be hexadecimal") from error
    return (left ^ right).bit_count()


def structural_similarity_score(
    first: Image.Image | np.ndarray, second: Image.Image | np.ndarray
) -> float:
    """Compute grayscale SSIM after resizing the second image to the first image size."""
    left = _gray_u8(first)
    right = _gray_u8(second)
    if min(left.shape) < 7 or min(right.shape) < 7:
        raise ValueError("SSIM requires both images to be at least 7 pixels per side")
    aligned = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    return float(structural_similarity(left, aligned, data_range=255, channel_axis=None))


def compare_pair(
    first: Image.Image | np.ndarray,
    second: Image.Image | np.ndarray,
    policy: SimilarityPolicy,
) -> dict[str, float | int | str | bool]:
    """Return pHash distance and SSIM evidence for one explicit image pair."""
    first_hash = perceptual_hash(first, policy)
    second_hash = perceptual_hash(second, policy)
    distance = hamming_distance(first_hash, second_hash)
    ssim = structural_similarity_score(first, second)
    return {
        "first_hash": first_hash,
        "second_hash": second_hash,
        "hamming_distance": distance,
        "ssim": ssim,
        "suspicious": distance <= policy.max_hamming_distance and ssim >= policy.min_ssim,
    }


def find_suspicious_pairs(
    images: Sequence[tuple[str, Image.Image | np.ndarray]], policy: SimilarityPolicy
) -> list[dict[str, object]]:
    """Compare an explicitly supplied small collection; refuse unbounded pair work."""
    pair_count = len(images) * (len(images) - 1) // 2
    if pair_count > policy.max_pairs:
        raise ValueError("pair count exceeds SimilarityPolicy.max_pairs")
    results = []
    for (first_id, first_image), (second_id, second_image) in combinations(images, 2):
        evidence = compare_pair(first_image, second_image, policy)
        if evidence["suspicious"]:
            results.append({"first_id": first_id, "second_id": second_id, **evidence})
    return results


def sha256_file(path: str | Path) -> str:
    """Hash one explicitly supplied file in chunks for demo pair evidence."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
