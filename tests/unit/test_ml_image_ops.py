"""Synthetic contract tests for opt-in image processing primitives."""

import numpy as np
import pytest
from PIL import Image

from food_assistant.ml import image_ops as ops


def test_color_layout_and_numeric_contracts():
    """Verify explicit channel/layout conversions and safe float scaling."""
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    assert ops.bgr_to_rgb(ops.rgb_to_bgr(rgb))[0, 0].tolist() == [255, 0, 0]
    assert ops.chw_to_hwc(ops.hwc_to_chw(rgb)).shape == rgb.shape
    assert ops.to_float01(rgb).dtype == np.float32
    assert ops.to_float01(rgb).max() == 1.0
    with pytest.raises(ValueError):
        ops.to_float01(np.full((2, 2), 2.0, dtype=np.float32))


def test_pil_grayscale_rgba_and_exif_conversion():
    """Normalize common PIL image modes into RGB without losing orientation semantics."""
    gray = Image.fromarray(np.full((7, 8), 80, dtype=np.uint8), mode="L")
    rgba = Image.new("RGBA", (8, 7), (20, 40, 60, 100))
    assert ops.to_rgb_array(gray).shape == (7, 8, 3)
    assert ops.to_rgb_array(rgba).shape == (7, 8, 3)
    assert ops.rgb_to_grayscale(gray).shape == (7, 8)
    assert ops.to_rgb_array(gray).dtype == np.uint8


def test_hsv_roundtrip_and_exif_orientation():
    """Check RGB/HSV round-trip tolerance and apply EXIF orientation for PIL inputs."""
    rgb = np.random.default_rng(8).integers(0, 256, (31, 25, 3), dtype=np.uint8)
    roundtrip = ops.hsv_to_rgb(ops.rgb_to_hsv(rgb))
    assert np.abs(roundtrip.astype(np.int16) - rgb.astype(np.int16)).max() <= 4
    oriented = Image.new("RGB", (32, 16), (200, 20, 15))
    oriented.getexif()[274] = 6
    assert ops.to_rgb_array(oriented).shape == (32, 16, 3)


def test_pixel_geometry_and_normalization_contracts():
    """Check clipping, photometric range preservation and deterministic geometry."""
    pixels = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    assert ops.clip_pixels(pixels.astype(np.int16) - 20, (0, 255)).min() == 0
    assert ops.adjust_brightness(pixels, 1.4).max() == 255
    assert ops.adjust_contrast(pixels, 0.8).dtype == np.uint8
    assert ops.gamma_correct(pixels, 0.7).dtype == np.uint8
    normalized = ops.normalize_channels(
        ops.to_float01(pixels), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    with pytest.raises(ValueError):
        ops.normalize_channels(normalized, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert ops.resize_image(pixels, (6, 5)).shape == (6, 5, 3)
    assert ops.center_crop(pixels, (6, 4)).shape == (6, 4, 3)
    assert ops.horizontal_flip(pixels).shape == pixels.shape
    with pytest.raises(ValueError):
        ops.rotate_image(pixels, 5)
    assert ops.rotate_image(pixels, 5, enabled=True).shape == pixels.shape
    with pytest.raises(ValueError):
        ops.gamma_correct(pixels, 0)


def test_noise_filters_and_edge_region_operations():
    """Check filtering reduces synthetic noise and binary/edge tools return valid masks."""
    base = np.zeros((64, 64), dtype=np.uint8)
    base[16:48, 16:48] = 180
    rng = np.random.default_rng(17)
    noisy = base.copy()
    noise_points = rng.integers(0, 64, size=(180, 2))
    noisy[noise_points[:, 0], noise_points[:, 1]] = 255
    median = ops.median_filter(noisy, 3)
    assert np.mean((median.astype(float) - base) ** 2) < np.mean((noisy.astype(float) - base) ** 2)
    assert ops.gaussian_blur(noisy, 5).shape == noisy.shape
    mask = ops.threshold_mask(base, 100)
    assert ops.erosion(mask, 3).sum() < mask.sum()
    assert ops.dilation(mask, 3).sum() > mask.sum()
    assert ops.opening(mask).shape == mask.shape
    assert ops.closing(mask).shape == mask.shape
    assert ops.sobel_magnitude(base).shape == base.shape
    assert ops.canny_edges(base, 20, 80).dtype == np.uint8
    count, labels, stats = ops.connected_components(mask)
    assert count == 1 and labels.shape == mask.shape and len(stats) == 1
    assert ops.extract_regions(mask)[0]["area"] == 32 * 32
