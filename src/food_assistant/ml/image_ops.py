"""Explicit, opt-in image processing primitives for experiments and demonstrations."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

ImageInput = Image.Image | np.ndarray
InterpolationName = Literal["nearest", "bilinear", "bicubic", "area"]

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}
BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
}


def _validate_image_array(image: ImageInput) -> np.ndarray:
    """Return a copied image array after validating dimensions and numeric values."""
    if isinstance(image, Image.Image):
        value = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    elif isinstance(image, np.ndarray):
        value = image
    else:
        raise TypeError("image must be a PIL Image or numpy array")
    if value.ndim not in {2, 3} or min(value.shape[:2]) < 1:
        raise ValueError("image must have non-empty HxW or HxWxC shape")
    if value.ndim == 3 and value.shape[2] not in {1, 3, 4}:
        raise ValueError("image channel count must be 1, 3, or 4")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("image must contain finite numeric values")
    return np.ascontiguousarray(value)


def _require_unit_float(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert supported pixels to float32 [0, 1] and record whether input was uint8."""
    value = _validate_image_array(image)
    was_uint8 = value.dtype == np.uint8
    if was_uint8:
        return value.astype(np.float32) / np.float32(255.0), True
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("pixel operations support uint8 [0,255] or float [0,1]")
    if value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("float image pixels must already be in [0,1]; clip or scale explicitly")
    return value.astype(np.float32), False


def _restore_pixel_range(value: np.ndarray, was_uint8: bool) -> np.ndarray:
    """Clip transformed pixels and restore the input image's supported dtype contract."""
    clipped = np.clip(value, 0.0, 1.0)
    if was_uint8:
        return np.rint(clipped * np.float32(255.0)).astype(np.uint8)
    return clipped.astype(np.float32)


def to_rgb_array(image: ImageInput, channel_order: str = "RGB") -> np.ndarray:
    """Convert PIL/uint8 HWC input to contiguous RGB uint8, applying EXIF orientation."""
    if channel_order not in {"RGB", "BGR"}:
        raise ValueError("channel_order must be 'RGB' or 'BGR'")
    if isinstance(image, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    value = _validate_image_array(image)
    if value.dtype != np.uint8:
        raise TypeError("array input to to_rgb_array must use uint8 [0,255]")
    if value.ndim == 2:
        return cv2.cvtColor(value, cv2.COLOR_GRAY2RGB)
    if value.shape[2] == 1:
        return np.repeat(value, 3, axis=2)
    if value.shape[2] == 4:
        conversion = cv2.COLOR_RGBA2RGB if channel_order == "RGB" else cv2.COLOR_BGRA2RGB
        return cv2.cvtColor(value, conversion)
    if channel_order == "BGR":
        return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return value.copy()


def rgb_to_bgr(image: ImageInput) -> np.ndarray:
    """Convert an RGB image to BGR uint8 for OpenCV APIs."""
    return cv2.cvtColor(to_rgb_array(image), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 HWC array to RGB uint8 without guessing channel order."""
    value = _validate_image_array(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("BGR input must be an HxWx3 uint8 array")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def rgb_to_grayscale(image: ImageInput) -> np.ndarray:
    """Convert RGB uint8 pixels to a two-dimensional grayscale uint8 image."""
    return cv2.cvtColor(to_rgb_array(image), cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(image: ImageInput) -> np.ndarray:
    """Convert RGB uint8 to OpenCV HSV uint8 (H: 0–179; S/V: 0–255)."""
    return cv2.cvtColor(to_rgb_array(image), cv2.COLOR_RGB2HSV)


def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV HSV uint8 (H: 0–179) to RGB uint8."""
    value = _validate_image_array(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("HSV input must be an HxWx3 uint8 OpenCV HSV array")
    return cv2.cvtColor(value, cv2.COLOR_HSV2RGB)


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """Reorder a three-dimensional HWC array to contiguous CHW layout."""
    value = _validate_layout_array(image, "HWC")
    return np.ascontiguousarray(np.transpose(value, (2, 0, 1)))


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """Reorder a three-dimensional CHW array to contiguous HWC layout."""
    value = _validate_layout_array(image, "CHW")
    return np.ascontiguousarray(np.transpose(value, (1, 2, 0)))


def _validate_layout_array(image: np.ndarray, layout: str) -> np.ndarray:
    """Validate a finite, non-empty three-dimensional tensor without assuming axis sizes."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or min(image.shape) < 1:
        raise ValueError(f"{layout} input must be a non-empty three-dimensional array")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError(f"{layout} input must contain finite numeric values")
    return image


def to_float01(image: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] to float32 [0,1], or validate existing float [0,1]."""
    value = _validate_image_array(image)
    if value.dtype == np.uint8:
        return value.astype(np.float32) / np.float32(255.0)
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("to_float01 accepts uint8 or floating-point images")
    if value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("floating-point image is outside [0,1]; refusing implicit rescaling")
    return value.astype(np.float32)


def clip_pixels(image: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """Clip finite numeric image values to an explicit range, preserving dtype and shape."""
    value = _validate_image_array(image)
    if len(value_range) != 2:
        raise ValueError("value_range must contain (minimum, maximum)")
    lower, upper = value_range
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("value_range must contain finite minimum < maximum")
    return np.clip(value, lower, upper).astype(value.dtype, copy=False)


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply deterministic multiplicative brightness and clip to the input's pixel range."""
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("brightness factor must be finite and non-negative")
    value, was_uint8 = _require_unit_float(image)
    return _restore_pixel_range(value * np.float32(factor), was_uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast around the per-channel spatial mean and preserve supported dtype."""
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("contrast factor must be finite and non-negative")
    value, was_uint8 = _require_unit_float(image)
    axes = (0, 1)
    mean = value.mean(axis=axes, keepdims=True)
    result = (value - mean) * np.float32(factor) + mean
    return _restore_pixel_range(result, was_uint8)


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply deterministic power-law correction; gamma below one brightens the image."""
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and greater than zero")
    value, was_uint8 = _require_unit_float(image)
    return _restore_pixel_range(np.power(value, np.float32(gamma)), was_uint8)


def normalize_channels(
    image: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]
) -> np.ndarray:
    """Normalize RGB float [0,1] HWC values once using explicit channel means/stds."""
    value = _validate_image_array(image)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("normalization requires an HxWx3 float32 RGB array")
    if value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("normalization input must be float32 in [0,1], not already normalized")
    if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(x) for x in (*mean, *std)):
        raise ValueError("mean and std must each contain three finite values")
    if any(value <= 0 for value in std):
        raise ValueError("normalization standard deviations must be positive")
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return (value - mean_array) / std_array


def resize_image(
    image: ImageInput,
    size: tuple[int, int],
    interpolation: InterpolationName = "bilinear",
) -> np.ndarray:
    """Resize an image to explicit (height,width), preserving aspect ratio only if caller opts."""
    value = _validate_image_array(image)
    if len(size) != 2 or any(
        isinstance(side, bool) or not isinstance(side, int) or side < 1 for side in size
    ):
        raise ValueError("size must contain positive (height,width) integers")
    if interpolation not in INTERPOLATIONS:
        raise ValueError("unsupported interpolation mode")
    height, width = size
    return cv2.resize(value, (width, height), interpolation=INTERPOLATIONS[interpolation])


def center_crop(image: ImageInput, size: tuple[int, int]) -> np.ndarray:
    """Take an exact (height,width) center crop; reject sizes larger than the source."""
    value = _validate_image_array(image)
    if len(size) != 2 or any(
        isinstance(side, bool) or not isinstance(side, int) or side < 1 for side in size
    ):
        raise ValueError("size must contain positive (height,width) integers")
    height, width = size
    source_height, source_width = value.shape[:2]
    if height > source_height or width > source_width:
        raise ValueError("crop size cannot exceed source dimensions")
    top = (source_height - height) // 2
    left = (source_width - width) // 2
    return value[top : top + height, left : left + width].copy()


def horizontal_flip(image: ImageInput) -> np.ndarray:
    """Return a deterministic left-right flip without changing shape or dtype."""
    return np.ascontiguousarray(np.flip(_validate_image_array(image), axis=1))


def rotate_image(
    image: ImageInput,
    degrees: float,
    *,
    enabled: bool = False,
    interpolation: InterpolationName = "bilinear",
) -> np.ndarray:
    """Rotate on the original canvas; require explicit opt-in because it is experimental."""
    if not enabled:
        raise ValueError("rotation is opt-in; pass enabled=True explicitly")
    if not math.isfinite(degrees):
        raise ValueError("rotation degrees must be finite")
    if interpolation not in INTERPOLATIONS:
        raise ValueError("unsupported interpolation mode")
    value = _validate_image_array(image)
    height, width = value.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        value,
        matrix,
        (width, height),
        flags=INTERPOLATIONS[interpolation],
        borderMode=cv2.BORDER_REFLECT_101,
    )


def gaussian_blur(
    image: ImageInput,
    kernel_size: int = 5,
    sigma: float = 1.0,
    border: str = "reflect",
) -> np.ndarray:
    """Apply an opt-in Gaussian blur with odd kernel, positive sigma and explicit border."""
    value = _validate_image_array(image)
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError("Gaussian kernel_size must be a positive odd integer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    if border not in BORDER_MODES:
        raise ValueError("unsupported border mode")
    return cv2.GaussianBlur(
        value,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=BORDER_MODES[border],
    )


def median_filter(image: ImageInput, kernel_size: int = 3) -> np.ndarray:
    """Apply an opt-in median filter to uint8 or float32 images using an odd kernel."""
    value = _validate_image_array(image)
    if isinstance(kernel_size, bool) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("median kernel_size must be an odd integer >= 3")
    if value.dtype not in {np.dtype(np.uint8), np.dtype(np.float32)}:
        raise TypeError("median filter supports uint8 and float32 arrays")
    return cv2.medianBlur(value, kernel_size)


def _binary_mask(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate a 2D binary foreground mask and return uint8 0/1 pixels."""
    if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
        value = mask
        if value.ndim != 2 or min(value.shape) < 1:
            raise ValueError("mask must be a non-empty 2D array")
    else:
        value = _validate_image_array(mask)
    if value.ndim != 2 or value.dtype not in {np.dtype(np.bool_), np.dtype(np.uint8)}:
        raise ValueError("mask must be a 2D bool or uint8 array")
    if value.dtype == np.uint8 and not np.isin(value, (0, 1, 255)).all():
        raise ValueError("uint8 mask foreground must be encoded as 1 or 255")
    return (value > 0).astype(np.uint8), value.dtype == np.bool_


def _morphology(
    mask: np.ndarray, kernel_size: int, operation: int, iterations: int = 1
) -> np.ndarray:
    """Apply a square all-ones structuring element to a 2D foreground mask."""
    value, was_bool = _binary_mask(mask)
    if isinstance(kernel_size, bool) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("morphology kernel_size must be a positive odd integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if operation == cv2.MORPH_ERODE:
        result = cv2.erode(value, kernel, iterations=iterations)
    elif operation == cv2.MORPH_DILATE:
        result = cv2.dilate(value, kernel, iterations=iterations)
    else:
        raise ValueError("unsupported morphology operation")
    return result.astype(bool) if was_bool else result


def erosion(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink foreground pixels using an all-ones square structuring element."""
    return _morphology(mask, kernel_size, cv2.MORPH_ERODE, iterations)


def dilation(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Expand foreground pixels using an all-ones square structuring element."""
    return _morphology(mask, kernel_size, cv2.MORPH_DILATE, iterations)


def opening(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Remove small foreground components by erosion followed by dilation."""
    value, was_bool = _binary_mask(mask)
    if isinstance(kernel_size, bool) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("morphology kernel_size must be a positive odd integer")
    if isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    result = cv2.morphologyEx(value, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return result.astype(bool) if was_bool else result


def closing(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Fill small foreground gaps by dilation followed by erosion."""
    value, was_bool = _binary_mask(mask)
    if isinstance(kernel_size, bool) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("morphology kernel_size must be a positive odd integer")
    if isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    result = cv2.morphologyEx(value, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return result.astype(bool) if was_bool else result


def threshold_mask(image: ImageInput, threshold: int) -> np.ndarray:
    """Threshold grayscale values into a uint8 0/1 mask using strict greater-than."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or threshold < 0
        or threshold > 255
    ):
        raise ValueError("threshold must be an integer in [0,255]")
    gray = rgb_to_grayscale(image)
    return (gray > threshold).astype(np.uint8)


def sobel_magnitude(image: ImageInput, kernel_size: int = 3) -> np.ndarray:
    """Return Sobel gradient magnitude as float32 for a uint8 grayscale/RGB image."""
    if kernel_size not in {1, 3, 5, 7}:
        raise ValueError("Sobel kernel_size must be 1, 3, 5, or 7")
    if isinstance(image, np.ndarray) and image.ndim == 2:
        gray = image
    else:
        gray = rgb_to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
    return cv2.magnitude(grad_x, grad_y)


def canny_edges(image: ImageInput, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Return uint8 Canny edges after validating ordered thresholds in [0,255]."""
    if (
        isinstance(low_threshold, bool)
        or isinstance(high_threshold, bool)
        or not 0 <= low_threshold < high_threshold <= 255
    ):
        raise ValueError("Canny thresholds require 0 <= low < high <= 255")
    if isinstance(image, np.ndarray) and image.ndim == 2:
        gray = image
    else:
        gray = rgb_to_grayscale(image)
    if gray.dtype != np.uint8:
        raise TypeError("Canny input must be uint8")
    return cv2.Canny(gray, low_threshold, high_threshold)


def connected_components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Label 4-connected components in a binary mask; return foreground count, labels, stats."""
    value, _ = _binary_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(value, connectivity=4)
    return count - 1, labels, stats[1:]


def extract_regions(mask: np.ndarray, minimum_area: int = 1) -> list[dict[str, int]]:
    """Return bounding boxes and areas for foreground components meeting minimum_area."""
    if isinstance(minimum_area, bool) or minimum_area < 1:
        raise ValueError("minimum_area must be a positive integer")
    _, _, stats = connected_components(mask)
    regions = []
    for component_id, stat in enumerate(stats, start=1):
        x, y, width, height, area = (int(part) for part in stat)
        if area >= minimum_area:
            regions.append(
                {
                    "component_id": component_id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": area,
                }
            )
    return regions
