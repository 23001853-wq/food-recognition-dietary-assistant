"""Decode untrusted JPEG/PNG bytes and apply normalized manual crops."""

import io
import math
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from food_assistant.domain.errors import AppError


class ImageService:
    def __init__(self, max_bytes=10 * 1024 * 1024, max_pixels=20_000_000):
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels

    def validate_and_decode(self, data: bytes, content_type: str | None = None):
        if len(data) > self.max_bytes:
            raise AppError("IMAGE_TOO_LARGE", "Ảnh vượt giới hạn dung lượng", 413)
        if content_type not in {None, "image/jpeg", "image/png"}:
            raise AppError("INVALID_IMAGE", "Chỉ chấp nhận JPEG hoặc PNG", 415)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in {"JPEG", "PNG"}:
                        raise ValueError("Unsupported image format")
                    if (
                        content_type
                        and content_type != {"JPEG": "image/jpeg", "PNG": "image/png"}[image.format]
                    ):
                        raise ValueError("Content type does not match decoded image")
                    if image.width * image.height > self.max_pixels:
                        raise AppError("IMAGE_TOO_LARGE", "Ảnh vượt giới hạn pixel", 413)
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    return ImageOps.exif_transpose(image).convert("RGB")
        except AppError:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise AppError("INVALID_IMAGE", "Không giải mã được ảnh JPEG/PNG hợp lệ", 422) from exc


class CropService:
    @staticmethod
    def crop(image, box):
        if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box):
            raise AppError("INVALID_CROP", "Tọa độ crop phải thuộc [0, 1]")
        x1, y1, x2, y2 = box
        pixels = tuple(round(v * size) for v, size in zip(box, [image.width, image.height] * 2))
        if x1 >= x2 or y1 >= y2 or pixels[2] - pixels[0] < 8 or pixels[3] - pixels[1] < 8:
            raise AppError("INVALID_CROP", "Vùng crop phải có kích thước tối thiểu 8×8 pixel")
        return image.crop(pixels)
