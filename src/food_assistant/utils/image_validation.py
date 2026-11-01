"""Reusable validation entry point for callers outside the HTTP boundary."""

from food_assistant.services.image_service import ImageService


def validate_image(data, content_type=None, max_bytes=10 * 1024 * 1024, max_pixels=20_000_000):
    return ImageService(max_bytes, max_pixels).validate_and_decode(data, content_type)
