from io import BytesIO

import pytest
from PIL import Image

from food_assistant.domain.errors import AppError
from food_assistant.services.image_service import ImageService


def test_decode(image_bytes):
    assert ImageService().validate_and_decode(image_bytes, "image/png").mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not a png", b"\x89PNG\r\n"])
def test_reject_invalid(data):
    with pytest.raises(AppError, match="Không giải mã"):
        ImageService().validate_and_decode(data)


def test_limits(image_bytes):
    for service in [ImageService(max_bytes=1), ImageService(max_pixels=4)]:
        with pytest.raises(AppError) as error:
            service.validate_and_decode(image_bytes)
        assert error.value.code == "IMAGE_TOO_LARGE"


def test_exif():
    image = Image.new("RGB", (32, 16))
    exif = image.getexif()
    exif[274] = 6
    data = BytesIO()
    image.save(data, "JPEG", exif=exif)
    assert ImageService().validate_and_decode(data.getvalue()).size == (16, 32)


def test_mime_mismatch(image_bytes):
    with pytest.raises(AppError):
        ImageService().validate_and_decode(image_bytes, "image/jpeg")
