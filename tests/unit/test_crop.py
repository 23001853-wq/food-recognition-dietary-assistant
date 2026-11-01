import pytest
from PIL import Image

from food_assistant.domain.errors import AppError
from food_assistant.services.image_service import CropService


def test_valid_crop():
    assert CropService.crop(Image.new("RGB", (100, 100)), [0, 0, 0.5, 0.5]).size == (50, 50)


@pytest.mark.parametrize(
    "box", [[-1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0.001, 0.001], [0, 0, float("nan"), 1], [0, 1]]
)
def test_invalid_crop(box):
    with pytest.raises(AppError):
        CropService.crop(Image.new("RGB", (100, 100)), box)
