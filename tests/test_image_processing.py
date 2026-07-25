import numpy as np

from src.image_processing import normalize, resize


def test_resize_and_normalize() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert resize(image, 10, 8).shape == (8, 10, 3)
    assert normalize(image).dtype == np.float32

