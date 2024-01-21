# test_utils_convert.py
import numpy as np

from src.image_utilities import rgba_2_rgb


def test_convert_to_rgb_returns_rgb_image():
    # Create a sample numpy array representing an RGBA image
    sample_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    processed_image = rgba_2_rgb(sample_image)
    assert isinstance(processed_image, np.ndarray)
    assert processed_image.shape[-1] == 3  # Image should be converted to RGB (3 channels)

