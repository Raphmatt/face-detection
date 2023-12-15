# test_utils_convert.py
import mediapipe as mp
import numpy as np
import pytest

from utils.convert import convert_to_opencv_image, convert_to_mp_image, convert_to_rgb

def test_convert_to_opencv_image_returns_opencv_image():
    # Create a sample numpy array representing an RGB image
    sample_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    opencv_image = convert_to_opencv_image(sample_image)
    assert isinstance(opencv_image, np.ndarray)
    assert opencv_image.shape == sample_image.shape

def test_convert_to_mp_image_returns_mp_image():
    # Create a sample numpy array representing an RGB image
    sample_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    mp_image = convert_to_mp_image(sample_image)
    assert isinstance(mp_image, mp.Image)

def test_convert_to_rgb_returns_rgb_image():
    # Create a sample numpy array representing an RGBA image
    sample_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    processed_image = convert_to_rgb(sample_image)
    assert isinstance(processed_image, np.ndarray)
    assert processed_image.shape[-1] == 3  # Image should be converted to RGB (3 channels)

