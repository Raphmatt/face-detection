# test_service.py
import os

import cv2
import mediapipe as mp
import pytest
import numpy as np

from src.service import detect_face_count, calculate_face_rotation, get_background_mask

current_dir = os.path.dirname(os.path.abspath(__file__))
testdata_path = os.path.join(current_dir, "testdata")

def dataPath(filename: str) -> str:
    return os.path.join(testdata_path, filename)

testdata_detect_face_count = [
    ("one_face.jpg", 1),
    ("two_faces.jpg", 2),
    ("no_face.jpg", 0),
    ("rotated_face_1.jpg", 1),
]


@pytest.mark.parametrize("file, expected_face_count", testdata_detect_face_count)
def test_detect_face_count_returns_count(file: str, expected_face_count: int):
    filepath = dataPath(file)
    # image = mp.Image.create_from_file(filepath)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB))
    face_count = detect_face_count(image)
    assert face_count == expected_face_count


testdata_detect_face_rotation = [
    ("one_face.jpg", 2),
    ("rotated_face_1.jpg", -17),
    ("rotated_face_2.jpg", -27),
    ("rotated_face_3.jpg", -10),
    ("rotated_face_4.jpg", 2),
    ("rotated_face_5.jpg", 46),
    ("rotated_face_6.jpg", 4),
]


@pytest.mark.parametrize("file, expected_face_rotation", testdata_detect_face_rotation)
def test_get_face_angle(file: str, expected_face_rotation: float):
    filepath = dataPath(file)
    image = cv2.imread(filepath)
    face_rotation = calculate_face_rotation(image)
    assert face_rotation == pytest.approx(expected_face_rotation, abs=1)


testdata_get_background_mask = [
    ("one_face.jpg", "one_face.mask.jpg"),
    ("rotated_face_1.jpg", "rotated_face_1.mask.jpg"),
    ("rotated_face_4.jpg", "rotated_face_4.mask.jpg"),
    ("rotated_face_6.jpg", "rotated_face_6.mask.jpg"),
    ("three_faces.jpg",
     "three_faces.mask.jpg")
]


@pytest.mark.parametrize("original_file, expected_mask_file", testdata_get_background_mask)
def test_get_background_mask(original_file: str, expected_mask_file: str):
    original_img_path = dataPath(original_file)
    expected_mask_path = dataPath(expected_mask_file)

    # Read the original image and expected mask
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(cv2.imread(original_img_path), cv2.COLOR_BGR2RGB))
    expected_mask = cv2.imread(expected_mask_path, cv2.IMREAD_GRAYSCALE)

    # Run the method to generate the mask of the original image
    generated_mask = get_background_mask(image)

    # Ensure the generated mask is in grayscale (if it isn't already)
    if len(generated_mask.shape) == 3:  # Checks if image has 3 dimensions
        generated_mask = cv2.cvtColor(generated_mask, cv2.COLOR_BGR2GRAY)

    # Now, both masks should be two-dimensional, and you can calculate the percentage difference
    difference = np.sum(np.abs(generated_mask.astype("float") - expected_mask.astype("float")))
    total = np.sum(expected_mask.astype("float"))
    percentage_diff = (difference / total) * 100

    print(f"Percentage difference: {percentage_diff}")

    # Assert if the difference is more than 20%
    assert percentage_diff <= 5
