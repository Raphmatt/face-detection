# test_service.py
import os

import cv2
import mediapipe as mp
import pytest

from src.service import detect_face_count, calculate_face_rotation

current_dir = os.path.dirname(os.path.abspath(__file__))
testdata_path = os.path.join(current_dir, "testdata")

testdata_detect_face_count = [
    (os.path.join(testdata_path, "one_face.jpg"), 1),
    (os.path.join(testdata_path, "two_faces.jpg"), 2),
    (os.path.join(testdata_path, "no_face.jpg"), 0),
    (os.path.join(testdata_path, "rotated_face_1.jpg"), 1),
]


@pytest.mark.parametrize("filepath, expected_face_count", testdata_detect_face_count)
def test_detect_face_count_returns_count(filepath: str, expected_face_count: int):
    # image = mp.Image.create_from_file(filepath)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB))
    face_count = detect_face_count(image)
    assert face_count == expected_face_count


testdata_detect_face_rotation = [
    (os.path.join(testdata_path, "one_face.jpg"), 2),
    (os.path.join(testdata_path, "rotated_face_1.jpg"), -17),
    (os.path.join(testdata_path, "rotated_face_2.jpg"), -27),
    (os.path.join(testdata_path, "rotated_face_3.jpg"), -10),
    (os.path.join(testdata_path, "rotated_face_4.jpg"), 2),
    (os.path.join(testdata_path, "rotated_face_5.jpg"), 46),
    (os.path.join(testdata_path, "rotated_face_6.jpg"), 4),
]


@pytest.mark.parametrize("filepath, expected_face_rotation", testdata_detect_face_rotation)
def test_get_face_angle(filepath: str, expected_face_rotation: float):
    image = cv2.imread(filepath)
    face_rotation = calculate_face_rotation(image)
    assert face_rotation == pytest.approx(expected_face_rotation, abs=1)
