# test_service.py

import mediapipe as mp
import pytest

from src.service import detect_face_count

testdata_detect_face_count = [
    ("testdata/one_face.jpg", 1),
    ("testdata/two_faces.jpg", 2),
    ("testdata/no_face.jpg", 0),
]


@pytest.mark.parametrize("filepath, expected_face_count", testdata_detect_face_count)
def test_detect_face_count_returns_count(filepath: str, expected_face_count: int):
    image = mp.Image.create_from_file(filepath)
    face_count = detect_face_count(image)
    assert face_count == expected_face_count
