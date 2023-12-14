import io
import os

import mediapipe as mp
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils.convert import convert_to_rgb


async def read_image_file(file: UploadFile = File(...)) -> np.ndarray:
    """
    Read an uploaded image file into a numpy array

    Note: This function cannot be tested with pytest because it requires an UploadFile object
    TODO (optional): Refactor this function to make it testable
    :param file: The uploaded file
    :return: The image as a numpy array
    """
    # Read the image file into memory
    contents = await file.read()

    # Convert the contents to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert the PIL image to a numpy array
    numpy_image = np.array(pil_image)

    return numpy_image


async def process_image(file: UploadFile = File(...)) -> np.ndarray:
    np_image = await read_image_file(file)
    np_image = convert_to_rgb(np_image)

    if detect_face_count(np_image) == 1:
        return np_image

    return False
    pass


def detect_face_count(mp_image: mp.Image) -> int:
    # Implement face recognition logic

    # STEP 1: Create an FaceDetector object.
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            'mp_models/face_detection/blaze_face_short_range.tflite'))
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.7)
    detector = vision.FaceDetector.create_from_options(options)

    # STEP 2: Detect faces in the input image.
    detection_result = detector.detect(mp_image)

    # STEP 3: Count the number of faces detected.
    face_count = len(detection_result.detections)

    return face_count


async def process_face(image: np.ndarray) -> np.ndarray:
    # Implement face processing logic
    pass
