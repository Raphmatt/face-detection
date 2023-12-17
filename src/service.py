import io
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils.convert import rgba2rgb


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
    np_image = rgba2rgb(np_image)

    if detect_face_count(np_image) == 1:
        return np_image

    return np.array([])
    pass


def detect_face_count(mp_image: np.ndarray) -> int:
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
    # noinspection PyTypeChecker
    detection_result = detector.detect(mp_image)  # type: mp.Image

    # STEP 3: Count the number of faces detected.
    face_count = len(detection_result.detections)

    return face_count


def shoulder_angle_valid(mp_image: mp.Image) -> bool:

    threshold_angle = 30.0
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            'mp_models/pose_detection/pose_landmarker_lite.task'))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=0.5)
    detector = vision.PoseLandmarker.create_from_options(options)

    pose_result = detector.detect(mp_image)

    # if there are no shoulders on the image it should return True
    if not pose_result.pose_landmarks:
        return True

    # person's perspective
    left_shoulder = pose_result.pose_landmarks[0][11]
    right_shoulder = pose_result.pose_landmarks[0][12]

    angle_radians = math.atan2(abs(right_shoulder.y - left_shoulder.y), abs(right_shoulder.x - left_shoulder.x))
    angle_degrees = math.degrees(angle_radians)

    if angle_degrees < threshold_angle:
        return True
    else:
        return False

    # Calculate the angle
    # left_shoulder = np.array([left_shoulder.x, left_shoulder.y])
    # right_shoulder = np.array([right_shoulder.x, right_shoulder.y])
    #
    # # Plot the line connecting the shoulders
    # plt.plot([left_shoulder[0], right_shoulder[0]], [left_shoulder[1], right_shoulder[1]], label='Shoulder Line')
    #
    # # Mark the shoulder points
    # plt.scatter(left_shoulder[0], left_shoulder[1], color='red', label='Left Shoulder')
    # plt.scatter(right_shoulder[0], right_shoulder[1], color='blue', label='Right Shoulder')
    #
    # # Set plot limits
    # plt.xlim(-1, 4)
    # plt.ylim(-1, 5)
    #
    # # Add a legend
    # plt.legend()
    #
    # # Add angle information to the plot
    # plt.text(left_shoulder[0], left_shoulder[1], f'{angle:.2f} degrees', fontsize=10, ha='right', va='bottom')
    #
    # # Show the plot
    # plt.grid(True)
    # plt.show()
    #
    # # Check if the angle is between -10 and 10 degrees
    # return -10 <= angle <= 10


async def process_face(image: np.ndarray) -> np.ndarray:
    # Implement face processing logic
    pass
