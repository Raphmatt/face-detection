import io
import math
import os
import cv2
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


async def process_face(image: np.ndarray) -> np.ndarray:
    # Implement face processing logic
    pass


def remove_background(mp_image: mp.Image, background_color=(255, 255, 255)) -> mp.Image:
    """
    Remove the background of an image using Mediapipe and replace it with a plain color.

    :param mp_image: The input image with the background to be removed.
    :param background_color: The color to replace the background with (default is white).
    :return: The image with the background removed.
    """

    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            'mp_models/segmentation/square_selfie_segmenter.tflite'))
    vision_running_mode = mp.tasks.vision.RunningMode

    # Create an image segmenter instance with the image mode:
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision_running_mode.IMAGE,
        output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)

    image_rgb = cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB)

    # Segment the image to obtain masks
    segmented_masks = segmenter.segment(mp_image)

    # Extract the mask from the segmented result
    category_mask = segmented_masks.category_mask

    # Select only the hair category (usually represented by index 1)
    person_condition = (category_mask.numpy_view() == 1)

    person_condition_stacked = np.stack((person_condition,) * 3, axis=-1)

    # Convert the mask to a binary image
    binary_mask = (person_condition > 0).astype(np.uint8)

    # Resize the binary mask to the original image size
    binary_mask = cv2.resize(binary_mask, (mp_image.width, mp_image.height))

    white_background = np.ones_like(mp_image) * 255
    # Apply the mask: keep only the hair, make everything else black
    output_image = np.where(person_condition_stacked, image_rgb, white_background)

    # Convert the RGB image back to BGR
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('DIY Background removal', output_image_bgr)

    return output_image
