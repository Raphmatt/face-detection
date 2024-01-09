import io
import os
from typing import Any

import math

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import ndarray, dtype

from src.utils.convert import rgba2rgb


async def convert_fastapi_obj_2_numpy_image(file: UploadFile = File(...)) -> np.ndarray:
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
    np_image = await convert_fastapi_obj_2_numpy_image(file)
    np_image = rgba2rgb(np_image)

    if detect_face_count(np_image) == 1:
        return np_image

    return np.array([])
    pass


def detect_face_count(mp_image: mp.Image):
    """
    Detect the number of faces in an image using Mediapipe.
    :param mp_image: The input image as a MediaPipe Image object.
    :return: The number of faces detected in the image.
    """
    # Implement face recognition logic

    # STEP 1: Create an FaceDetector object.
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            "mp_models",
            "face_detection",
            "blaze_face_short_range.tflite",
        )
    )
    options = vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=0.7
    )

    # Using 'with' statement for automatic resource management.
    with vision.FaceDetector.create_from_options(options) as detector:
        # STEP 2: Detect faces in the input image.
        detection_result = detector.detect(mp_image)

        # STEP 3: Count the number of faces detected.
        face_count = len(detection_result.detections)
        face_centers = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            cx = int(bbox.origin_x + bbox.width / 2)
            cy = int(bbox.origin_y + bbox.height / 2)
            face_centers.append((cx, cy))

        return face_count, cx, cy


def calculate_face_rotation(cv_image: np.ndarray) -> float or None:
    """
    Calculate the rotation of a face in an image using Mediapipe.
    :param cv_image: The input image with the face to be rotated as a numpy array.
    :return: The rotation of the face in degrees. Returns None if no or more than one face is detected.
    """
    # Initialize the FaceMesh object with a 'with' statement for automatic resource management.
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5
    ) as face_mesh:
        # STEP 2: Detect faces in the input image.
        face_mesh_results = face_mesh.process(cv_image)

        # STEP 3: Extract landmarks for left and right eyes
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Check if there is exactly one face, as intended for rotation calculation.
                if len(face_mesh_results.multi_face_landmarks) == 1:
                    # Extract landmarks for left and right eyes
                    # Assuming landmarks 130 and 359 are the points we are interested in
                    left_eye = face_landmarks.landmark[130]
                    right_eye = face_landmarks.landmark[359]

                    # Convert from relative coordinates to image coordinates
                    left_eye_point = (
                        int(left_eye.x * cv_image.shape[1]),
                        int(left_eye.y * cv_image.shape[0]),
                    )
                    right_eye_point = (
                        int(right_eye.x * cv_image.shape[1]),
                        int(right_eye.y * cv_image.shape[0]),
                    )

                    # Calculate the angle
                    dy = right_eye_point[1] - left_eye_point[1]
                    dx = right_eye_point[0] - left_eye_point[0]

                    # Store angle in degrees and return
                    return np.degrees(np.arctan2(dy, dx))
                else:
                    # Return None if more than one face is detected.
                    return None
        # Return None if no faces are detected or there is any other issue.
        return None


def shoulder_angle_valid(mp_image: mp.Image) -> bool:
    threshold_angle = 30.0
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            "mp_models",
            "pose_detection",
            "pose_landmarker_lite.task",
        )
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, min_pose_detection_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    pose_result = detector.detect(mp_image)

    # if there are no shoulders on the image it should return True
    if not pose_result.pose_landmarks:
        return True

    # person's perspective
    left_shoulder = pose_result.pose_landmarks[0][11]
    right_shoulder = pose_result.pose_landmarks[0][12]

    angle_radians = math.atan2(
        abs(right_shoulder.y - left_shoulder.y), abs(right_shoulder.x - left_shoulder.x)
    )
    angle_degrees = math.degrees(angle_radians)

    if angle_degrees < threshold_angle:
        return True
    else:
        return False


def get_background_mask(mp_image: mp.Image) -> ndarray[Any, dtype[Any]]:
    """
    Get the background mask of an image using Mediapipe.
    :param mp_image: The input image as a MediaPipe Image object.
    :return: The background mask as a numpy array. The background is white, the foreground is black.
    """

    base_options = python.BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            "mp_models",
            "segmentation",
            "square_selfie_segmenter.tflite",
        )
    )

    # Create an image segmenter instance with the image mode:
    options = vision.ImageSegmenterOptions(
        base_options=base_options, output_category_mask=True
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        person_color = (0, 0, 0)
        background_color = (255, 255, 255)

        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        # Generate solid color images for showing the output segmentation mask.
        image_data = mp_image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = background_color
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = person_color

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)

        return output_image


def crop_image(image, cx, cy) -> ndarray:
    height, width, shape = image.shape
    # Define crop dimensions
    crop_width = 250  # TODO: replace with percent
    crop_height = 500  # TODO: replace with percent

    # Calculate crop region boundaries
    left = max(0, cx - crop_width)
    right = min(width, cx + crop_width)
    top = max(0, cy - crop_height)
    bottom = min(height, cy + crop_height)

    # Crop the image
    cropped_image_np = image[top:bottom, left:right]

    return cropped_image_np


def resize_image(original_image, target_width=500, target_height=1000):
    # Calculate the scaling factors for width and height
    width_scale = target_width / original_image.shape[1]
    height_scale = target_height / original_image.shape[0]

    # Choose the smaller scaling factor to maintain the original aspect ratio
    scale_factor = min(width_scale, height_scale)

    # Resize the image
    resized_image = cv2.resize(
        original_image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_AREA,
    )

    return resized_image


def detect_occlusion(image_np):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    result = face_detection.process(image_rgb)

    if result.detections:
        for detection in result.detections:
            if detection.location_data.relative_bounding_box.ymin < 0.2:
                return "Face covered by an object"
            else:
                return "Face not covered by an object"
    else:
        return "No face detected"
