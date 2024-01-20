import io
import os
from typing import Any

import cv2
import math

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import ndarray, dtype

from src.face_utilities import get_face_count, align_face
from src.image_utilities import uploadFile_2_np_image
from src.face_aligner import FaceAligner


async def process_image(file: UploadFile = File(...)) -> np.ndarray:
    np_image = await uploadFile_2_np_image(file)

    face_count, face_boxes = get_face_count(mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image))

    if face_count != 1:
        raise ValueError("Image must contain exactly one face. Current face count: " + str(face_count))

    # additional checks if the face is valid

    # align face
    final_image = align_face(np_image)

    return final_image


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
                return False
            else:
                return True
    else:
        return False
