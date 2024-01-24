import io
import math
import os

import cv2
import mediapipe as mp
import numpy as np
from PIL.Image import Image
from fastapi import UploadFile, File
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import ndarray

from face_utilities import get_face_count, align_face
from image_utilities import uploadFile_2_np_image


async def process_image(file: UploadFile = File(...)) -> np.ndarray:
    np_image = await uploadFile_2_np_image(file)

    face_count, face_boxes = get_face_count(mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image))

    if face_count != 1:
        raise ValueError("Image must contain exactly one face. Current face count: " + str(face_count))

    # additional checks if the face is valid

    # align face
    final_image = align_face(np_image)

    return final_image


def face_looking_streight(image: mp.Image) -> bool:
    threshold_angle = 6
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    img_h = image.height
    img_w = image.width
    face_3d = []
    face_2d = []

    np_array = image.numpy_view()

    results = face_mesh.process(np_array)

    if len(results.multi_face_landmarks) != 1:
        raise ValueError("Image must contain exactly one face. Current face count: " + str(len(results.multi_face_landmarks)))

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360

        if x > threshold_angle or x < -threshold_angle:
            return False
        elif y > threshold_angle or y < -threshold_angle:
            return False

    return True


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
