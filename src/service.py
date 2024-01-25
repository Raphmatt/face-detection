import math
import os

import cv2
import mediapipe as mp
import numpy as np
from fastapi import UploadFile, File
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_utilities import get_face_count, align_face
from image_utilities import uploadFile_2_np_image


async def process_image(file: UploadFile = File(...)) -> np.ndarray:
    np_image = await uploadFile_2_np_image(file)

    face_count, face_boxes = get_face_count(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image),
        method="mediapipe")

    method = "mediapipe"

    if face_count != 1:
        face_count, face_boxes = get_face_count(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image),
            method="dlib")
        if face_count != 1:
            raise ValueError("Image must contain exactly one face. Current face count: " + str(face_count))
        else:
            method = "dlib"

    # additional checks if the face is valid

    # align face
    final_image = align_face(np_image, method=method)

    return final_image


def face_looking_streight(image: mp.Image) -> bool:
    threshold_angle = 6

    with mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        img_h = image.height
        img_w = image.width
        face_3d = []
        face_2d = []

        np_array = image.numpy_view()

        results = face_mesh.process(np_array)

        if len(results.multi_face_landmarks) != 1:
            raise ValueError(
                "Image must contain exactly one face. Current face count: " + str(len(results.multi_face_landmarks)))

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

    model_path = os.path.join(
        os.path.dirname(__file__),
        "models/mp_models",
        "pose_detection",
        "pose_landmarker_lite.task",
    )
    with open(model_path, "rb") as f:
        model_data = f.read()

    options = vision.PoseLandmarkerOptions(
        base_options=(python.BaseOptions(
            model_asset_buffer=model_data
        )), min_pose_detection_confidence=0.5
    )

    with python.vision.PoseLandmarker.create_from_options(options) as detector:
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
