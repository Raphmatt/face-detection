import os
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import ndarray, dtype

from src.face_aligner import FaceAligner


def align_face(cv_image: np.ndarray) -> np.ndarray:
    # Convert the BGR image to RGB (if your model expects RGB input)
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Detect faces and get bounding boxes
    face_angle, left, right = get_face_details(image_rgb)

    aligned_image = FaceAligner(eye_spacing=(0.36, 0.4), desired_width=512, desired_height=640).align(cv_image, left, right)

    if face_angle is None:
        raise ValueError("No face detected.")

    binary_mask = get_binary_mask(mp.Image(mp.ImageFormat.SRGB, aligned_image))

    inverted_binary_mask = cv2.bitwise_not(binary_mask)
    _, mask = cv2.threshold(cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    _, inverted_mask = cv2.threshold(cv2.cvtColor(inverted_binary_mask, cv2.COLOR_BGR2GRAY), 200, 255,
                                     cv2.THRESH_BINARY_INV)

    # optimized version (optional)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    inverted_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

    # Apply inverted mask to get the foreground (face)
    foreground = cv2.bitwise_and(aligned_image, aligned_image, mask=mask)

    # Create a red background
    bg = np.zeros_like(aligned_image)
    bg[:, :] = [0xF0, 0xFF, 0xEB]  # ffffeb

    background = cv2.bitwise_and(bg, bg, mask=inverted_mask)

    # Combine foreground and background
    final_image = cv2.add(foreground, background)

    return final_image


def get_face_count(mp_image: mp.Image) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Detect the number of faces in an image using Mediapipe.
    :param mp_image: The input image as a MediaPipe Image object.
    :return: The number of faces detected and their bounding box coordinates.
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

        # STEP 3: Count the number of faces detected and get their bounding boxes.
        face_count = len(detection_result.detections)
        face_boxes = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)
            face_boxes.append((x, y, w, h))

        return face_count, face_boxes


def get_face_details(cv_image: np.ndarray) -> tuple[Any, tuple[int, int], tuple[int, int]] | tuple[None, None, None]:
    """
    Gets the angle of the face as well as the coordinates of the left and right eye.
    :param cv_image: The input image with the face to be rotated as a numpy array.
    :return: The angle of the face, the coordinates of the left eye and the coordinates of the right eye.
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

                    # Store angle in degrees and return (also return left_eye and right_eye points for testing)
                    return np.degrees(np.arctan2(dy, dx)), left_eye_point, right_eye_point
                else:
                    # Return None if more than one face is detected.
                    return None, None, None
        # Return None if no faces are detected or there is any other issue.
        return None, None, None


def get_binary_mask(mp_image: mp.Image, method: str = "selfie") -> ndarray[Any, dtype[Any]]:
    """
    Returns a binary mask of the input image using Mediapipe.
    A 3 diamensional numpy array is returned. The background is white, the foreground is black. (255, 255, 255) and (0, 0, 0) respectively.
    :param mp_image: The input image as a MediaPipe Image object.
    :param method: Method used for background masking. "selfie" or "multiclass"
    :return: Binary mask as a numpy array. The bg is white, the fg is black.
    """

    if method == "selfie":
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

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1)
            output_image = np.where(condition, fg_image, bg_image)

            return output_image
