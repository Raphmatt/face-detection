import os
from typing import Any

import dlib
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import ndarray, dtype
from PIL import Image, ImageFilter
import math

from face_aligner import FaceAligner


def align_face(
        cv_image: np.ndarray,
        method: str = 'mediapipe',
        allow_out_of_bounds: bool = False) -> np.ndarray:
    # Convert the BGR image to RGB (if your model expects RGB input)
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Detect faces and get bounding boxes
    face_angle, left, right = get_face_details(image_rgb, method=method)

    if face_angle is None:
        raise ValueError("No face detected.")

    # Align the face
    aligned_image, out_of_bounds, rgba_aligned_image = (FaceAligner(
        eye_spacing=(0.36, 0.4),
        desired_width=512,
        desired_height=640)
                                                        .align(cv_image, left, right))

    if out_of_bounds and not allow_out_of_bounds:
        raise ValueError("Face is out of bounds. (The face is too close to the edge of the image.)")

    binary_mask = get_binary_mask(mp.Image(mp.ImageFormat.SRGB, aligned_image))
    _, binary_mask = cv2.threshold(cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    binary_mask = np.array(Image.fromarray(binary_mask).filter(ImageFilter.ModeFilter(size=10)))

    if out_of_bounds:
        # Use the alpha channel from rgba_aligned_image as an additional mask
        alpha_mask = rgba_aligned_image[:, :, 3]

        # Combine the alpha mask with the binary mask
        combined_mask = cv2.bitwise_and(binary_mask, alpha_mask)

        # Apply the combined mask to get the foreground (face)
        foreground = cv2.bitwise_and(rgba_aligned_image, rgba_aligned_image, mask=combined_mask)
    else:
        # If not out of bounds, use the binary mask alone
        foreground = cv2.bitwise_and(aligned_image, aligned_image, mask=binary_mask)

    # Convert the foreground to RGBA
    foreground_rgba = cv2.cvtColor(foreground, cv2.COLOR_RGB2RGBA)

    # Set the alpha channel to the combined mask or binary mask
    foreground_rgba[:, :, 3] = combined_mask if out_of_bounds else binary_mask

    return foreground_rgba  # Return the RGBA image with transparent background


def get_face_count(mp_image: mp.Image, method: str = "mediapipe") -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Detect the number of faces in an image using Mediapipe.
    :param mp_image: The input image as a MediaPipe Image object.
    :param method: The method to use for face detection('mediapipe' or 'dlib').
    :return: The number of faces detected and their bounding box coordinates.
    """

    if method == "mediapipe":
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "mp_models",
            "face_detection",
            "blaze_face_short_range.tflite",
        )
        with open(model_path, "rb") as f:
            model = f.read()

        # STEP 1: Create an FaceDetector object.
        options = vision.FaceDetectorOptions(
            base_options=(python.BaseOptions(
                model_asset_buffer=model
            )), min_detection_confidence=0.7
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
    elif method == "dlib":
        # Initialize dlib's face detector and facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "dlib_models",
            "shape_predictor_68_face_landmarks.dat"
        )
        predictor = dlib.shape_predictor(model_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray, 1)

        # Return the number of faces detected and their bounding boxes

        bbox = []

        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            bbox.append((x, y, w, h))

        return len(faces), bbox


def get_face_details(cv_image: np.ndarray, method: str = 'mediapipe') -> tuple[Any, tuple[int, int], tuple[int, int]] | \
                                                                         tuple[None, None, None]:
    """
    Gets the angle of the face as well as the coordinates of the left and right eye.
    :param cv_image: The input image with the face to be rotated as a numpy array.
    :param method: The method to use for face detection ('mediapipe' or 'dlib').
    :return: The angle of the face, the coordinates of the left eye and the coordinates of the right eye.
    """
    if method == 'mediapipe':
        # Initialize the FaceMesh object with a 'with' statement for automatic resource management.
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                min_detection_confidence=0.5) as face_mesh:
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

    elif method == 'dlib':
        # Initialize dlib's face detector and facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "dlib_models",
            "shape_predictor_68_face_landmarks.dat"
        )
        predictor = dlib.shape_predictor(model_path)  # Path to the landmark predictor model

        # Convert the image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray, 1)

        # Check if exactly one face is detected
        if len(faces) == 1:
            for face in faces:
                landmarks = predictor(gray, face)

                # Get the coordinates for the left and right eye
                left_eye = landmarks.part(36)  # Assuming points 36 is the left eye
                right_eye = landmarks.part(45)  # Assuming points 45 is the right eye

                left_eye_point = (left_eye.x, left_eye.y)
                right_eye_point = (right_eye.x, right_eye.y)

                # Calculate the angle
                dy = right_eye_point[1] - left_eye_point[1]
                dx = right_eye_point[0] - left_eye_point[0]

                # Store angle in degrees and return
                return np.degrees(np.arctan2(dy, dx)), left_eye_point, right_eye_point

        # Return None if more than one face is detected or no faces are detected
        return None, None, None

    else:
        raise ValueError("Invalid method specified. Choose 'mediapipe' or 'dlib'.")


def get_binary_mask(mp_image: mp.Image, method: str = "selfie") -> ndarray[Any, dtype[Any]]:
    """
    Returns a binary mask of the input image using Mediapipe.
    A 3 diamensional numpy array is returned. The background is white, the foreground is black. (255, 255, 255) and (0, 0, 0) respectively.
    :param mp_image: The input image as a MediaPipe Image object.
    :param method: Method used for background masking. "selfie" or "multiclass"
    :return: Binary mask as a numpy array. The bg is white, the fg is black.
    """

    final_image = None

    if method == "selfie":

        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "mp_models",
            "segmentation",
            "square_selfie_segmenter.tflite",
        )
        with open(model_path, "rb") as f:
            model = f.read()

        # Create an image segmenter instance with the image mode:
        options = vision.ImageSegmenterOptions(
            base_options=(python.BaseOptions(
                model_asset_buffer=model
            )), output_category_mask=True
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

            final_image = output_image

    elif method == "multiclass":
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "mp_models",
            "segmentation",
            "selfie_multiclass_256x256.tflite",
        )

        with open(model_path, "rb") as f:
            model = f.read()

        options = vision.ImageSegmenterOptions(
            base_options=(python.BaseOptions(
                model_asset_buffer=model
            )),
            output_category_mask=True
        )

        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask

            # Select only the hair category (usually represented by index 1)
            condition_background = (category_mask.numpy_view() == 0)
            condition_hair = (category_mask.numpy_view() == 1)
            condition_body = (category_mask.numpy_view() == 2)
            condition_face_skin = (category_mask.numpy_view() == 3)
            condition_clothes = (category_mask.numpy_view() == 4)
            condition_others = (category_mask.numpy_view() == 5)

            # Combine the conditions (logical OR)
            combined_condition = np.logical_or(condition_hair, condition_body)
            combined_condition = np.logical_or(combined_condition, condition_face_skin)
            combined_condition = np.logical_or(combined_condition, condition_clothes)
            combined_condition = np.logical_or(combined_condition, condition_others)

            # Stack the combined condition to apply it on RGB channels
            combined_condition_stacked = np.stack((combined_condition,) * 3, axis=-1)

            # Create a black background image
            black_background = np.zeros_like(mp_image.numpy_view())

            # create a white background image
            white_background = np.ones_like(mp_image.numpy_view()) * 255

            # Apply the combined condition on the black background
            black_background = np.where(combined_condition_stacked, black_background, white_background)

            final_image = black_background

    pil_image_mask = Image.fromarray(final_image)
    pil_image_mask = pil_image_mask.filter(ImageFilter.ModeFilter(size=10))

    return np.array(pil_image_mask)

def face_looking_streight(image: np.ndarray, x_threshold_angle_up=5, x_threshold_angle_down=-2, y_threshold_angle=5) -> bool:
    threshold_angle = 6

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    with mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        img_h = mp_image.height
        img_w = mp_image.width
        face_3d = []
        face_2d = []

        np_array = image

        results = face_mesh.process(np_array)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected.")


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

            focal_length = img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            if x > x_threshold_angle_up or x < x_threshold_angle_down:
                print("x: " + str(x))
                return False
            if y > y_threshold_angle or y < -y_threshold_angle:
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
