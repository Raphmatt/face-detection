import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


DESIRED_FACE_WIDTH = 300  # Set your desired face width here

BG_COLOR = (0, 255, 196)

MODEL = os.path.join(
    os.path.dirname(__file__),
    "../src/models/mp_models/segmentation/selfie_multiclass_256x256.tflite")

with open(MODEL, 'rb') as f:
    model = f.read()


cap = cv2.VideoCapture(0)
prevTime = 0

# Create the options that will be used for ImageSegmenter
base_options_segmenter = python.BaseOptions(model_asset_buffer=model)
options_segmenter = vision.ImageSegmenterOptions(base_options=base_options_segmenter,
                                                 output_category_mask=True)

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    with python.vision.ImageSegmenter.create_from_options(options_segmenter) as segmenter:
        bg_image = None

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # image = cv2.imread(
            #    "/src/tests/testdata/rotated_face_1.jpg")


            # Convert the BGR image to RGB (if your model expects RGB input)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create the MediaPipe Image from numpy array (assuming RGB format)
            mp_image = mp.Image(mp.ImageFormat.SRGB, image_rgb)

            face_mesh_results = face_mesh.process(image)

            output_image = image_rgb

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    # Extract landmarks for left and right eyes
                    # Assuming landmarks 130 and 359 are the points we are interested in
                    left_eye = face_landmarks.landmark[130]
                    right_eye = face_landmarks.landmark[359]

                    # Convert from relative coordinates to image coordinates
                    left_eye_point = (int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0]))
                    right_eye_point = (int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0]))

                    # Calculate the angle
                    dy = right_eye_point[1] - left_eye_point[1]
                    dx = right_eye_point[0] - left_eye_point[0]
                    angle = np.degrees(np.arctan2(dy, dx))

                    # round angle to 2 decimal places
                    angle = round(angle, 2)
                    print(angle)

                    # Calculate the center for rotation
                    center = (
                        (left_eye_point[0] + right_eye_point[0]) // 2, (left_eye_point[1] + right_eye_point[1]) // 2)

                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)  # Changed -angle to angle

                    # Rotate the image
                    output_image = cv2.warpAffine(image_rgb, rotation_matrix, (image.shape[1], image.shape[0]))

                    # Recalculate eye positions after rotation
                    left_eye_point_rotated = np.dot(rotation_matrix,
                                                    np.array([left_eye_point[0], left_eye_point[1], 1]))
                    right_eye_point_rotated = np.dot(rotation_matrix,
                                                     np.array([right_eye_point[0], right_eye_point[1], 1]))

                    # Calculate the current distance between the eyes
                    eye_distance = np.linalg.norm(
                        np.array(right_eye_point_rotated[:2]) - np.array(left_eye_point_rotated[:2]))

                    # Calculate the scaling factor
                    scaling_factor = DESIRED_FACE_WIDTH / eye_distance

                    # Apply uniform scaling to the rotated image
                    new_width = int(image.shape[1] * scaling_factor)
                    new_height = int(image.shape[0] * scaling_factor)
                    output_image = cv2.resize(output_image, (new_width, new_height))

                    # Recalculate the face center after scaling
                    face_center_x_scaled = (left_eye_point_rotated[0] + right_eye_point_rotated[0]) / 2 * scaling_factor
                    face_center_y_scaled = (left_eye_point_rotated[1] + right_eye_point_rotated[1]) / 2 * scaling_factor

                    # Calculate shift to move face center to image center
                    shift_x = image.shape[1] / 2 - face_center_x_scaled
                    shift_y = image.shape[0] / 2 - face_center_y_scaled

                    # Apply the translation
                    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    output_image = cv2.warpAffine(output_image, translation_matrix, (new_width, new_height))

                    # Draw landmarks after rotation
                    # mp_drawing.draw_landmarks(
                    #    image=output_image,
                    #    landmark_list=face_landmarks,
                    #    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    # )

                # Convert the RGB image back to BGR
                output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                cv2.putText(output_image_bgr, f'fps: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

                cv2.imshow('DIY Background removal', output_image_bgr)
                #cv2.waitKey(0)
                if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
                    break
