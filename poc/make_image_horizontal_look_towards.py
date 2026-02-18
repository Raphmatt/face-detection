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

BG_COLOR = (0, 255, 196)

MODEL = os.path.join(
    os.path.dirname(__file__),
    "../src/models/mp_models/segmentation/selfie_multiclass_256x256.tflite",
)

with open(MODEL, "rb") as f:
    model = f.read()


cap = cv2.VideoCapture(1)
prevTime = 0

# Create the options that will be used for ImageSegmenter
base_options_segmenter = python.BaseOptions(model_asset_buffer=model)
options_segmenter = vision.ImageSegmenterOptions(
    base_options=base_options_segmenter, output_category_mask=True
)


def calculate_face_transformation(face_landmarks, image_shape):
    # Select key landmarks for the transformation
    # For example: corners of the eyes, nose tip, and chin
    # Replace indices with the correct ones for your model
    landmarks_indices = [33, 133, 0, 152]  # Example indices

    # Destination points - where we want the landmarks to be after transformation
    # These would be the points for a front-facing face
    dst_points = np.float32(
        [
            [100, 100],  # Corresponding point for landmark 33
            [200, 100],  # Corresponding point for landmark 133
            # ... add more corresponding points ...
        ]
    )

    # Extract the corresponding source points from the face landmarks
    src_points = np.float32(
        [
            [
                face_landmarks.landmark[i].x * image_shape[1],
                face_landmarks.landmark[i].y * image_shape[0],
            ]
            for i in landmarks_indices
        ]
    )

    # Calculate the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return transformation_matrix


with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    with python.vision.ImageSegmenter.create_from_options(
        options_segmenter
    ) as segmenter:
        bg_image = None

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB (if your model expects RGB input)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create the MediaPipe Image from numpy array (assuming RGB format)
            mp_image = mp.Image(mp.ImageFormat.SRGB, image_rgb)

            # segmentation_result = segmenter.segment(mp_image)
            # category_mask = segmentation_result.category_mask

            face_mesh_results = face_mesh.process(image)

            output_image = image_rgb

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    transformation_matrix = calculate_face_transformation(
                        face_landmarks, image.shape
                    )

                    # Perform the transformation
                    output_image = cv2.warpPerspective(
                        output_image,
                        transformation_matrix,
                        (image.shape[1], image.shape[0]),
                    )

                    # Draw landmarks after transformation
                    mp_drawing.draw_landmarks(
                        image=output_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                    )

            # Convert the RGB image back to BGR
            output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(
                output_image_bgr,
                f"fps: {int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 0, 0),
                2,
            )

            cv2.imshow("DIY Background removal", output_image_bgr)
            if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
                break
