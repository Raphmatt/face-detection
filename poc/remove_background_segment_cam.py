import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (0, 255, 196)

MODEL = os.path.join(
    os.path.dirname(__file__),
    "../src/models/mp_models/segmentation/selfie_multiclass_256x256.tflite",
)

with open(MODEL, "rb") as f:
    model = f.read()


cap = cv2.VideoCapture(0)
prevTime = 0

# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_buffer=model)
options = vision.ImageSegmenterOptions(
    base_options=base_options, output_category_mask=True
)

with python.vision.ImageSegmenter.create_from_options(options) as segmenter:
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

        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        # Select only the hair category (usually represented by index 1)
        hair_condition = category_mask.numpy_view() == 1
        body_condition = category_mask.numpy_view() == 2
        face_skin_condition = category_mask.numpy_view() == 3
        clothes_condition = category_mask.numpy_view() == 4

        # Combine the conditions (logical OR)
        combined_condition = np.logical_or(hair_condition, body_condition)
        combined_condition = np.logical_or(combined_condition, face_skin_condition)
        combined_condition = np.logical_or(combined_condition, clothes_condition)

        # Stack the combined condition to apply it on RGB channels
        combined_condition_stacked = np.stack((combined_condition,) * 3, axis=-1)

        # Create a black background image
        black_background = np.zeros_like(image_rgb)

        # create a white background image
        white_background = np.ones_like(image_rgb) * 255

        # Apply the mask: keep only the hair, make everything else black
        output_image = np.where(combined_condition_stacked, image_rgb, white_background)

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
