import cv2
import numpy as np
import mediapipe as mp


def convert_to_opencv_image(numpy_image: np.ndarray) -> np.ndarray:
    # Convert to OpenCV image format if needed
    # (Assuming the input is already in RGB format)
    opencv_image = numpy_image.copy()
    return opencv_image


def convert_to_mp_image(numpy_image: np.ndarray) -> mp.Image:
    # Create the MediaPipe Image from numpy array (assuming RGB format)
    mp_image = mp.Image(mp.ImageFormat.SRGB, numpy_image)
    return mp_image


def convert_to_rgb(numpy_image: np.ndarray) -> np.ndarray:
    # If the image has 4 channels (e.g., RGBA), convert to 3 channels (RGB)
    if numpy_image.shape[-1] == 4:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)

    # Convert to RGB if not already (in case of BGR format)
    if numpy_image.shape[-1] == 3 and numpy_image.dtype == np.uint8:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    return numpy_image
