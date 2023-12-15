import cv2
import numpy as np
import mediapipe as mp


def convert_to_mp_image(numpy_image: np.ndarray) -> mp.Image:
    """
    Convert a numpy array to a MediaPipe image
    :param numpy_image: The numpy array to convert
    :return: The MediaPipe image
    """
    # Create the MediaPipe Image from numpy array (assuming RGB format)
    mp_image = mp.Image(mp.ImageFormat.SRGB, numpy_image)
    return mp_image


def rgba2rgb(numpy_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGBA numpy image to an RGB numpy image
    :param numpy_image: The RGBA numpy image
    :return: The RGB numpy image
    """
    # If the image has 4 channels (e.g., RGBA), convert to 3 channels (RGB)
    if numpy_image.shape[-1] == 4:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)

    # Convert to RGB if not already (in case of BGR format)
    if numpy_image.shape[-1] == 3 and numpy_image.dtype == np.uint8:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    return numpy_image
