import cv2
import numpy as np


def rgba2rgb(numpy_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGBA numpy image to an RGB numpy image
    :param numpy_image: The RGBA numpy image
    :return: The RGB numpy image
    """
    # If the image has 4 channels (e.g., RGBA), convert to 3 channels (RGB)
    if numpy_image.shape[-1] == 4:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)

    return numpy_image
