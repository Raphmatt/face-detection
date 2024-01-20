import io

import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile, File


def rgba_2_rgb(numpy_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGBA numpy image to an RGB numpy image
    :param numpy_image: The RGBA numpy image
    :return: The RGB numpy image
    """
    # If the image has 4 channels (e.g., RGBA), convert to 3 channels (RGB)
    if numpy_image.shape[-1] == 4:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)

    return numpy_image


async def uploadFile_2_np_image(file: UploadFile = File(...)) -> np.ndarray:
    """
    Read an uploaded image file into a numpy array
    Note: This function cannot be tested with pytest because it requires an UploadFile object
    :param file: The uploaded file
    :return: The image as a numpy array
    """
    # Read the image file into memory
    contents = await file.read()

    # Convert the contents to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert the PIL image to a numpy array
    numpy_image = rgba_2_rgb(np.array(pil_image))

    return numpy_image