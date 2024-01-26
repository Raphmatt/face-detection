import mediapipe as mp
import numpy as np
from fastapi import UploadFile, File

from face_utilities import get_face_count, align_face
from image_utilities import uploadFile_2_np_image


async def process_image(
        file: UploadFile = File(...),
        override_file=None,
        allow_out_of_bounds=False,
        spacing_side=0.72,
        spacing_top=0.4
) -> np.ndarray:
    if override_file is None:
        np_image = await uploadFile_2_np_image(file)
    else:
        np_image = override_file
    method = "mediapipe"

    # check the count of faces
    face_count, face_boxes = get_face_count(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image),
        method="mediapipe")
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
    final_image = align_face(
        np_image,
        method=method,
        allow_out_of_bounds=allow_out_of_bounds,
        spacing_side=spacing_side,
        spacing_top=spacing_top)
    return final_image
