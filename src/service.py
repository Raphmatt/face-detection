import mediapipe as mp
import numpy as np
from fastapi import UploadFile, File

from face_utilities import (
    get_face_count,
    align_face,
    get_face_details,
    face_looking_straight,
    shoulder_angle_valid,
)
from image_utilities import uploadFile_2_np_image


async def process_image(
    file: UploadFile = File(...),
    override_file=None,
    allow_out_of_bounds=False,
    spacing_side=0.72,
    spacing_top=0.4,
    desired_width=512,
    default_height=640,
    binary_method="multiclass",
) -> np.ndarray:
    """
    Processes an image and returns the aligned face.

    :param file: UploadFile from FastAPI
    :param override_file: If UploadFile is not used, this can be used to pass a numpy array directly (for manual testing
    :param allow_out_of_bounds: Allow the face to be out of bounds (e.g. if the face is on the edge of the image)
    :param spacing_side: The spacing of the eye and the side edge of the image
    :param spacing_top: The spacing of the eye and the top edge of the image
    :param desired_width: The desired width of the final image
    :param default_height: The desired height of the final image
    :return: Returns the aligned face as a numpy array
    """
    if override_file is None:
        np_image = await uploadFile_2_np_image(file)
    else:
        np_image = override_file

    method = "mediapipe"

    # check the count of faces
    face_count, face_boxes = get_face_count(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image), method="mediapipe"
    )

    if face_count != 1:
        face_count, face_boxes = get_face_count(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image), method="dlib"
        )
        if face_count != 1:
            raise ValueError(
                "Image must contain exactly one face. face count: " + str(face_count)
            )
        else:
            method = "dlib"

    face_angle, left, right = get_face_details(np_image, method=method)

    if face_angle > 10:
        raise ValueError("Face angle too large. angle: " + str(face_angle))

    if method == "mediapipe":
        shoulder_angle, shoulder_angle_okay = shoulder_angle_valid(np_image)

        if not shoulder_angle_okay:
            raise ValueError("Shoulder angle too large. angle: " + str(shoulder_angle))

        if not face_looking_straight(np_image):
            raise ValueError("Face not looking straight into camera.")

    # align face
    final_image = align_face(
        np_image,
        allow_out_of_bounds=allow_out_of_bounds,
        spacing_side=spacing_side,
        spacing_top=spacing_top,
        desired_width=desired_width,
        desired_height=default_height,
        face_angle=face_angle,
        left_eye_point=left,
        right_eye_point=right,
        binary_method=binary_method,
    )
    return final_image
