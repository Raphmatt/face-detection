from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import APIRouter
from fastapi import UploadFile, File, Query
from fastapi.responses import StreamingResponse

import service

router = APIRouter()


@router.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}


@router.post("/image/process")
async def process_image(
        file: UploadFile = File(...),
        bounds: bool = Query(False, alias="allow ouf of bounds"),
        spacing_side: float = Query(0.72, alias="spacing side (0 = edge, 0.99998 = middle)"),
        spacing_top: float = Query(0.4,
                                   alias="spacing top (0 = top edge, 0.99998 = bottom edge, 0.5 = middle)"),
):
    try:
        image_array = await service.process_image(file, allow_out_of_bounds=bounds, spacing_side=spacing_side,
                                                  spacing_top=spacing_top)

        # Ensure the NumPy array is in the correct format (e.g., uint8)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # Convert the NumPy array to an image
        image = Image.fromarray(image_array)

        # Save the image to a BytesIO object
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        # Return the image in response
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
