from fastapi import APIRouter
from fastapi import UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi.responses import StreamingResponse

from src import service

router = APIRouter()


@router.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}


@router.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    try:
        image_array = await service.process_image(file)

        # Ensure the NumPy array is in the correct format (e.g., uint8)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # Convert the NumPy array to an image
        image = Image.fromarray(image_array)

        # Save the image to a BytesIO object
        img_io = BytesIO()
        image.save(img_io, 'JPEG', quality=100)
        img_io.seek(0)

        # Return the image in response
        return StreamingResponse(img_io, media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}