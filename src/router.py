from fastapi import APIRouter
from fastapi import UploadFile, File

router = APIRouter()


@router.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}


@router.post("/image/process")
async def create_upload_file(file: UploadFile = File(...)):
    return False
    # return {"item_id": image_id, "file_info": file.filename}
