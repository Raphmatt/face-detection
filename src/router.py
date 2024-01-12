from fastapi import APIRouter
from fastapi import UploadFile, File

from src import service

router = APIRouter()


@router.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}


@router.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    try:
        return service.process_image(file)
    except Exception as e:
        return {"error": str(e)}
