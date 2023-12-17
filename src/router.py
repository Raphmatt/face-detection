from fastapi import APIRouter
from fastapi import UploadFile, File

from src import service

router = APIRouter()


@router.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}


@router.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    return service.process_image(file)