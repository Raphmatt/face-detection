from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import uvicorn

app = FastAPI()

fake_db = {}
upload_folder = "temp"  # Folder to store uploaded images

# Create the upload folder if it doesn't exist
os.makedirs(upload_folder, exist_ok=True)


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "item_info": fake_db[item_id]}


@app.post("/items/{item_id}")
async def create_item(item_id: int, item_info: str):
    if item_id in fake_db:
        raise HTTPException(status_code=400, detail="Item already exists")
    fake_db[item_id] = item_info
    return {"item_id": item_id, "item_info": item_info}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    image_id = str(uuid.uuid4())
    file_path = os.path.join(upload_folder, f"{image_id}.{file_extension}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"item_id": image_id, "file_info": file.filename}


@app.get("/downloadfile/{image_id}")
async def download_file(image_id: str):
    # Look for any file in the folder with the provided image_id
    matching_files = [f for f in os.listdir(upload_folder) if f.startswith(image_id)]

    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")

    # Assuming there's only one matching file, you can choose the first one
    file_path = os.path.join(upload_folder, matching_files[0])

    return FileResponse(file_path, filename=matching_files[0])


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
