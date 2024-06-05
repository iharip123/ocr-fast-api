import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from project import predict

app = FastAPI()

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadResponse(BaseModel):
    message: str
    filename: str = None
    output: str = None

class ErrorResponse(BaseModel):
    error: str

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/api")
async def get_api():
    return {"message": "Welcome to Malayalam OCR Tool"}

@app.post("/api/upload_image", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_image(image: UploadFile = File(...)):
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="File not allowed")

    filename = "image.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await image.read())
        
        value = predict.convert(file_path)
        return {"message": "Image uploaded", "filename": filename, "output": value}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(status_code=404, content={"error": "Not found"})

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(status_code=500, content={"error": "Server error"})

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=5000)
