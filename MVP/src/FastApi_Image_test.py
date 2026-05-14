from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import uvicorn
import base64
from pathlib import Path

from preprocessing import CameraDenoiser

app = FastAPI(title= "YOLO Object detection API")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Best_multiclass_with_faults_yolov11m.pt"
model = YOLO(str(MODEL_PATH))
denoiser = CameraDenoiser()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>YOLO Детекция</title></head>
        <body>
            <h1>Загрузите изображение для детекции объектов</h1>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Обработать">
            </form>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Читаем файл
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Не удалось загрузить изображение"}

    img = denoiser(img)

    results = model(img, conf=0.25)  

    annotated_img = results[0].plot()  

    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    img_byte_arr = encoded_img.tobytes()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/jpeg")


# Запуск: uvicorn main:app --reload
