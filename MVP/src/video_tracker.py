from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from preprocessing import CameraDenoiseConfig, CameraDenoiser

def load_model(model_name: str = "Best_multiclass_with_faults_yolov11m.pt"):
    repo_root = Path(__file__).parent.parent  
    model_path = repo_root / "models" / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    print(f"Загружаю модель: {model_path}")
    return YOLO(str(model_path))

def load_video(video_name: str = "video_drone.mp4"):
    repo_root = Path(__file__).parent.parent  
    video_path = repo_root / "data" / video_name
    
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    
    print(f"Загружаю видео: {video_path}")
    return cv2.VideoCapture(str(video_path))

# Загрузка видео и модели
model = load_model()
video = load_video()
denoiser = CameraDenoiser(
    CameraDenoiseConfig(
        enabled=True,
        temporal_alpha=0.15,
        bilateral=True,
        clahe=True,
    )
)

# Параметры видео

fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = Path(__file__).parent.parent / "results" / "result_with_yolo.mp4"
output_path.parent.mkdir(exist_ok=True)  

out = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (w, h)
)


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frame = denoiser(frame)

    results = model.predict(
        frame,
        conf=0.25,
        iou=0.45,
        verbose=False,      
        imgsz=640,
        device=0            
    )
    if results and len(results) > 0 and results[0].boxes is not None:
            annotated_frame = results[0].plot()
    else:
            annotated_frame = frame.copy()  # просто копируем оригинал
    # print(annotated_frame.shape[2])
    out.write(annotated_frame)

video.release()
out.release()
cv2.destroyAllWindows()
