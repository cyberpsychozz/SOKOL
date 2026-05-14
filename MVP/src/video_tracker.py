from pathlib import Path
import cv2
from ultralytics import RTDETR, YOLO
from preprocessing import DroneFramePreprocessor, DronePreprocessConfig


def load_model(
    model_name: str = "Best_multiclass_with_faults_yolov11m.pt",
    model_type: str = "auto",
):
    repo_root = Path(__file__).parent.parent
    model_path = repo_root / "models" / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    normalized_type = model_type.strip().lower()
    if normalized_type not in {"auto", "yolo", "rtdetr"}:
        raise ValueError("model_type должен быть одним из: auto, yolo, rtdetr")

    model_name_lower = model_path.name.lower()
    if normalized_type == "auto":
        if model_path.suffix.lower() == ".onnx":
            normalized_type = "yolo"
        elif "rtdetr" in model_name_lower or "rt-detr" in model_name_lower:
            normalized_type = "rtdetr"
        else:
            normalized_type = "yolo"

    print(f"Загружаю модель: {model_path} ({normalized_type})")
    if normalized_type == "rtdetr":
        return RTDETR(str(model_path))
    return YOLO(str(model_path))

def load_video(video_name: str = "video_drone.mp4"):
    repo_root = Path(__file__).parent.parent
    video_path = repo_root / "data" / video_name

    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    print(f"Загружаю видео: {video_path}")
    return cv2.VideoCapture(str(video_path))


# Загрузка видео и модели
model = load_model(
    model_name="Best_multiclass_with_faults_yolov11m.pt",
    model_type="auto",
)
video = load_video()
preprocessor = DroneFramePreprocessor(
    DronePreprocessConfig(
        enabled=True,
        temporal_alpha=0.10,
        bilateral=False,
        clahe=True,
        gamma=1.05,
        sharpen_amount=0.15,
        white_balance=False,
    )
)

# Параметры видео

fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = Path(__file__).parent.parent / "results" / "result_with_detections.mp4"
output_path.parent.mkdir(exist_ok=True)

out = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frame = preprocessor(frame)

    results = model.predict(
        frame,
        conf=0.25,
        iou=0.45,
        verbose=False,
        imgsz=640,
        device=0,
    )
    if results and len(results) > 0 and results[0].boxes is not None:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()

    out.write(annotated_frame)

video.release()
out.release()
cv2.destroyAllWindows()
