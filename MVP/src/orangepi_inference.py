from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import RTDETR, YOLO

from preprocessing import DroneFramePreprocessor, DronePreprocessConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best.onnx"
DEFAULT_OUTPUT = ROOT / "results" / "orangepi_inference.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference inference pipeline for Orange Pi 5 onboard detection."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a YOLO/RT-DETR model (.pt or exported .onnx).",
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=("auto", "yolo", "rtdetr"),
        help="Force model loader type when auto-detection from filename is ambiguous.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video file path, or RTSP URL.",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. For Orange Pi 5 start with 'cpu'.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Run detector every N-th frame to save compute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Optional output video path.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Show preview window. Usually disable this on headless boards.",
    )
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def build_preprocessor() -> DroneFramePreprocessor:
    return DroneFramePreprocessor(
        DronePreprocessConfig(
            enabled=True,
            temporal_alpha=0.10,
            clahe=True,
            bilateral=False,
            gamma=1.05,
            sharpen_amount=0.10,
            white_balance=False,
        )
    )


def load_model(model_path: Path, model_type: str):
    normalized_type = model_type.strip().lower()
    if normalized_type == "auto":
        model_name_lower = model_path.name.lower()
        if model_path.suffix.lower() == ".onnx":
            normalized_type = "yolo"
        elif "rtdetr" in model_name_lower or "rt-detr" in model_name_lower:
            normalized_type = "rtdetr"
        else:
            normalized_type = "yolo"

    if normalized_type == "rtdetr":
        return RTDETR(str(model_path))
    return YOLO(str(model_path))


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    capture = open_source(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    model = load_model(args.model, args.model_type)
    preprocessor = build_preprocessor()

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    last_annotated = None
    start_time = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        processed = preprocessor(frame)

        if frame_index % max(args.frame_stride, 1) == 0 or last_annotated is None:
            results = model.predict(
                processed,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            last_annotated = results[0].plot() if results else processed

        frame_index += 1
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        effective_fps = frame_index / elapsed

        annotated = last_annotated.copy()
        cv2.putText(
            annotated,
            f"FPS {effective_fps:.1f} | stride {args.frame_stride}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)

        if args.view:
            cv2.imshow("orangepi-inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = max(time.perf_counter() - start_time, 1e-6)
    print(
        f"Processed {frame_index} frames in {total_time:.1f}s "
        f"({frame_index / total_time:.2f} FPS average)."
    )
    print(f"Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()
