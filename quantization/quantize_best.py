from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "quantization" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Экспорт best.pt в ONNX и квантование модели в INT8."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Путь к файлу best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для экспортированной и квантованной моделей.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения для экспорта и калибровки.")
    parser.add_argument("--opset", type=int, default=17, help="Версия ONNX opset для экспорта.")
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Упростить ONNX-граф при экспорте, если локальная среда это поддерживает.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "yolo", "rtdetr"),
        default="rtdetr",
        help="Тип загрузчика модели. Для чекпоинтов RT-DETR используйте rtdetr.",
    )
    parser.add_argument(
        "--mode",
        choices=("dynamic", "static"),
        default="dynamic",
        help=(
            "dynamic: INT8-квантование весов без калибровки. "
            "static: INT8-квантование весов и активаций по калибровочным изображениям."
        ),
    )
    parser.add_argument(
        "--calibration-images",
        type=Path,
        help="Папка с репрезентативными изображениями для static INT8-калибровки.",
    )
    parser.add_argument(
        "--calibration-limit",
        type=int,
        default=100,
        help="Максимальное количество калибровочных изображений для static-режима.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Использовать готовый output/best_fp32.onnx вместо повторного экспорта best.pt.",
    )
    return parser.parse_args()


def load_ultralytics_model(weights: Path, model_type: str):
    try:
        from ultralytics import RTDETR, YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Сначала установите зависимости проекта: pip install -r requirements.txt") from exc

    normalized_type = model_type
    if normalized_type == "auto":
        name = weights.name.lower()
        normalized_type = "rtdetr" if "rtdetr" in name or "rt-detr" in name else "yolo"

    if normalized_type == "rtdetr":
        return RTDETR(str(weights))
    return YOLO(str(weights))


def export_onnx(
    weights: Path,
    output_dir: Path,
    imgsz: int,
    opset: int,
    model_type: str,
    simplify: bool,
) -> Path:
    if not weights.exists():
        raise FileNotFoundError(f"Файл весов не найден: {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_ultralytics_model(weights, model_type)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,
        )
    )

    target = output_dir / "best_fp32.onnx"
    shutil.copy2(exported, target)
    return target


def quantize_dynamic_model(fp32_model: Path, int8_model: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ONNX Runtime не установлен. Выполните: pip install -r requirements.txt"
        ) from exc

    quantize_dynamic(
        model_input=str(fp32_model),
        model_output=str(int8_model),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def image_files(path: Path, limit: int) -> Iterator[Path]:
    suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
    count = 0
    for file in sorted(path.rglob("*")):
        if file.suffix.lower() not in suffixes:
            continue
        yield file
        count += 1
        if count >= limit:
            break


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


class ImageCalibrationDataReader:
    def __init__(self, model_path: Path, images_dir: Path, imgsz: int, limit: int) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "ONNX Runtime не установлен. Выполните: pip install -r requirements.txt"
            ) from exc

        if not images_dir.exists():
            raise FileNotFoundError(f"Папка с калибровочными изображениями не найдена: {images_dir}")

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        self.images = list(image_files(images_dir, limit))
        if not self.images:
            raise FileNotFoundError(f"Калибровочные изображения не найдены в папке: {images_dir}")

        self.imgsz = imgsz
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.images):
            return None

        image_path = self.images[self.index]
        self.index += 1

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return self.get_next()

        image = letterbox(image, self.imgsz)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return {self.input_name: image}


def quantize_static_model(
    fp32_model: Path,
    int8_model: Path,
    calibration_images: Path,
    imgsz: int,
    calibration_limit: int,
) -> None:
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ONNX Runtime не установлен. Выполните: pip install -r requirements.txt"
        ) from exc

    reader = ImageCalibrationDataReader(
        model_path=fp32_model,
        images_dir=calibration_images,
        imgsz=imgsz,
        limit=calibration_limit,
    )
    quantize_static(
        model_input=str(fp32_model),
        model_output=str(int8_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fp32_model = args.output_dir / "best_fp32.onnx"
    if args.skip_export:
        if not fp32_model.exists():
            raise FileNotFoundError(f"FP32 ONNX-модель не найдена: {fp32_model}")
    else:
        fp32_model = export_onnx(
            weights=args.weights,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            opset=args.opset,
            model_type=args.model_type,
            simplify=args.simplify,
        )

    int8_model = args.output_dir / f"best_int8_{args.mode}.onnx"
    if args.mode == "dynamic":
        quantize_dynamic_model(fp32_model, int8_model)
    else:
        if args.calibration_images is None:
            raise SystemExit("Для static-квантования нужно указать --calibration-images.")
        quantize_static_model(
            fp32_model=fp32_model,
            int8_model=int8_model,
            calibration_images=args.calibration_images,
            imgsz=args.imgsz,
            calibration_limit=args.calibration_limit,
        )

    print(f"FP32 ONNX-модель: {fp32_model}")
    print(f"INT8 ONNX-модель: {int8_model}")


if __name__ == "__main__":
    main()
