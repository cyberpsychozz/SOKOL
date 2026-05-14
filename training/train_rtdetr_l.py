from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RT-DETR-L for the SOKOL wildfire detection project."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "training" / "sokol_dataset.yaml",
        help="Path to YOLO-format dataset yaml.",
    )
    parser.add_argument(
        "--model",
        default="rtdetr-l.pt",
        help="Ultralytics RT-DETR checkpoint. Use rtdetr-l.pt for RT-DETR-L.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0, help="CUDA device id, 'cpu', or comma list.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", type=Path, default=ROOT / "training" / "runs")
    parser.add_argument("--name", default="sokol_rtdetr_l")
    parser.add_argument("--resume", action="store_true", help="Resume the last run.")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export best.pt to ONNX after training.",
    )
    return parser.parse_args()


def _parse_primary_cuda_device(device: str | int) -> int | None:
    if isinstance(device, int):
        return device

    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return None

    match = re.match(r"^\d+", normalized)
    if match:
        return int(match.group(0))

    return 0


def validate_torch_cuda_compatibility(device: str | int) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in this environment. Install it before training."
        ) from exc

    primary_device = _parse_primary_cuda_device(device)
    if primary_device is None:
        return

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA device was requested, but PyTorch does not see a working CUDA runtime. "
            "Use '--device cpu' or reinstall a CUDA-enabled PyTorch build."
        )

    device_count = torch.cuda.device_count()
    if primary_device >= device_count:
        raise SystemExit(
            f"Requested CUDA device {primary_device}, but only {device_count} device(s) are visible."
        )

    capability = torch.cuda.get_device_capability(primary_device)
    arch_token = f"sm_{capability[0]}{capability[1]}"
    arch_list = set(torch.cuda.get_arch_list())

    if arch_list and arch_token not in arch_list:
        gpu_name = torch.cuda.get_device_name(primary_device)
        raise SystemExit(
            "This PyTorch build is incompatible with the selected GPU.\n"
            f"GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]}, {arch_token})\n"
            f"PyTorch CUDA architectures: {sorted(arch_list)}\n"
            "Install a PyTorch build that includes this GPU architecture, or run with '--device cpu'."
        )


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import RTDETR
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {args.data}\n"
            "Edit training/sokol_dataset.yaml or pass --data /path/to/data.yaml"
        )

    validate_torch_cuda_compatibility(args.device)

    model = RTDETR(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0001,
        cos_lr=True,
        close_mosaic=10,
        cache=False,
        plots=True,
        val=True,
        resume=args.resume,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training finished. Best weights: {best_model}")

    if args.export:
        trained = RTDETR(str(best_model))
        trained.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True)


if __name__ == "__main__":
    main()
