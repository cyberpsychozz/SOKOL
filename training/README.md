# RT-DETR-L training for SOKOL

1. Put the downloaded dataset in YOLO format, for example:

```text
datasets/sokol/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

2. Edit `training/sokol_dataset.yaml` so `path` and `names` match the dataset.

3. Start training:

```bash
python3 training/train_rtdetr_l.py --data training/sokol_dataset.yaml --epochs 100 --imgsz 640 --batch 8 --device 0
```

For a weaker GPU, reduce `--batch` to `2` or `4`. To export the best model to ONNX after training, add `--export`.

The best checkpoint will be saved to:

```text
training/runs/sokol_rtdetr_l/weights/best.pt
```

Use `RTDETR("training/runs/sokol_rtdetr_l/weights/best.pt")` for inference.
