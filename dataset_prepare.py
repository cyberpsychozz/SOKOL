from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any


DEFAULT_CLASS_NAMES = ("fire", "smoke")
SPLIT_NAME_MAP = {
    "train": "train",
    "val": "val",
    "test": "test",
}


def _copy_or_link_file(source: Path, destination: Path, use_symlinks: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if use_symlinks:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def _read_split_file(split_file: Path) -> list[str]:
    entries: list[str] = []

    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entries.append(Path(line).name)

    return entries


def _write_dataset_yaml(output_dir: Path, class_names: tuple[str, ...]) -> None:
    yaml_path = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    yaml_path.write_text(
        "\n".join(
            (
                f"path: {output_dir.resolve()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                names_block,
                "",
            )
        ),
        encoding="utf-8",
    )


def convert_fasdd_uav_to_yolo(
    source_root: Path,
    output_dir: Path,
    *,
    use_symlinks: bool = False,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> None:
    images_dir = source_root / "images"
    yolo_dir = source_root / "annotations" / "YOLO_UAV"
    labels_dir = yolo_dir / "labels"

    required_paths = (images_dir, yolo_dir, labels_dir)
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(f"Missing required dataset paths: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_source_name, split_output_name in SPLIT_NAME_MAP.items():
        split_file = yolo_dir / f"{split_source_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        split_images_dir = output_dir / split_output_name / "images"
        split_labels_dir = output_dir / split_output_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        missing_images: list[str] = []
        missing_labels: list[str] = []
        processed = 0

        for image_name in _read_split_file(split_file):
            image_source = images_dir / image_name
            label_source = labels_dir / f"{Path(image_name).stem}.txt"

            if not image_source.exists():
                missing_images.append(image_name)
                continue

            if not label_source.exists():
                missing_labels.append(label_source.name)
                continue

            _copy_or_link_file(
                image_source,
                split_images_dir / image_source.name,
                use_symlinks=use_symlinks,
            )
            _copy_or_link_file(
                label_source,
                split_labels_dir / label_source.name,
                use_symlinks=use_symlinks,
            )
            processed += 1

        print(
            f"[{split_output_name}] processed={processed} "
            f"missing_images={len(missing_images)} missing_labels={len(missing_labels)}"
        )

        if missing_images:
            print(f"First missing images: {missing_images[:5]}")
        if missing_labels:
            print(f"First missing labels: {missing_labels[:5]}")

    _write_dataset_yaml(output_dir, class_names)
    print(f"YOLO dataset prepared at: {output_dir.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for preprocessing frames and converting FASDD_UAV to YOLO layout."
    )
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser(
        "convert-fasdd-uav",
        help="Convert FASDD_UAV annotations into train/val/test YOLO folder structure.",
    )
    convert_parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("FASDD_UAV"),
        help="Path to the FASDD_UAV dataset root.",
    )
    convert_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/fasdd_uav_yolo"),
        help="Destination directory for the YOLO-structured dataset.",
    )
    convert_parser.add_argument(
        "--symlinks",
        action="store_true",
        help="Create symbolic links instead of copying files.",
    )
    convert_parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(DEFAULT_CLASS_NAMES),
        help="Class names written into dataset.yaml in index order.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "convert-fasdd-uav":
        convert_fasdd_uav_to_yolo(
            source_root=args.source_root,
            output_dir=args.output_dir,
            use_symlinks=args.symlinks,
            class_names=tuple(args.class_names),
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
