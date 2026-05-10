#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create single-class CORS-SAR aircraft datasets from the original two-class
CORS-SAR dataset.

Original labels:
  0: opt_plane
  1: sar_plane

Merged labels:
  0: aircraft

Outputs:
  CORS-SAR-aircraft          train/val/test mixed dataset
  CORS-SAR-aircraft-optical  source-only subset for validation/testing
  CORS-SAR-aircraft-sar      source-only subset for validation/testing
"""

import argparse
import os
import shutil
from pathlib import Path


SOURCE_ROOT = Path("/root/autodl-tmp/OptiSAR-Net/datasets/CORS-SAR")
OUTPUT_BASE = Path("/root/autodl-tmp/OptiSAR-Net/datasets")
MERGED_NAME = "CORS-SAR-aircraft"
OPTICAL_NAME = "CORS-SAR-aircraft-optical"
SAR_NAME = "CORS-SAR-aircraft-sar"
SUBSET_SPLIT = "test"

IMAGE_SUBDIR = "images"
LABEL_SUBDIR = "labels"
OPTICAL_PREFIX = "optical_"
SAR_PREFIX = "sar_"
MERGED_CLASS_ID = "0"
MERGED_CLASS_NAME = "aircraft"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    reset_path(dst)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rewrite_label_to_single_class(src_label: Path, dst_label: Path) -> int:
    reset_path(dst_label)
    count = 0
    with open(src_label, "r", encoding="utf-8") as src, open(dst_label, "w", encoding="utf-8") as dst:
        for line in src:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {src_label}: {line.rstrip()}")
            parts[0] = MERGED_CLASS_ID
            dst.write(" ".join(parts) + "\n")
            count += 1
    return count


def iter_images(images_dir: Path):
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
        yield from images_dir.glob(suffix)


def write_dataset_yaml(root: Path) -> None:
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# CORS-SAR single-class aircraft dataset",
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  0: {MERGED_CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_subset_yaml(root: Path) -> None:
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# CORS-SAR single-class aircraft source subset",
                f"path: {root}",
                "train: images",
                "val: images",
                "test: images",
                "names:",
                f"  0: {MERGED_CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def create_merged_dataset(source_root: Path, output_root: Path, copy_images: bool) -> None:
    total_images = 0
    total_instances = 0

    for split in ("train", "val", "test"):
        src_images_dir = source_root / IMAGE_SUBDIR / split
        src_labels_dir = source_root / LABEL_SUBDIR / split
        if not src_images_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {src_images_dir}")
        if not src_labels_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {src_labels_dir}")

        dst_images_dir = output_root / IMAGE_SUBDIR / split
        dst_labels_dir = output_root / LABEL_SUBDIR / split
        ensure_dir(dst_images_dir)
        ensure_dir(dst_labels_dir)

        split_images = sorted(iter_images(src_images_dir))
        split_instances = 0
        for src_img in split_images:
            src_label = src_labels_dir / f"{src_img.stem}.txt"
            if not src_label.exists():
                print(f"Warning: missing label, skipped image: {src_img}")
                continue

            dst_img = dst_images_dir / src_img.name
            dst_label = dst_labels_dir / src_label.name
            link_or_copy(src_img, dst_img, copy_images)
            split_instances += rewrite_label_to_single_class(src_label, dst_label)

        total_images += len(split_images)
        total_instances += split_instances
        print(f"[{output_root.name}/{split}] images={len(split_images)}, instances={split_instances}")

    write_dataset_yaml(output_root)
    print(f"[{output_root.name}] total_images={total_images}, total_instances={total_instances}")


def create_source_subset(
    merged_root: Path,
    output_root: Path,
    prefix: str,
    split: str,
    copy_images: bool,
) -> None:
    src_images_dir = merged_root / IMAGE_SUBDIR / split
    src_labels_dir = merged_root / LABEL_SUBDIR / split
    if not src_images_dir.exists():
        raise FileNotFoundError(f"Missing merged image directory: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Missing merged label directory: {src_labels_dir}")

    dst_images_dir = output_root / IMAGE_SUBDIR
    dst_labels_dir = output_root / LABEL_SUBDIR
    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    image_files = sorted(p for p in iter_images(src_images_dir) if p.name.startswith(prefix))
    instances = 0
    for src_img in image_files:
        src_label = src_labels_dir / f"{src_img.stem}.txt"
        if not src_label.exists():
            print(f"Warning: missing label, skipped image: {src_img}")
            continue

        dst_img = dst_images_dir / src_img.name
        dst_label = dst_labels_dir / src_label.name
        link_or_copy(src_img, dst_img, copy_images)
        shutil.copy2(src_label, dst_label)
        with open(src_label, "r", encoding="utf-8") as f:
            instances += sum(1 for line in f if line.strip())

    write_subset_yaml(output_root)
    print(f"[{output_root.name}] split={split}, images={len(image_files)}, instances={instances}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create single-class CORS-SAR aircraft datasets.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--merged-name", type=str, default=MERGED_NAME)
    parser.add_argument("--optical-name", type=str, default=OPTICAL_NAME)
    parser.add_argument("--sar-name", type=str, default=SAR_NAME)
    parser.add_argument("--subset-split", type=str, default=SUBSET_SPLIT, choices=("train", "val", "test"))
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_root = args.output_base / args.merged_name
    optical_root = args.output_base / args.optical_name
    sar_root = args.output_base / args.sar_name

    create_merged_dataset(args.source_root, merged_root, args.copy_images)
    create_source_subset(merged_root, optical_root, OPTICAL_PREFIX, args.subset_split, args.copy_images)
    create_source_subset(merged_root, sar_root, SAR_PREFIX, args.subset_split, args.copy_images)

    print("Done.")
    print(f"Mixed dataset:   {merged_root}")
    print(f"Optical subset:  {optical_root}")
    print(f"SAR subset:      {sar_root}")


if __name__ == "__main__":
    main()
