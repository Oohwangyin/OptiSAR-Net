#!/usr/bin/env python3
"""
Sample M4-SAR dataset while keeping optical/SAR pairs and class balance.

Original directory structure:
M4-SAR_original/
├── optical/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── sar/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/   (optional, may be empty or same as optical labels)

Output directory structure (OptiSAR-Net compatible):
M4-SAR_sampled/
├── images/
│   ├── train/
│   │   ├── optical_xxx.jpg
│   │   └── sar_xxx.jpg
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── optical_xxx.txt
    │   └── sar_xxx.txt
    ├── val/
    └── test/

Usage:
    python sample_m4sar.py --src /path/to/M4-SAR_original --dst /path/to/M4-SAR_sampled --ratio 0.5 --balance
"""

import os
import shutil
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

def parse_yolo_label(label_path):
    """
    Parse YOLO format label file and return list of class ids (integers).
    Each line: class_id x_center y_center width height
    """
    classes = []
    if not os.path.exists(label_path):
        return classes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    cls = int(parts[0])
                    classes.append(cls)
                except ValueError:
                    continue
    return classes

def gather_pairs(src_root, split):
    """
    Gather all optical-SAR pairs for a given split (train/val/test).
    Returns: list of tuples (optical_path, sar_path, label_path, class_counts)
    where class_counts is dict {class_id: count} for that image.
    """
    optical_img_dir = os.path.join(src_root, 'optical', 'images', split)
    sar_img_dir = os.path.join(src_root, 'sar', 'images', split)
    label_dir = os.path.join(src_root, 'optical', 'labels', split)

    if not os.path.isdir(optical_img_dir) or not os.path.isdir(sar_img_dir):
        print(f"Warning: Missing optical/sar images dir for split {split}, skipping.")
        return []

    pairs = []
    for fname in os.listdir(optical_img_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(fname)[0]
        optical_path = os.path.join(optical_img_dir, fname)
        sar_path = os.path.join(sar_img_dir, fname)
        label_path = os.path.join(label_dir, base + '.txt')

        # Ensure SAR image exists
        if not os.path.exists(sar_path):
            print(f"Warning: Missing SAR image for {fname}, skipping.")
            continue

        # Parse class counts
        class_counts = defaultdict(int)
        classes = parse_yolo_label(label_path)
        for c in classes:
            class_counts[c] += 1

        pairs.append((optical_path, sar_path, label_path, class_counts))

    return pairs

def balanced_sample(pairs, ratio, seed=42):
    """
    Greedy sampling to preserve class instance proportions.
    Returns: list of selected pairs.
    """
    random.seed(seed)
    # Compute total instances per class across all pairs
    total_instances = defaultdict(int)
    for _, _, _, cls_counts in pairs:
        for cls, cnt in cls_counts.items():
            total_instances[cls] += cnt

    # Target instances after sampling
    target_instances = {cls: int(cnt * ratio) for cls, cnt in total_instances.items()}
    if not target_instances:
        return []

    selected_indices = set()
    current_instances = defaultdict(int)
    pair_list = list(enumerate(pairs))  # (idx, pair)

    # Greedy selection: pick a pair that contains the class with largest deficit
    max_iter = len(pair_list) * 2
    for _ in range(max_iter):
        if len(selected_indices) >= int(len(pairs) * ratio):
            break
        # Find class with largest deficit
        deficits = {cls: max(0, target_instances[cls] - current_instances[cls]) for cls in target_instances}
        if all(d == 0 for d in deficits.values()):
            break
        max_def_class = max(deficits, key=deficits.get)
        # Candidates that contain this class
        candidates = [idx for idx, (_, _, _, cls_counts) in pair_list
                      if idx not in selected_indices and max_def_class in cls_counts]
        if not candidates:
            # Fallback: any remaining pair
            candidates = [idx for idx, _ in pair_list if idx not in selected_indices]
        if not candidates:
            break
        chosen = random.choice(candidates)
        selected_indices.add(chosen)
        _, _, _, cls_counts = pairs[chosen]
        for cls, cnt in cls_counts.items():
            current_instances[cls] += cnt

    # If still need more, add random pairs to fill target count
    needed = int(len(pairs) * ratio) - len(selected_indices)
    if needed > 0:
        remaining = [idx for idx, _ in pair_list if idx not in selected_indices]
        if remaining:
            extra = random.sample(remaining, min(needed, len(remaining)))
            selected_indices.update(extra)

    return [pairs[i] for i in selected_indices]

def random_sample(pairs, ratio, seed=42):
    """Simple random sampling."""
    random.seed(seed)
    n = int(len(pairs) * ratio)
    indices = random.sample(range(len(pairs)), n)
    return [pairs[i] for i in indices]

def copy_pairs(pairs, src_root, dst_root, split, prefix_optical='optical_', prefix_sar='sar_'):
    """
    Copy selected pairs to destination.
    Optical and SAR images are placed in images/{split}/ with prefixes.
    Labels are duplicated for both optical and SAR, placed in labels/{split}/.
    """
    dst_img_dir = os.path.join(dst_root, 'images', split)
    dst_label_dir = os.path.join(dst_root, 'labels', split)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    for optical_path, sar_path, label_path, _ in tqdm(pairs, desc=f"Copying {split}"):
        # Get base filename
        base = os.path.splitext(os.path.basename(optical_path))[0]

        # Copy optical image
        dst_optical_img = os.path.join(dst_img_dir, f"{prefix_optical}{base}.jpg")
        shutil.copy2(optical_path, dst_optical_img)

        # Copy SAR image
        dst_sar_img = os.path.join(dst_img_dir, f"{prefix_sar}{base}.jpg")
        shutil.copy2(sar_path, dst_sar_img)

        # Copy label file (two copies)
        if os.path.exists(label_path):
            dst_optical_label = os.path.join(dst_label_dir, f"{prefix_optical}{base}.txt")
            dst_sar_label = os.path.join(dst_label_dir, f"{prefix_sar}{base}.txt")
            shutil.copy2(label_path, dst_optical_label)
            shutil.copy2(label_path, dst_sar_label)

def main():
    parser = argparse.ArgumentParser(description="Sample M4-SAR dataset with pairing and balance")
    parser.add_argument('--src', default='../datasets/M4-SAR', help='Source dataset root directory')
    parser.add_argument('--dst', default='../datasets/M4-SAR-sampled', help='Destination dataset root directory')
    parser.add_argument('--ratio', type=float, default=0.5, help='Sampling ratio (default: 0.5)')
    parser.add_argument('--balance', dest='balance', action='store_true', default=True,
                        help='Enable class-balanced sampling (default: enabled)')
    parser.add_argument('--no-balance', dest='balance', action='store_false',
                        help='Disable class-balanced sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\nProcessing {split} split...")
        pairs = gather_pairs(args.src, split)
        if not pairs:
            print(f"No valid pairs found in {split}, skipping.")
            continue

        print(f"Found {len(pairs)} image pairs.")
        if args.balance:
            sampled = balanced_sample(pairs, args.ratio, args.seed)
        else:
            sampled = random_sample(pairs, args.ratio, args.seed)

        print(f"Selected {len(sampled)} pairs (target {int(len(pairs)*args.ratio)}).")

        copy_pairs(sampled, args.src, args.dst, split)

    print("\nDone.")

if __name__ == '__main__':
    main()