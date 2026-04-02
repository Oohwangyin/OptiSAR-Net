#!/usr/bin/env python3
"""
从已整理好的 M4-SAR 数据集中按比例采样，保持类别平衡，使用软链接。
目录结构假设：
M4-SAR/
├── images/
│   ├── train/   (optical_xxx.jpg, sar_xxx.jpg)
│   ├── val/
│   └── test/
└── labels/
    ├── train/   (optical_xxx.txt, sar_xxx.txt)
    ├── val/
    └── test/
用法: python sample_m4sar_balanced.py --fraction 0.3
"""

import os
import sys
import argparse
import random
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def parse_label(label_path):
    """
    解析 YOLO HBB 格式的标签文件，返回类别ID列表。
    每行: class_id x_center y_center width height
    """
    classes = []
    if not os.path.exists(label_path):
        return classes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    cls = int(parts[0])
                    classes.append(cls)
                except ValueError:
                    continue
    return classes


def gather_pairs(src_root, split):
    """
    收集指定 split 中的所有光学-SAR 图像对。
    假设图像在 images/{split}/ 下，标签在 labels/{split}/ 下，
    配对依据：去掉 'optical_' 或 'sar_' 前缀后基础名相同。
    返回: 列表，元素为 (optical_path, sar_path, label_path, class_counts)
    """
    img_dir = src_root / 'images' / split
    label_dir = src_root / 'labels' / split

    if not img_dir.exists() or not label_dir.exists():
        print(f"警告: {split} 的 images 或 labels 目录不存在，跳过")
        return []

    pairs = []
    # 遍历所有 optical_ 图像
    for fname in os.listdir(img_dir):
        if not fname.startswith('optical_'):
            continue
        base = fname[8:]  # 去掉 'optical_' 前缀，保留原始文件名，例如 "12345.jpg"
        # 对应的 SAR 图像
        sar_fname = f'sar_{base}'
        optical_path = img_dir / fname
        sar_path = img_dir / sar_fname
        if not sar_path.exists():
            print(f"警告: 找不到对应的 SAR 图像 {sar_fname}，跳过 {fname}")
            continue
        # 标签文件（使用 optical 的标签，内容与 sar 相同）
        label_fname = f'optical_{base}'  # 标签文件名与光学图像同名但扩展名为 .txt
        label_fname = os.path.splitext(label_fname)[0] + '.txt'
        label_path = label_dir / label_fname
        if not label_path.exists():
            # 尝试不带 optical_ 前缀的标签？根据你的结构，应该是带前缀的
            print(f"警告: 找不到标签文件 {label_fname}，跳过 {fname}")
            continue

        class_counts = defaultdict(int)
        classes = parse_label(label_path)
        for c in classes:
            class_counts[c] += 1

        pairs.append((optical_path, sar_path, label_path, class_counts))

    return pairs


def balanced_sample(pairs, fraction, seed=42):
    """
    贪心采样，保持各类别实例的比例。
    """
    random.seed(seed)
    if fraction <= 0 or fraction >= 1:
        print("错误: fraction 必须在 (0,1) 范围内")
        sys.exit(1)

    target_count = int(len(pairs) * fraction)

    total_instances = defaultdict(int)
    for _, _, _, cls_counts in pairs:
        for cls, cnt in cls_counts.items():
            total_instances[cls] += cnt

    target_instances = {cls: int(cnt * fraction) for cls, cnt in total_instances.items()}
    if not target_instances:
        return []

    selected_indices = set()
    current_instances = defaultdict(int)
    pair_list = list(enumerate(pairs))

    for _ in range(target_count * 2):
        if len(selected_indices) >= target_count:
            break
        deficits = {cls: max(0, target_instances[cls] - current_instances[cls])
                    for cls in target_instances}
        if all(d == 0 for d in deficits.values()):
            break
        max_def_class = max(deficits, key=deficits.get)
        candidates = [idx for idx, (_, _, _, cls_counts) in pair_list
                      if idx not in selected_indices and max_def_class in cls_counts]
        if not candidates:
            candidates = [idx for idx, _ in pair_list if idx not in selected_indices]
        if not candidates:
            break
        chosen = random.choice(candidates)
        selected_indices.add(chosen)
        _, _, _, cls_counts = pairs[chosen]
        for cls, cnt in cls_counts.items():
            current_instances[cls] += cnt

    if len(selected_indices) < target_count:
        remaining = [idx for idx, _ in pair_list if idx not in selected_indices]
        needed = target_count - len(selected_indices)
        if remaining:
            extra = random.sample(remaining, min(needed, len(remaining)))
            selected_indices.update(extra)

    return [pairs[i] for i in selected_indices]


def create_symlinks(pairs, src_root, dst_root, split):
    """
    为选中的图像对创建软链接，保持与原始数据集相同的目录结构。
    """
    dst_img_dir = dst_root / 'images' / split
    dst_label_dir = dst_root / 'labels' / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for optical_path, sar_path, label_path, _ in tqdm(pairs, desc=f"创建 {split} 软链接"):
        # 光学图像
        optical_dst = dst_img_dir / optical_path.name
        if not optical_dst.exists():
            os.symlink(optical_path, optical_dst)
        # SAR 图像
        sar_dst = dst_img_dir / sar_path.name
        if not sar_dst.exists():
            os.symlink(sar_path, sar_dst)
        # 标签文件（optical）
        label_dst = dst_label_dir / label_path.name
        if not label_dst.exists():
            os.symlink(label_path, label_dst)
        
        # 新增：为 SAR 图像创建对应的标签文件软链接
        # 从 optical 标签文件名生成对应的 sar 标签文件名
        sar_label_name = label_path.name.replace('optical_', 'sar_', 1)
        sar_label_dst = dst_label_dir / sar_label_name
        if not sar_label_dst.exists():
            os.symlink(label_path, sar_label_dst)


def main():
    parser = argparse.ArgumentParser(description='从 M4-SAR 采样并创建软链接（保持类别平衡）')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='采样比例，例如 0.3 表示保留 30% 的数据')
    parser.add_argument('--src', type=str, default='../datasets/M4-SAR-sampled',
                        help='原始数据集路径 (默认：../datasets/M4-SAR-sampled)')
    parser.add_argument('--dst', type=str, default=None,
                        help='输出数据集路径 (默认：../datasets/M4-SAR-sampled_{fraction})')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于可复现采样 (默认：42)')
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    
    # 如果未指定 dst，则使用 "原始名称_fraction" 格式
    if args.dst is None:
        src_name = Path(args.src).name  # 获取源目录名，例如 "M4-SAR-sampled"
        args.dst = f'../datasets/{src_name}_{args.fraction}'
    
    dst_root = Path(args.dst).resolve()

    if not src_root.exists():
        print(f"错误：源数据集路径不存在：{src_root}")
        sys.exit(1)

    print(f"源数据集：{src_root}")
    print(f"目标数据集：{dst_root}")
    print(f"采样比例：{args.fraction}")

    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n处理 {split} 集...")
        pairs = gather_pairs(src_root, split)
        if not pairs:
            print(f"警告: {split} 中没有找到有效的图像对，跳过")
            continue
        print(f"找到 {len(pairs)} 对图像")
        sampled = balanced_sample(pairs, args.fraction, args.seed)
        print(f"采样后: {len(sampled)} 对 (目标: {int(len(pairs) * args.fraction)})")
        create_symlinks(sampled, src_root, dst_root, split)

    print(f"\n完成！采样后的数据集位于: {dst_root}")
    print("注意: 使用软链接，原始数据集请勿移动或删除。")


if __name__ == '__main__':
    main()