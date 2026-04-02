#!/usr/bin/env python3
"""
验证 YOLO 格式标注的正确性，并统计各类别实例数量。
用法:
    python verify_dataset.py --data_root ../datasets/M4-SAR-sampled --split train --num_samples 5
"""

import os
import random
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_yolo_label(label_path, img_w, img_h):
    """
    解析 YOLO 格式的标签文件，返回边界框列表。
    每行: class_id x_center y_center width height (归一化)
    返回: list of (class_id, x1, y1, x2, y2) 像素坐标
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"警告: 标签格式错误 (应为5个值): {line}")
                continue
            try:
                cls = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                # 转换为像素坐标
                x1 = int((x_c - w/2) * img_w)
                y1 = int((y_c - h/2) * img_h)
                x2 = int((x_c + w/2) * img_w)
                y2 = int((y_c + h/2) * img_h)
                # 边界裁剪
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    bboxes.append((cls, x1, y1, x2, y2))
            except ValueError:
                print(f"警告: 解析数值错误: {line}")
                continue
    return bboxes


def count_classes_in_split(data_root, split):
    """
    统计指定 split 中所有标签文件的类别实例数量。
    """
    label_dir = data_root / 'labels' / split
    if not label_dir.exists():
        print(f"错误: 标签目录不存在 {label_dir}")
        return defaultdict(int)

    class_counts = defaultdict(int)
    # 遍历所有标签文件（optical_xxx.txt 或 sar_xxx.txt，避免重复计数）
    # 我们只统计 optical_ 前缀的标签文件，因为内容与 sar 相同
    for label_file in label_dir.glob('optical_*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        cls = int(parts[0])
                        class_counts[cls] += 1
                    except ValueError:
                        continue
    return class_counts


def visualize_random_samples(data_root, split, num_samples=5, seed=42):
    """
    随机抽取 num_samples 张图像，绘制边界框并显示。
    """
    img_dir = data_root / 'images' / split
    label_dir = data_root / 'labels' / split
    if not img_dir.exists() or not label_dir.exists():
        print(f"错误: 图像或标签目录不存在: {img_dir} 或 {label_dir}")
        return

    # 获取所有光学图像文件
    image_files = list(img_dir.glob('optical_*.jpg')) + list(img_dir.glob('optical_*.png'))
    if not image_files:
        print(f"错误: 在 {img_dir} 中没有找到 optical_ 图像")
        return

    random.seed(seed)
    selected = random.sample(image_files, min(num_samples, len(image_files)))

    for img_path in selected:
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 对应的标签文件
        label_name = img_path.name.replace('optical_', 'optical_').replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = label_dir / label_name
        if not label_path.exists():
            print(f"标签文件不存在: {label_path}")
            continue

        bboxes = parse_yolo_label(label_path, w, h)

        # 绘制
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_rgb)
        for cls, x1, y1, x2, y2 in bboxes:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'Class {cls}', color='red', fontsize=10, backgroundcolor='white')
        ax.set_title(f"{img_path.name}  (共 {len(bboxes)} 个目标)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='验证 YOLO 标注并统计类别数量')
    parser.add_argument('--data_root', type=str, default='../datasets/M4-SAR-sampled_0.1',
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='要验证的数据集划分')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='随机可视化的图像数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"错误: 数据集根目录不存在: {data_root}")
        return

    print(f"数据集根目录: {data_root}")
    print(f"验证 split: {args.split}")

    # 统计类别数量
    print("\n正在统计各类别实例数量...")
    class_counts = count_classes_in_split(data_root, args.split)
    if not class_counts:
        print("未找到任何标签文件或没有有效标注。")
    else:
        print(f"\n{args.split} 集中各类别实例数量:")
        for cls_id in sorted(class_counts.keys()):
            print(f"  类别 {cls_id}: {class_counts[cls_id]} 个实例")
        total = sum(class_counts.values())
        print(f"  总计: {total} 个实例")

    # 可视化随机样本
    print(f"\n随机抽取 {args.num_samples} 张图像进行可视化...")
    visualize_random_samples(data_root, args.split, args.num_samples, args.seed)


if __name__ == '__main__':
    main()