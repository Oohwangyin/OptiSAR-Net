#!/usr/bin/env python3
"""
create_cors_optical_sar.py
从 CDHD 验证集中提取光学图像（HRSC2016）和 SAR 图像（HRSID）及其标注，
分别创建软链接到两个新目录，用于单独测试光学或 SAR 部分。
不复制实际文件，仅创建软链接以节省磁盘空间。
"""

import os
import shutil
from pathlib import Path

# ========== 配置路径 ==========
# CDHD 数据集根目录（请修改为你的实际路径）
CDHD_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/CORS-SAR"

# 输出根目录（两个子数据集将创建在此目录下）
OUTPUT_BASE = "/root/autodl-tmp/OptiSAR-Net/datasets"

# 子数据集名称
OPTICAL_OUTPUT = "CORS-SAR_optical"
SAR_OUTPUT = "CORS-SAR_sar"

# 子目录名称（与 CDHD 内部结构一致）
IMAGE_SUBDIR = "images"
LABEL_SUBDIR = "labels"

# 验证集子目录（假设 CDHD 中验证集文件夹名为 'val'）
VAL_IMAGES_DIR = os.path.join(CDHD_ROOT, IMAGE_SUBDIR, "val")
VAL_LABELS_DIR = os.path.join(CDHD_ROOT, LABEL_SUBDIR, "val")

# 文件名前缀区分光学和 SAR
OPTICAL_PREFIX = "optical_"
SAR_PREFIX = "sar_"

# ========== 辅助函数 ==========
def create_dataset(dataset_name, prefix, src_images_dir, src_labels_dir, output_base):
    """创建单个数据集（软链接）"""
    output_dir = os.path.join(output_base, dataset_name)
    img_dst_dir = os.path.join(output_dir, IMAGE_SUBDIR)
    lbl_dst_dir = os.path.join(output_dir, LABEL_SUBDIR)

    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    # 筛选匹配前缀的图像文件
    image_files = [f for f in os.listdir(src_images_dir) if f.startswith(prefix)]
    print(f"[{dataset_name}] 找到 {len(image_files)} 张图像")

    for img_file in image_files:
        src_img = os.path.join(src_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(src_labels_dir, label_file)

        dst_img = os.path.join(img_dst_dir, img_file)
        dst_label = os.path.join(lbl_dst_dir, label_file)

        # 删除已存在的链接（避免冲突）
        if os.path.exists(dst_img):
            os.remove(dst_img)
        if os.path.exists(dst_label):
            os.remove(dst_label)

        # 创建软链接
        os.symlink(src_img, dst_img)
        if os.path.exists(src_label):
            os.symlink(src_label, dst_label)
        else:
            print(f"  警告: 标注文件不存在 {src_label}")

    print(f"[{dataset_name}] 完成，输出目录: {output_dir}\n")

# ========== 主函数 ==========
def main():
    # 检查验证集目录是否存在
    if not os.path.exists(VAL_IMAGES_DIR):
        raise FileNotFoundError(f"验证集图像目录不存在: {VAL_IMAGES_DIR}")
    if not os.path.exists(VAL_LABELS_DIR):
        raise FileNotFoundError(f"验证集标签目录不存在: {VAL_LABELS_DIR}")

    # 创建光学数据集
    create_dataset(OPTICAL_OUTPUT, OPTICAL_PREFIX, VAL_IMAGES_DIR, VAL_LABELS_DIR, OUTPUT_BASE)

    # 创建 SAR 数据集
    create_dataset(SAR_OUTPUT, SAR_PREFIX, VAL_IMAGES_DIR, VAL_LABELS_DIR, OUTPUT_BASE)

    print("所有数据集创建完成！")
    print(f"光学测试集: {os.path.join(OUTPUT_BASE, OPTICAL_OUTPUT)}")
    print(f"SAR 测试集: {os.path.join(OUTPUT_BASE, SAR_OUTPUT)}")

if __name__ == "__main__":
    main()