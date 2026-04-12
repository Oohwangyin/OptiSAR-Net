#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集CORS-ADD和SAR-AIRcraft-1.0重新组织为CORS-SAR
CORS-SAR格式与CDHD相似
用于进行模型的双源飞机识别
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ==================== CONFIGURATION ====================
# 原始数据集路径（请根据实际位置修改）
CORS_ADD_ROOT = "../datasets/CORS-ADD"          # 包含 images/, labels/, train2017.json, val2017.json
SAR_AIRCRAFT_ROOT = "../datasets/SAR-AIRcraft-1.0"  # 包含 Annotations/, ImageSets/, JPEGImages/

# 输出路径
OUTPUT_ROOT = "../datasets/CORS-SAR"            # 最终生成的数据集根目录

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# 随机种子（保证可复现）
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 辅助函数 ====================
def safe_mkdir(path):
    """创建目录（如果不存在）"""
    Path(path).mkdir(parents=True, exist_ok=True)

def convert_tif_to_jpg(tif_path, jpg_path):
    """将TIFF图像转换为JPEG格式"""
    img = Image.open(tif_path)
    # 如果图像是RGBA模式，转为RGB
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(jpg_path, "JPEG", quality=95)

def copy_jpg(src, dst):
    """复制JPEG文件"""
    shutil.copy2(src, dst)

def parse_cors_add_labels(txt_path):
    """
    解析CORS-ADD的txt标注文件
    格式：class_id x_center y_center width height (归一化)
    返回列表，每个元素为 [class_id, x_center, y_center, width, height]
    """
    labels = []
    if not os.path.exists(txt_path):
        return labels
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            # 光学图像中的飞机类别设为0
            labels.append([0, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return labels

def parse_sar_aircraft_xml(xml_path):
    """
    解析SAR-AIRcraft-1.0的XML标注文件
    返回列表，每个元素为 [class_id, x_center, y_center, width, height] 归一化坐标
    """
    labels = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            return labels
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        for obj in root.findall("object"):
            name = obj.find("name").text
            # SAR图像中的飞机类别设为1
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 转换为YOLO归一化格式 (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 确保坐标在[0,1]内（避免浮点误差）
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            width = np.clip(width, 0.0, 1.0)
            height = np.clip(height, 0.0, 1.0)

            labels.append([1, x_center, y_center, width, height])
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return labels

def collect_cors_add_images(root_dir):
    """
    收集CORS-ADD数据集中所有有效图像（有标注）
    返回列表，每个元素为 (img_path, labels_list)
    """
    images_dir = Path(root_dir) / "images"
    labels_dir = Path(root_dir) / "labels"

    valid_items = []
    # 遍历 train2017 和 val2017
    for split in ["train2017", "val2017"]:
        img_split_dir = images_dir / split
        label_split_dir = labels_dir / split
        if not img_split_dir.exists() or not label_split_dir.exists():
            print(f"Warning: {img_split_dir} or {label_split_dir} not found, skipping.")
            continue
        # 遍历所有tif图片
        tif_files = list(img_split_dir.glob("*.tif"))
        for tif_path in tqdm(tif_files, desc=f"CORS-ADD {split}"):
            txt_path = label_split_dir / (tif_path.stem + ".txt")
            labels = parse_cors_add_labels(txt_path)
            if len(labels) > 0:  # 只保留有标注的图像
                valid_items.append((str(tif_path), labels))
    return valid_items

def collect_sar_aircraft_images(root_dir):
    """
    收集SAR-AIRcraft-1.0数据集中所有有效图像（有标注）
    返回列表，每个元素为 (img_path, labels_list)
    """
    jpeg_dir = Path(root_dir) / "JPEGImages"
    annot_dir = Path(root_dir) / "Annotations"

    valid_items = []
    # 遍历所有jpg图片
    jpg_files = list(jpeg_dir.glob("*.jpg"))
    for jpg_path in tqdm(jpg_files, desc="SAR-AIRcraft-1.0"):
        xml_path = annot_dir / (jpg_path.stem + ".xml")
        if not xml_path.exists():
            continue
        labels = parse_sar_aircraft_xml(xml_path)
        if len(labels) > 0:  # 只保留有标注的图像
            valid_items.append((str(jpg_path), labels))
    return valid_items

def write_labels_file(labels, out_path):
    """将标签列表写入YOLO格式txt文件"""
    with open(out_path, 'w') as f:
        for label in labels:
            # label: [class, x, y, w, h]
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def split_and_balance(optical_items, sar_items, train_ratio, val_ratio, test_ratio):
    """
    将光学和SAR数据分别按比例划分，然后平衡每个子集的数量（取较小的数量）
    返回字典: {'train': (opt_list, sar_list), 'val': (...), 'test': (...)}
    """
    random.shuffle(optical_items)
    random.shuffle(sar_items)

    n_opt = len(optical_items)
    n_sar = len(sar_items)

    # 计算各划分的原始数量
    n_train_opt = int(n_opt * train_ratio)
    n_val_opt = int(n_opt * val_ratio)
    n_test_opt = n_opt - n_train_opt - n_val_opt

    n_train_sar = int(n_sar * train_ratio)
    n_val_sar = int(n_sar * val_ratio)
    n_test_sar = n_sar - n_train_sar - n_val_sar

    # 分割原始列表
    opt_train = optical_items[:n_train_opt]
    opt_val = optical_items[n_train_opt:n_train_opt+n_val_opt]
    opt_test = optical_items[n_train_opt+n_val_opt:]

    sar_train = sar_items[:n_train_sar]
    sar_val = sar_items[n_train_sar:n_train_sar+n_val_sar]
    sar_test = sar_items[n_train_sar+n_val_sar:]

    # 平衡每个子集：取较小数量，随机截取
    balanced = {}
    for split_name, opt_list, sar_list in [('train', opt_train, sar_train),
                                            ('val', opt_val, sar_val),
                                            ('test', opt_test, sar_test)]:
        min_len = min(len(opt_list), len(sar_list))
        if len(opt_list) > min_len:
            opt_list = random.sample(opt_list, min_len)
        if len(sar_list) > min_len:
            sar_list = random.sample(sar_list, min_len)
        balanced[split_name] = (opt_list, sar_list)

    return balanced

def reorganize_dataset(cors_root, sar_root, output_root):
    """
    主流程：收集、划分、复制并转换格式
    """
    # 1. 收集有效数据
    print("Collecting CORS-ADD (optical) images...")
    optical_items = collect_cors_add_images(cors_root)
    print(f"Found {len(optical_items)} valid optical images with annotations.")

    print("\nCollecting SAR-AIRcraft-1.0 (SAR) images...")
    sar_items = collect_sar_aircraft_images(sar_root)
    print(f"Found {len(sar_items)} valid SAR images with annotations.")

    if len(optical_items) == 0 or len(sar_items) == 0:
        print("Error: No valid images found in one of the datasets. Check paths.")
        return

    # 2. 划分并平衡
    print("\nSplitting and balancing datasets...")
    balanced_splits = split_and_balance(optical_items, sar_items,
                                        TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # 3. 创建输出目录
    for split in ['train', 'val', 'test']:
        safe_mkdir(Path(output_root) / "images" / split)
        safe_mkdir(Path(output_root) / "labels" / split)

    # 4. 处理每个划分
    for split, (opt_list, sar_list) in balanced_splits.items():
        print(f"\nProcessing {split} split: {len(opt_list)} optical, {len(sar_list)} SAR")
        img_counter = 1

        # 处理光学图像
        for idx, (img_path, labels) in enumerate(opt_list, start=1):
            # 新文件名: optical_xxx.jpg
            new_name = f"optical_{idx:06d}.jpg"
            dst_img = Path(output_root) / "images" / split / new_name
            dst_label = Path(output_root) / "labels" / split / (new_name.replace('.jpg', '.txt'))

            # 转换并保存图像（tif -> jpg）
            if img_path.lower().endswith('.tif'):
                convert_tif_to_jpg(img_path, dst_img)
            else:
                copy_jpg(img_path, dst_img)

            # 保存标签
            write_labels_file(labels, dst_label)

        # 处理SAR图像
        for idx, (img_path, labels) in enumerate(sar_list, start=1):
            new_name = f"sar_{idx:06d}.jpg"
            dst_img = Path(output_root) / "images" / split / new_name
            dst_label = Path(output_root) / "labels" / split / (new_name.replace('.jpg', '.txt'))

            # 复制图像（已经是jpg）
            copy_jpg(img_path, dst_img)

            # 保存标签
            write_labels_file(labels, dst_label)

    # 5. 生成数据集描述文件
    with open(Path(output_root) / "dataset.yaml", 'w') as f:
        f.write(f"# CORS-SAR dataset for multi-source aircraft detection\n")
        f.write(f"path: {output_root}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"nc: 2\n")
        f.write(f"names: ['optical_aircraft', 'sar_aircraft']\n")

    print(f"\nDone! Dataset saved to {output_root}")
    print(f"Dataset YAML: {output_root}/dataset.yaml")

if __name__ == "__main__":
    # 请根据实际情况修改下面的路径
    # 示例：
    # CORS_ADD_ROOT = "/data/CORS-ADD"
    # SAR_AIRCRAFT_ROOT = "/data/SAR-AIRcraft-1.0"
    # OUTPUT_ROOT = "/data/CORS-SAR"

    reorganize_dataset(CORS_ADD_ROOT, SAR_AIRCRAFT_ROOT, OUTPUT_ROOT)