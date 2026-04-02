#!/usr/bin/env python3
"""
convert_hrsc.py
将原始 HRSC2016 数据集（FullDataSet）转换为 YOLO 格式，并按 4.1:1.7:4.2 划分训练/验证/测试集。
输出目录结构：
  HRSC2016_new/
    images/
      train/  (图像软链接或复制)
      val/
      test/
    labels/
      train/  (对应 txt 文件)
      val/
      test/
    dataset.yaml
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

# ========== 配置参数 ==========
# 原始 HRSC2016 路径（包含 AllImages, Annotations, ImageSets）
HRSC_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/HRSC2016"
# 输出路径
OUTPUT_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/HRSC2016_new"

# 划分比例
TRAIN_RATIO = 0.41
VAL_RATIO = 0.17
TEST_RATIO = 0.42
RANDOM_SEED = 42

# 是否使用软链接（节省空间）还是复制文件
USE_SYMLINK = True  # 若为 False 则复制文件

# 类别名称（二值检测，只有 ship）
CLASS_ID = 0
CLASS_NAME = "ship"

# ========== 路径设置 ==========
ALL_IMAGES_DIR = os.path.join(HRSC_ROOT, "AllImages")
ANNOTATIONS_DIR = os.path.join(HRSC_ROOT, "Annotations")
# ImageSets 中的划分文件（官方划分），但我们将不使用官方划分，而是自定义随机划分
# 因此这里不需要读取 ImageSets

# 输出目录
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
LABELS_DIR = os.path.join(OUTPUT_ROOT, "labels")
TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")
TEST_IMAGES_DIR = os.path.join(IMAGES_DIR, "test")
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")
TEST_LABELS_DIR = os.path.join(LABELS_DIR, "test")

for d in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 辅助函数 ==========
def get_image_path_from_xml(xml_path: str) -> Optional[str]:
    """从 XML 中读取图像文件名和格式，返回完整路径"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        file_name_elem = root.find("Img_FileName")
        file_fmt_elem = root.find("Img_FileFmt")
        if file_name_elem is None or file_fmt_elem is None:
            return None
        img_name = file_name_elem.text
        img_fmt = file_fmt_elem.text
        if not img_fmt.startswith('.'):
            img_fmt = '.' + img_fmt
        img_path = os.path.join(ALL_IMAGES_DIR, img_name + img_fmt)
        if os.path.exists(img_path):
            return img_path
        return None
    except Exception:
        return None

def parse_xml(xml_path: str, img_width: int, img_height: int) -> List[Tuple[int, List[float]]]:
    """
    解析 XML，提取所有目标并转换为 YOLO 归一化水平框坐标。
    返回列表，每个元素为 (class_id, [x_center, y_center, width, height])
    """
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall(".//HRSC_Object"):
            # 检查是否 difficult（可选过滤，此处不过滤）
            difficult_elem = obj.find("difficult")
            if difficult_elem is not None and int(difficult_elem.text) == 1:
                # 可选：跳过困难样本，这里保留
                pass

            # 获取水平框坐标
            xmin_elem = obj.find("box_xmin")
            ymin_elem = obj.find("box_ymin")
            xmax_elem = obj.find("box_xmax")
            ymax_elem = obj.find("box_ymax")
            if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                continue
            xmin = float(xmin_elem.text)
            ymin = float(ymin_elem.text)
            xmax = float(xmax_elem.text)
            ymax = float(ymax_elem.text)

            # 归一化
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 边界裁剪
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            if width > 0 and height > 0:
                boxes.append((CLASS_ID, [x_center, y_center, width, height]))
    except Exception as e:
        print(f"解析 XML 失败 {xml_path}: {e}")
    return boxes

def process_xml(xml_path: str) -> Optional[Tuple[str, List[Tuple[int, List[float]]]]]:
    """处理单个 XML，返回 (图像路径, boxes列表) 或 None"""
    # 获取图像尺寸
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        width_elem = root.find("Img_SizeWidth")
        height_elem = root.find("Img_SizeHeight")
        if width_elem is None or height_elem is None:
            return None
        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
    except Exception:
        return None

    boxes = parse_xml(xml_path, img_width, img_height)
    if not boxes:
        return None

    img_path = get_image_path_from_xml(xml_path)
    if img_path is None:
        return None

    return img_path, boxes

# ========== 主流程 ==========
def main():
    random.seed(RANDOM_SEED)

    # 收集所有有效样本
    valid_samples = []  # 每个元素: (img_path, label_content)
    xml_files = list(Path(ANNOTATIONS_DIR).glob("*.xml"))
    print(f"找到 {len(xml_files)} 个 XML 文件")

    for xml_path in xml_files:
        result = process_xml(str(xml_path))
        if result is None:
            continue
        img_path, boxes = result
        # 生成标签内容
        label_lines = [f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cid, (x, y, w, h) in boxes]
        label_content = "\n".join(label_lines)
        valid_samples.append((img_path, label_content))

    print(f"有效图像（至少一个有效目标）: {len(valid_samples)}")

    if len(valid_samples) == 0:
        print("错误：没有找到有效样本")
        return

    # 随机打乱并划分
    random.shuffle(valid_samples)
    train_idx = int(len(valid_samples) * TRAIN_RATIO)
    val_idx = int(len(valid_samples) * (TRAIN_RATIO + VAL_RATIO))
    train_samples = valid_samples[:train_idx]
    val_samples = valid_samples[train_idx:val_idx]
    test_samples = valid_samples[val_idx:]

    print(f"训练集：{len(train_samples)} 张")
    print(f"验证集：{len(val_samples)} 张")
    print(f"测试集：{len(test_samples)} 张")

    # 复制/软链接图像并写入标签
    def save_samples(samples, split_name):
        for img_path, label_content in samples:
            img_name = Path(img_path).name
            # 目标图像路径
            dst_img = os.path.join(IMAGES_DIR, split_name, img_name)
            if USE_SYMLINK:
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.symlink(img_path, dst_img)
            else:
                shutil.copy2(img_path, dst_img)

            # 标签文件
            label_name = Path(img_name).stem + ".txt"
            dst_label = os.path.join(LABELS_DIR, split_name, label_name)
            with open(dst_label, 'w') as f:
                f.write(label_content)

    save_samples(train_samples, "train")
    save_samples(val_samples, "val")
    save_samples(test_samples, "test")

    # 生成 dataset.yaml
    yaml_content = f"""# HRSC2016 dataset converted to YOLO format
# Train/Val/Test split: {TRAIN_RATIO*100:.1f}% / {VAL_RATIO*100:.1f}% / {TEST_RATIO*100:.1f}%
# Class: {CLASS_NAME}

path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['{CLASS_NAME}']
"""
    yaml_path = os.path.join(OUTPUT_ROOT, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n转换完成！")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"配置文件: {yaml_path}")
    print("\n使用示例:")
    print(f"  yolo train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")

if __name__ == "__main__":
    main()