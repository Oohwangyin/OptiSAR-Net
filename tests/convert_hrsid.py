#!/usr/bin/env python3
"""
convert_hrsid.py
将原始 HRSID 数据集（COCO 格式）转换为 YOLO 格式，使用原始的 train/test 划分。
输出目录结构：
  HRSID_new.yaml/
    images/
      train/  (图像软链接或复制)
      val/
    labels/
      train/  (对应 txt 文件)
      val/
    dataset.yaml
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ========== 配置参数 ==========
# 原始 HRSID 路径（包含 annotations, JPEGImages）
HRSID_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/HRSID_JPG"
# 输出路径
OUTPUT_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/HRSID_new.yaml"

# 是否使用软链接（节省空间）还是复制文件
USE_SYMLINK = True  # 若为 False 则复制文件

# 类别名称（二值检测，只有 ship）
CLASS_NAME = "ship"

# ========== 路径设置 ==========
IMAGES_DIR = os.path.join(HRSID_ROOT, "JPEGImages")
TRAIN_ANNOTATIONS_FILE = os.path.join(HRSID_ROOT, "annotations", "train2017.json")
TEST_ANNOTATIONS_FILE = os.path.join(HRSID_ROOT, "annotations", "test2017.json")

# 输出目录
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_ROOT, "labels")
TRAIN_IMAGES_DIR = os.path.join(OUTPUT_IMAGES_DIR, "train")
VAL_IMAGES_DIR = os.path.join(OUTPUT_IMAGES_DIR, "val")
TRAIN_LABELS_DIR = os.path.join(OUTPUT_LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(OUTPUT_LABELS_DIR, "val")

for d in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ========== 辅助函数 ==========
def load_coco_annotations(json_path: str) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]]]:
    """
    加载 COCO 格式的标注文件
    返回：
        - images_dict: {image_id: image_info}
        - anns_dict: {image_id: [annotation1, annotation2, ...]}
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像字典
    images_dict = {img['id']: img for img in coco_data.get('images', [])}
    
    # 构建标注字典（按 image_id 分组）
    anns_dict = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in anns_dict:
            anns_dict[img_id] = []
        anns_dict[img_id].append(ann)
    
    return images_dict, anns_dict


def coco_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    将 COCO 格式的 bbox [x_min, y_min, width, height] 
    转换为 YOLO 格式 [x_center, y_center, width, height]（归一化）
    """
    x_min, y_min, box_width, box_height = bbox
    
    # 计算中心点和尺寸
    x_center = (x_min + box_width / 2.0) / img_width
    y_center = (y_min + box_height / 2.0) / img_height
    norm_width = box_width / img_width
    norm_height = box_height / img_height
    
    # 边界裁剪
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))
    
    return [x_center, y_center, norm_width, norm_height]


def parse_image_annotations(
    image_info: Dict[str, Any],
    annotations: List[Dict[str, Any]]
) -> Optional[List[Tuple[int, List[float]]]]:
    """
    解析单个图像的所有标注，转换为 YOLO 格式
    返回列表，每个元素为 (class_id, [x_center, y_center, width, height])
    """
    img_width = image_info['width']
    img_height = image_info['height']
    
    boxes = []
    for ann in annotations:
        # HRSID 只有 1 个类别（ship），category_id 应该都是 0
        class_id = ann.get('category_id', 0)
        
        # COCO 格式中 category_id 从 1 开始，需要转换为从 0 开始
        # 如果数据集中只有一个类别，直接使用 0
        if class_id > 0:
            class_id = 0  # 强制转换为 0（二值检测）
        
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        # 检查 bbox 是否有效
        if bbox[2] <= 0 or bbox[3] <= 0:  # width 或 height 为 0 或负数
            continue
        
        # 转换为 YOLO 格式
        yolo_bbox = coco_bbox_to_yolo(bbox, img_width, img_height)
        
        if yolo_bbox[2] > 0 and yolo_bbox[3] > 0:  # 确保宽度和高度有效
            boxes.append((class_id, yolo_bbox))
    
    return boxes if boxes else None


def get_image_path(image_info: Dict[str, str]) -> Optional[str]:
    """根据图像信息获取完整的图像路径"""
    file_name = image_info.get('file_name', '')
    img_path = os.path.join(IMAGES_DIR, file_name)
    
    if os.path.exists(img_path):
        return img_path
    return None


def process_dataset(
    annotations_file: str, 
    split_name: str
) -> List[Tuple[str, str]]:
    """
    处理单个数据集（训练集或测试集）
    返回：[(img_path, label_content), ...]
    """
    print(f"正在加载 {split_name} 标注文件：{annotations_file}")
    images_dict, anns_dict = load_coco_annotations(annotations_file)
    
    print(f"找到 {len(images_dict)} 张 {split_name} 图像")
    
    valid_samples = []
    
    for img_id, image_info in images_dict.items():
        # 获取该图像的所有标注
        annotations = anns_dict.get(img_id, [])
        
        if not annotations:
            # 没有标注的图像跳过
            continue
        
        # 转换为 YOLO 格式
        boxes = parse_image_annotations(image_info, annotations)
        
        if boxes is None or len(boxes) == 0:
            # 没有有效目标框的图像跳过
            continue
        
        # 获取图像路径
        img_path = get_image_path(image_info)
        if img_path is None:
            print(f"警告：图像文件不存在 {image_info.get('file_name', '')}")
            continue
        
        # 生成标签内容
        label_lines = [f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cid, (x, y, w, h) in boxes]
        label_content = "\n".join(label_lines)
        
        valid_samples.append((img_path, label_content))
    
    print(f"{split_name} 有效图像（至少一个有效目标）: {len(valid_samples)}")
    return valid_samples


# ========== 主流程 ==========
def main():
    # 使用原始的 train/test 划分
    train_samples = process_dataset(TRAIN_ANNOTATIONS_FILE, "train")
    val_samples = process_dataset(TEST_ANNOTATIONS_FILE, "test")
    
    print(f"\n训练集：{len(train_samples)} 张")
    print(f"验证集：{len(val_samples)} 张")
    
    
    
    # 复制/软链接图像并写入标签
    def save_samples(samples, split_name):
        for img_path, label_content in samples:
            img_name = Path(img_path).name
            # 目标图像路径
            dst_img = os.path.join(OUTPUT_IMAGES_DIR, split_name, img_name)
            
            if USE_SYMLINK:
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.symlink(img_path, dst_img)
            else:
                shutil.copy2(img_path, dst_img)
            
            # 标签文件
            label_name = Path(img_name).stem + ".txt"
            dst_label = os.path.join(OUTPUT_LABELS_DIR, split_name, label_name)
            with open(dst_label, 'w') as f:
                f.write(label_content)
    
    save_samples(train_samples, "train")
    save_samples(val_samples, "val")
    
    # 生成 dataset.yaml
    yaml_content = f"""# HRSID dataset converted to YOLO format
# Using original train/test split from HRSID dataset
# Class: {CLASS_NAME}

path: {OUTPUT_ROOT}
train: images/train
val: images/val
# test: not included

nc: 1
names: ['{CLASS_NAME}']
"""
    yaml_path = os.path.join(OUTPUT_ROOT, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n转换完成！")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"配置文件：{yaml_path}")
    print("\n使用示例:")
    print(f"  yolo train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
