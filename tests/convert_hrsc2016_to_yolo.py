#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ============ 配置 ============
HRSC_DATASET_PATH = "/root/autodl-tmp/OptiSAR-Net/datasets/HRSC2016"   # 修改为你的路径
OUTPUT_PATH = "/root/autodl-tmp/OptiSAR-Net/datasets/HRSC2016_YOLO"    # 修改为你的输出路径
TRAIN_RATIO = 0.41
VAL_RATIO = 0.17
TEST_RATIO = 0.42
RANDOM_SEED = 42
USE_SYMLINK = True  # True: 创建软连接，False: 复制文件

# 类别映射（根据上述分析）
CLASS_ID_TO_L2 = {
    100000001: 0,   # 航母
    100000002: 1, 100000003: 1, 100000004: 1, 100000005: 1,
    100000006: 1, 100000007: 1, 100000008: 1, 100000009: 1,
    100000015: 1, 100000016: 1, 100000017: 1, 100000018: 1,
    100000019: 1, 100000020: 1, 100000024: 1, 100000025: 1,
    100000032: 1,
    100000010: 2, 100000011: 2, 100000012: 2, 100000013: 2,
    100000022: 3, 100000026: 3, 100000027: 3, 100000028: 3,
    100000029: 3, 100000030: 3,
}
L2_NAMES = {0: "aircraft_carrier", 1: "warship", 2: "merchant_ship", 3: "submarine"}

# ============ 辅助函数 ============
def get_image_path_from_xml(xml_path: str, image_dir: str) -> Optional[str]:
    """从XML中读取Img_FileName和Img_FileFmt，构建完整路径"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        file_name = root.find("Img_FileName")
        file_fmt = root.find("Img_FileFmt")
        if file_name is None or file_fmt is None:
            return None
        img_name = file_name.text
        img_fmt = file_fmt.text
        if not img_fmt.startswith('.'):
            img_fmt = '.' + img_fmt
        img_path = os.path.join(image_dir, img_name + img_fmt)
        if os.path.exists(img_path):
            return img_path
        return None
    except Exception:
        return None

def parse_xml(xml_path: str) -> Optional[Tuple[List[Tuple[int, List[float]]], str]]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        width_elem = root.find("Img_SizeWidth")
        height_elem = root.find("Img_SizeHeight")
        if width_elem is None or height_elem is None:
            return None
        img_w = int(width_elem.text)
        img_h = int(height_elem.text)

        boxes = []
        class_names = set()
        for obj in root.findall(".//HRSC_Object"):
            class_id_elem = obj.find("Class_ID")
            if class_id_elem is None:
                continue
            class_id = int(class_id_elem.text)
            if class_id not in CLASS_ID_TO_L2:
                continue  # 过滤未映射或不需要的ID（如1）
            l2_id = CLASS_ID_TO_L2[class_id]

            # 读取水平框
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

            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h
            # 裁剪到[0,1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            if width > 0 and height > 0:
                boxes.append((l2_id, [x_center, y_center, width, height]))
                class_names.add(L2_NAMES[l2_id])

        if not boxes:
            return None
        # 返回第一个类别名称（仅用于统计）
        return boxes, list(class_names)[0]
    except Exception as e:
        print(f"解析失败 {xml_path}: {e}")
        return None

def main():
    random.seed(RANDOM_SEED)
    annotations_dir = os.path.join(HRSC_DATASET_PATH, "Annotations")
    images_dir = os.path.join(HRSC_DATASET_PATH, "AllImages")

    output_images = os.path.join(OUTPUT_PATH, "images")
    output_labels = os.path.join(OUTPUT_PATH, "labels")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_images, split), exist_ok=True)
        os.makedirs(os.path.join(output_labels, split), exist_ok=True)

    # 收集有效样本
    valid_samples = []
    xml_files = list(Path(annotations_dir).glob("*.xml"))
    print(f"找到 {len(xml_files)} 个 XML 文件")
    
    # 用于统计每个类别的实例数量
    class_instance_count = {class_id: 0 for class_id in L2_NAMES.keys()}
    class_sample_count = {class_id: 0 for class_id in L2_NAMES.keys()}  # 包含该类别的样本数
    
    for xml_path in xml_files:
        result = parse_xml(str(xml_path))
        if result is None:
            continue
        boxes, class_name = result
        img_path = get_image_path_from_xml(str(xml_path), images_dir)
        if img_path is None:
            print(f"警告：找不到图像文件 {xml_path.name}")
            continue
        
        # 统计当前 XML 中每个类别的实例数量
        current_classes = set()
        for l2_id, _ in boxes:
            class_instance_count[l2_id] += 1
            current_classes.add(l2_id)
        
        # 统计包含该类别的样本数
        for l2_id in current_classes:
            class_sample_count[l2_id] += 1
        
        label_name = Path(img_path).stem + ".txt"
        label_content = "\n".join([f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cid, (x,y,w,h) in boxes])
        valid_samples.append({
            "img_path": img_path,
            "img_name": Path(img_path).name,
            "label_name": label_name,
            "label_content": label_content,
            "class_name": class_name,
        })

    print(f"有效图像数：{len(valid_samples)}")
    
    # 打印详细的实例统计信息
    print("\n" + "="*50)
    print("数据集实例统计详情:")
    print("="*50)
    total_instances = 0
    total_samples_with_objects = 0
    for class_id in sorted(L2_NAMES.keys()):
        class_name = L2_NAMES[class_id]
        inst_count = class_instance_count[class_id]
        samp_count = class_sample_count[class_id]
        total_instances += inst_count
        total_samples_with_objects += samp_count
        print(f"{class_name:20s}: 实例数={inst_count:4d}, 样本数={samp_count:4d}")
    
    print("-"*50)
    print(f"{'总计':20s}: 实例数={total_instances:4d}, 包含对象的样本数={total_samples_with_objects:4d}")
    print("="*50 + "\n")

    if len(valid_samples) == 0:
        return

    random.shuffle(valid_samples)
    train_idx = int(len(valid_samples) * TRAIN_RATIO)
    val_idx = int(len(valid_samples) * (TRAIN_RATIO + VAL_RATIO))
    train_samples = valid_samples[:train_idx]
    val_samples = valid_samples[train_idx:val_idx]
    test_samples = valid_samples[val_idx:]

    print(f"训练集：{len(train_samples)} 验证集：{len(val_samples)} 测试集：{len(test_samples)}")

    def copy_samples(samples, split):
        for s in samples:
            dst_img = os.path.join(output_images, split, s["img_name"])
            if USE_SYMLINK:
                # 创建软连接，节省空间
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.symlink(s["img_path"], dst_img)
            else:
                # 复制文件
                shutil.copy2(s["img_path"], dst_img)
            dst_lbl = os.path.join(output_labels, split, s["label_name"])
            with open(dst_lbl, 'w') as f:
                f.write(s["label_content"])

    copy_samples(train_samples, "train")
    copy_samples(val_samples, "val")
    copy_samples(test_samples, "test")
    
    # 统计各子集的实例数量
    def count_instances(samples):
        counts = {class_id: 0 for class_id in L2_NAMES.keys()}
        for s in samples:
            lines = s["label_content"].split("\n")
            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    counts[class_id] += 1
        return counts
    
    train_counts = count_instances(train_samples)
    val_counts = count_instances(val_samples)
    test_counts = count_instances(test_samples)
    
    print("\n" + "="*80)
    print("各子集实例分布详情:")
    print("="*80)
    print(f"{'类别':20s} | {'训练集':>10s} | {'验证集':>10s} | {'测试集':>10s} | {'总计':>10s}")
    print("-"*80)
    
    for class_id in sorted(L2_NAMES.keys()):
        class_name = L2_NAMES[class_id]
        train_cnt = train_counts[class_id]
        val_cnt = val_counts[class_id]
        test_cnt = test_counts[class_id]
        total_cnt = train_cnt + val_cnt + test_cnt
        print(f"{class_name:20s} | {train_cnt:10d} | {val_cnt:10d} | {test_cnt:10d} | {total_cnt:10d}")
    
    print("-"*80)
    total_all = sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())
    print(f"{'总计':20s} | {sum(train_counts.values()):10d} | {sum(val_counts.values()):10d} | {sum(test_counts.values()):10d} | {total_all:10d}")
    print("="*80 + "\n")

    # 生成YAML配置文件
    yaml_content = f"""path: {OUTPUT_PATH}
train: images/train
val: images/val
test: images/test

nc: {len(L2_NAMES)}
names: {list(L2_NAMES.values())}
"""
    with open(os.path.join(OUTPUT_PATH, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)

    print(f"\n转换完成！输出目录: {OUTPUT_PATH}")
    print(f"配置文件: {os.path.join(OUTPUT_PATH, 'dataset.yaml')}")

if __name__ == "__main__":
    main()