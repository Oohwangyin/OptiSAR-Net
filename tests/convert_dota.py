
#!/usr/bin/env python3
"""
convert_dota.py
将原始 DOTA 数据集转换为 YOLO 格式，只保留船只 (ship) 目标
输出目录结构:
  DOTA_new/
    images/
      train/  (图像复制)
      val/
      test/
    labels/
      train/  (对应 txt 文件，只包含 ship)
      val/
      test/
    dataset.yaml
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# ========== 配置参数 ==========
# 原始 DOTA 数据集路径
DOTA_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/DOTA"

# 输出路径
OUTPUT_ROOT = "/root/autodl-tmp/OptiSAR-Net/datasets/DOTA_new"

# 是否使用软链接（节省空间）还是复制文件
USE_SYMLINK = False  # 若为 True 则使用软链接，False 则复制文件

# 类别名称（只检测 ship）
CLASS_ID = 0
CLASS_NAME = "ship"

# ========== 路径设置 ==========
# 原始数据集路径
TRAIN_IMAGES_DIR = os.path.join(DOTA_ROOT, "train", "images")
TRAIN_LABELS_DIR = os.path.join(DOTA_ROOT, "train", "labels")
VAL_IMAGES_DIR = os.path.join(DOTA_ROOT, "val", "images")
VAL_LABELS_DIR = os.path.join(DOTA_ROOT, "val", "labels")
TEST_IMAGES_DIR = os.path.join(DOTA_ROOT, "test", "images")
TEST_JSON_FILE = os.path.join(DOTA_ROOT, "test", "test_info.json")
# 测试集标注文件（可能的路径）
TEST_ANNOTATIONS_FILE = os.path.join(DOTA_ROOT, "test", "test_annotations.json")

# 输出目录
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_ROOT, "labels")
OUTPUT_TRAIN_IMAGES = os.path.join(OUTPUT_IMAGES_DIR, "train")
OUTPUT_VAL_IMAGES = os.path.join(OUTPUT_IMAGES_DIR, "val")
OUTPUT_TEST_IMAGES = os.path.join(OUTPUT_IMAGES_DIR, "test")
OUTPUT_TRAIN_LABELS = os.path.join(OUTPUT_LABELS_DIR, "train")
OUTPUT_VAL_LABELS = os.path.join(OUTPUT_LABELS_DIR, "val")
OUTPUT_TEST_LABELS = os.path.join(OUTPUT_LABELS_DIR, "test")

for d in [OUTPUT_TRAIN_IMAGES, OUTPUT_VAL_IMAGES, OUTPUT_TEST_IMAGES,
          OUTPUT_TRAIN_LABELS, OUTPUT_VAL_LABELS, OUTPUT_TEST_LABELS]:
    os.makedirs(d, exist_ok=True)


# ========== 辅助函数 ==========
def parse_dota_label_file(label_path: str) -> List[Tuple[str, List[float]]]:
    """
    解析 DOTA 格式的标签文件
    返回 [(class_name, [x1, y1, x2, y2, x3, y3, x4, y4]), ...]
    """
    objects = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            # 前 8 个是坐标，第 9 个是类别，第 10 个是难度
            coords = [float(x) for x in parts[:8]]
            class_name = parts[8]
            # difficult = int(parts[9]) if len(parts) > 9 else 0

            objects.append((class_name, coords))
    except Exception as e:
        print(f"解析标签文件失败 {label_path}: {e}")

    return objects


def hbb_to_yolo(coords: List[float], img_width: int, img_height: int) -> Optional[Tuple[float, float, float, float]]:
    """
    将水平边界框 (HBB) 坐标转换为 YOLO 格式
    coords: [x1, y1, x2, y1, x2, y2, x1, y2] 或任意四边形坐标
    返回：[x_center, y_center, width, height] (归一化)
    """
    if len(coords) != 8:
        return None

    # 从四边形坐标中提取最小外接矩形
    x_coords = [coords[i] for i in range(0, 8, 2)]
    y_coords = [coords[i] for i in range(1, 8, 2)]

    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)

    # 计算水平和垂直边界框
    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        return None

    # 归一化到 [0, 1]
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    box_width = width / img_width
    box_height = height / img_height

    # 边界裁剪
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    box_width = max(0.0, min(1.0, box_width))
    box_height = max(0.0, min(1.0, box_height))

    return [x_center, y_center, box_width, box_height]


def get_image_size_from_file(img_path: str) -> Optional[Tuple[int, int]]:
    """获取图像尺寸"""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"无法读取图像尺寸 {img_path}: {e}")
        return None


def process_split(split_name: str,
                  src_images_dir: str,
                  src_labels_dir: str,
                  dst_images_dir: str,
                  dst_labels_dir: str) -> Tuple[int, int]:
    """
    处理一个数据集划分（train/val/test）
    返回：(处理的图像数，保留的船只目标数)
    """
    image_count = 0
    ship_count = 0

    # 获取所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    img_files = []
    for ext in img_extensions:
        img_files.extend(Path(src_images_dir).glob(f"*{ext}"))
        img_files.extend(Path(src_images_dir).glob(f"*{ext.upper()}"))

    print(f"\n处理 {split_name} 集...")
    print(f"找到 {len(img_files)} 张图像")

    for img_path in img_files:
        img_name = img_path.name
        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(src_labels_dir, label_name)

        # 复制/软链接图像
        dst_img_path = os.path.join(dst_images_dir, img_name)
        if USE_SYMLINK:
            if os.path.exists(dst_img_path):
                os.remove(dst_img_path)
            os.symlink(str(img_path), dst_img_path)
        else:
            shutil.copy2(str(img_path), dst_img_path)

        # 获取图像尺寸
        img_size = get_image_size_from_file(str(img_path))
        if img_size is None:
            continue

        img_width, img_height = img_size

        # 解析标签
        yolo_labels = []
        if os.path.exists(label_path):
            objects = parse_dota_label_file(label_path)

            for class_name, coords in objects:
                # 只保留 ship 类别
                if class_name.lower() == 'ship':
                    yolo_box = hbb_to_yolo(coords, img_width, img_height)
                    if yolo_box is not None:
                        yolo_labels.append((CLASS_ID, yolo_box))
                        ship_count += 1

        # 保存 YOLO 格式标签（即使为空也创建文件）
        dst_label_path = os.path.join(dst_labels_dir, label_name)
        with open(dst_label_path, 'w') as f:
            for cid, (xc, yc, w, h) in yolo_labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        
        # 如果标签为空，打印警告信息
        if not yolo_labels:
            print(f"  ⚠️  {img_name}: 没有检测到船只目标（创建空标签文件）")

        image_count += 1

        if image_count % 100 == 0:
            print(f"  已处理 {image_count}/{len(img_files)} 张图像")

    return image_count, ship_count


def process_test_set():
    """处理测试集（从 JSON 文件或标注文件创建标签）"""
    print(f"\n处理测试集...")
    
    # 首先尝试加载测试集标注
    annotations_dict = {}
    
    # 尝试读取标注文件
    if os.path.exists(TEST_ANNOTATIONS_FILE):
        print(f"找到测试集标注文件：{TEST_ANNOTATIONS_FILE}")
        try:
            with open(TEST_ANNOTATIONS_FILE, 'r') as f:
                annotations_data = json.load(f)
            
            # 解析标注（COCO 格式或 DOTA 格式）
            if 'annotations' in annotations_data:
                # COCO 格式
                for ann in annotations_data['annotations']:
                    img_id = ann.get('image_id')
                    if img_id not in annotations_dict:
                        annotations_dict[img_id] = []
                    
                    category = ann.get('category', ann.get('class', ''))
                    if isinstance(category, dict):
                        category = category.get('name', '')
                    
                    # 检查是否是 ship
                    if category.lower() == 'ship' or (isinstance(category, int) and category == 1):
                        # 获取边界框 [x, y, width, height] 或 多边形
                        bbox = ann.get('bbox', [])
                        segmentation = ann.get('segmentation', [])
                        
                        if bbox:
                            # 转换为四边形坐标
                            x, y, w, h = bbox
                            coords = [x, y, x+w, y, x+w, y+h, x, y+h]
                            annotations_dict[img_id].append(('ship', coords))
                        elif segmentation:
                            # 从分割掩码提取边界框
                            if isinstance(segmentation[0], list):
                                poly = segmentation[0]
                                x_coords = [poly[i] for i in range(0, len(poly), 2)]
                                y_coords = [poly[i] for i in range(1, len(poly), 2)]
                                xmin, xmax = min(x_coords), max(x_coords)
                                ymin, ymax = min(y_coords), max(y_coords)
                                coords = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                                annotations_dict[img_id].append(('ship', coords))
            else:
                # 可能是其他格式的标注
                print("标注文件格式未知，跳过标注加载")
                
        except Exception as e:
            print(f"读取测试集标注失败：{e}")
            annotations_dict = {}
    
    # 如果没有标注文件，检查是否有与 train/val 相同的 txt 标签文件
    test_labels_dir_candidate = os.path.join(DOTA_ROOT, "test", "labels")
    if not annotations_dict and os.path.exists(test_labels_dir_candidate):
        print(f"发现测试集 labels 目录：{test_labels_dir_candidate}")
        # 使用普通方式处理
        return process_split("test", TEST_IMAGES_DIR, test_labels_dir_candidate,
                           OUTPUT_TEST_IMAGES, OUTPUT_TEST_LABELS)
    
    # 复制图像并生成标签
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    img_files = []
    for ext in img_extensions:
        img_files.extend(Path(TEST_IMAGES_DIR).glob(f"*{ext}"))
        img_files.extend(Path(TEST_IMAGES_DIR).glob(f"*{ext.upper()}"))
    
    print(f"找到 {len(img_files)} 张测试图像")
    
    image_count = 0
    ship_count = 0
    
    for img_path in img_files:
        img_name = img_path.name
        label_name = Path(img_name).stem + ".txt"
        
        # 复制图像
        dst_img_path = os.path.join(OUTPUT_TEST_IMAGES, img_name)
        if USE_SYMLINK:
            if os.path.exists(dst_img_path):
                os.remove(dst_img_path)
            os.symlink(str(img_path), dst_img_path)
        else:
            shutil.copy2(str(img_path), dst_img_path)
        
        # 获取图像尺寸
        img_size = get_image_size_from_file(str(img_path))
        if img_size is None:
            continue
        
        img_width, img_height = img_size
        
        # 从 annotations_dict 中获取标注
        yolo_labels = []
        img_key = img_name.replace('.png', '').replace('.jpg', '')
        
        # 尝试不同的键名匹配
        if img_key not in annotations_dict:
            # 尝试使用完整的文件名
            img_key = Path(img_name).stem
        
        if img_key in annotations_dict:
            objects = annotations_dict.get(img_key, [])
            
            for class_name, coords in objects:
                if class_name.lower() == 'ship':
                    yolo_box = hbb_to_yolo(coords, img_width, img_height)
                    if yolo_box is not None:
                        yolo_labels.append((CLASS_ID, yolo_box))
                        ship_count += 1
        
        # 保存 YOLO 格式标签（即使为空也创建文件）
        dst_label_path = os.path.join(OUTPUT_TEST_LABELS, label_name)
        with open(dst_label_path, 'w') as f:
            for cid, (xc, yc, w, h) in yolo_labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        
        # 如果标签为空，打印警告信息
        if not yolo_labels:
            print(f"  ⚠️  {img_name}: 没有检测到船只目标（创建空标签文件）")
        
        image_count += 1
        
        if image_count % 100 == 0:
            print(f"  已处理 {image_count}/{len(img_files)} 张图像")
    
    if not annotations_dict:
        print("\n警告：测试集没有可用的标注文件，生成的测试集将不包含标签")
        print("提示：请将测试集标注文件放置在以下位置之一:")
        print(f"  1. {TEST_ANNOTATIONS_FILE} (COCO 格式)")
        print(f"  2. {test_labels_dir_candidate}/ (与 train/val 相同的 txt 格式)")
    
    return image_count, ship_count


def generate_dataset_yaml():
    """生成 dataset.yaml 配置文件"""
    yaml_content = f"""# DOTA dataset converted to YOLO format (ship only)
# Only ship class is retained from the original 15 classes

path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test  # optional, no labels

nc: 1
names: ['{CLASS_NAME}']

# Original DOTA classes (for reference):
# 0: plane, 1: ship, 2: storage tank, 3: baseball diamond,
# 4: tennis court, 5: basketball court, 6: ground track field,
# 7: harbor, 8: bridge, 9: large vehicle, 10: small vehicle,
# 11: helicopter, 12: roundabout, 13: soccer ball field,
# 14: swimming pool
"""

    yaml_path = os.path.join(OUTPUT_ROOT, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n生成配置文件：{yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("DOTA 数据集转换工具 (只保留船只目标)")
    print("=" * 60)
    print(f"\n源数据集：{DOTA_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"文件复制方式：{'软链接' if USE_SYMLINK else '复制'}")
    
    # 处理训练集
    train_imgs, train_ships = process_split(
        "train",
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        OUTPUT_TRAIN_IMAGES,
        OUTPUT_TRAIN_LABELS
    )
    print(f"训练集：{train_imgs} 张图像，{train_ships} 个船只目标")
    
    # 处理验证集
    val_imgs, val_ships = process_split(
        "val",
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        OUTPUT_VAL_IMAGES,
        OUTPUT_VAL_LABELS
    )
    print(f"验证集：{val_imgs} 张图像，{val_ships} 个船只目标")
    
    # 处理测试集
    test_imgs, test_ships = process_test_set()
    print(f"测试集：{test_imgs} 张图像，{test_ships} 个船只目标")
    
    # 生成配置文件
    generate_dataset_yaml()
    
    # 统计信息
    total_imgs = 0 + 0 + test_imgs
    total_ships = 0 + 0 + test_ships
    print("\n" + "=" * 60)
    print(f"转换完成！")
    print(f"总计：{total_imgs} 张图像，{total_ships} 个船只目标")
    print(f"输出目录：{OUTPUT_ROOT}")
    print("\n使用示例:")
    print(f"  yolo train data={os.path.join(OUTPUT_ROOT, 'dataset.yaml')} model=yolov8n.pt epochs=100 imgsz=640")
    print("=" * 60)


if __name__ == "__main__":
    main()
