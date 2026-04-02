import os
import shutil
import random
import xml.etree.ElementTree as ET
import json
from PIL import Image

# ==================== 配置参数 ====================
# 请根据实际路径修改
HRSC2016_ROOT = "datasets/HRSC2016/HRSC2016"           # 包含 FullDataSet, ImageSets 的根目录
HRSID_ROOT = "datasets/HRSID_JPG"             # 包含 JPEGImages 和 annotations 的根目录
OUTPUT_ROOT = "datasets/CDHD"                 # 输出目录

# 数量设置（论文中 1:1）
TRAIN_OPTICAL = 416
VAL_OPTICAL = 181
TRAIN_SAR = 416
VAL_SAR = 181

# 随机种子（保证可重复）
RANDOM_SEED = 42

# =================================================

def create_dirs():
    """创建输出目录结构"""
    dirs = [
        OUTPUT_ROOT,
        os.path.join(OUTPUT_ROOT, "images", "train"),
        os.path.join(OUTPUT_ROOT, "images", "val"),
        os.path.join(OUTPUT_ROOT, "labels", "train"),
        os.path.join(OUTPUT_ROOT, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def convert_xml_to_yolo(xml_path, img_width, img_height):
    """
    从 HRSC2016 XML 中提取水平框，返回 YOLO 格式的行列表。
    使用 <box_xmin>, <box_ymin>, <box_xmax>, <box_ymax>。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall(".//HRSC_Object")
    lines = []
    for obj in objects:
        xmin = float(obj.find("box_xmin").text)
        ymin = float(obj.find("box_ymin").text)
        xmax = float(obj.find("box_xmax").text)
        ymax = float(obj.find("box_ymax").text)
        # 转换为 YOLO 格式
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        # 边界限制（防止微小偏移）
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines

def process_hrsc2016():
    """处理 HRSC2016 光学数据集"""
    print("处理 HRSC2016...")
    img_dir = os.path.join(HRSC2016_ROOT, "FullDataSet", "AllImages")
    ann_dir = os.path.join(HRSC2016_ROOT, "FullDataSet", "Annotations")
    train_list_file = os.path.join(HRSC2016_ROOT, "ImageSets", "train.txt")
    val_list_file = os.path.join(HRSC2016_ROOT, "ImageSets", "val.txt")

    # 读取划分文件（每行一个图片名，不含扩展名）
    with open(train_list_file) as f:
        train_names = [line.strip() for line in f.readlines()]
    with open(val_list_file) as f:
        val_names = [line.strip() for line in f.readlines()]

    # 截取所需数量
    train_names = train_names[:TRAIN_OPTICAL]
    val_names = val_names[:VAL_OPTICAL]

    # 处理训练集
    for idx, name in enumerate(train_names):
        # 查找图片（支持 .bmp .jpg .png）
        img_path = None
        for ext in ['.bmp', '.jpg', '.png']:
            candidate = os.path.join(img_dir, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            print(f"未找到图片: {name}")
            continue

        xml_path = os.path.join(ann_dir, name + ".xml")
        if not os.path.exists(xml_path):
            print(f"未找到 XML: {xml_path}")
            continue

        # 获取图片尺寸（优先从 XML 读取，否则用 PIL 读取）
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            width = int(root.find("Img_SizeWidth").text)
            height = int(root.find("Img_SizeHeight").text)
        except:
            with Image.open(img_path) as img:
                width, height = img.size

        # 转换标签
        yolo_lines = convert_xml_to_yolo(xml_path, width, height)
        if not yolo_lines:
            print(f"警告: {xml_path} 没有有效标注，跳过")
            continue

        # 复制并转换图片为 .jpg
        new_img_name = f"HRSC2016_{idx:04d}.jpg"
        new_img_path = os.path.join(OUTPUT_ROOT, "images", "train", new_img_name)
        with Image.open(img_path) as img:
            img.save(new_img_path, "JPEG")

        # 写入标签
        new_label_path = os.path.join(OUTPUT_ROOT, "labels", "train", new_img_name.replace('.jpg', '.txt'))
        with open(new_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx+1} 张 HRSC2016 训练图片")

    # 处理验证集
    for idx, name in enumerate(val_names):
        img_path = None
        for ext in ['.bmp', '.jpg', '.png']:
            candidate = os.path.join(img_dir, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            continue
        xml_path = os.path.join(ann_dir, name + ".xml")
        if not os.path.exists(xml_path):
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            width = int(root.find("Img_SizeWidth").text)
            height = int(root.find("Img_SizeHeight").text)
        except:
            with Image.open(img_path) as img:
                width, height = img.size

        yolo_lines = convert_xml_to_yolo(xml_path, width, height)
        if not yolo_lines:
            continue

        new_img_name = f"HRSC2016_{idx:04d}.jpg"
        new_img_path = os.path.join(OUTPUT_ROOT, "images", "val", new_img_name)
        with Image.open(img_path) as img:
            img.save(new_img_path, "JPEG")

        new_label_path = os.path.join(OUTPUT_ROOT, "labels", "val", new_img_name.replace('.jpg', '.txt'))
        with open(new_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx+1} 张 HRSC2016 验证图片")

    print(f"HRSC2016 处理完成：训练集 {len(train_names)} 张，验证集 {len(val_names)} 张")

def process_hrsid():
    """处理 HRSID SAR 数据集"""
    print("处理 HRSID...")
    img_dir = os.path.join(HRSID_ROOT, "JPEGImages")
    ann_dir = os.path.join(HRSID_ROOT, "annotations")

    # SAR 船舶的类别 ID（根据 CDHD.yaml：0=opt_ship, 1=sar_ship）
    SAR_CLASS_ID = 1

    # 确定 JSON 文件
    json_path = None
    candidate = os.path.join(ann_dir, "train_test2017.json")
    if os.path.exists(candidate):
        json_path = candidate
        print("使用 train_test2017.json")
    else:
        # 尝试合并 train2017.json 和 test2017.json
        train_json = os.path.join(ann_dir, "train2017.json")
        test_json = os.path.join(ann_dir, "test2017.json")
        if os.path.exists(train_json) and os.path.exists(test_json):
            print("合并 train2017.json 和 test2017.json")
            with open(train_json) as f:
                train_data = json.load(f)
            with open(test_json) as f:
                test_data = json.load(f)
            merged = {
                "images": train_data["images"] + test_data["images"],
                "annotations": train_data["annotations"] + test_data["annotations"],
                "categories": train_data["categories"]
            }
            # 临时保存合并后的 JSON（可选，不保存也可以）
            # 直接使用内存中的数据
            coco = merged
        else:
            raise FileNotFoundError(f"未找到可用的标注文件，请检查 {ann_dir}")

    if json_path:
        with open(json_path, 'r') as f:
            coco = json.load(f)

    # 构建映射
    images = {img['id']: img for img in coco['images']}
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann['bbox'])

    all_img_ids = list(images.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(all_img_ids)

    total_needed = TRAIN_SAR + VAL_SAR
    if len(all_img_ids) < total_needed:
        print(f"警告: HRSID 总共只有 {len(all_img_ids)} 张，少于需要 {total_needed}，将使用全部。")
        selected_ids = all_img_ids
    else:
        selected_ids = all_img_ids[:total_needed]

    train_ids = selected_ids[:TRAIN_SAR]
    val_ids = selected_ids[TRAIN_SAR:TRAIN_SAR+VAL_SAR]

    # 处理训练集
    for idx, img_id in enumerate(train_ids):
        img_info = images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            print(f"图片不存在：{img_path}")
            continue

        width = img_info['width']
        height = img_info['height']
        bboxes = annotations_by_image.get(img_id, [])

        yolo_lines = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            # 使用 SAR_CLASS_ID 作为类别
            yolo_lines.append(f"{SAR_CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if not yolo_lines:
            print(f"警告：{file_name} 没有标注，跳过")
            continue

        # 复制并转换图片为 .jpg
        new_img_name = f"HRSID_{idx:04d}.jpg"
        new_img_path = os.path.join(OUTPUT_ROOT, "images", "train", new_img_name)
        with Image.open(img_path) as img:
            img.save(new_img_path, "JPEG")

        new_label_path = os.path.join(OUTPUT_ROOT, "labels", "train", new_img_name.replace('.jpg', '.txt'))
        with open(new_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx+1} 张 HRSID 训练图片")

    # 处理验证集
    for idx, img_id in enumerate(val_ids):
        img_info = images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue

        width = img_info['width']
        height = img_info['height']
        bboxes = annotations_by_image.get(img_id, [])

        yolo_lines = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            # 使用 SAR_CLASS_ID 作为类别
            yolo_lines.append(f"{SAR_CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if not yolo_lines:
            continue

        new_img_name = f"HRSID_{idx:04d}.jpg"
        new_img_path = os.path.join(OUTPUT_ROOT, "images", "val", new_img_name)
        with Image.open(img_path) as img:
            img.save(new_img_path, "JPEG")

        new_label_path = os.path.join(OUTPUT_ROOT, "labels", "val", new_img_name.replace('.jpg', '.txt'))
        with open(new_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx+1} 张 HRSID 验证图片")

    print(f"HRSID 处理完成：训练集 {len(train_ids)} 张，验证集 {len(val_ids)} 张")

def generate_dataset_yaml():
    """生成 CDHD.yaml 配置文件"""
    yaml_content = f"""
# CDHD dataset
path: {OUTPUT_ROOT}  # dataset root dir
train: images/train  # train images
val: images/val      # val images

nc: 1  # number of classes
names: ['ship']  # class names
"""
    yaml_path = os.path.join(OUTPUT_ROOT, "CDHD.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"已生成数据集配置文件: {yaml_path}")

def main():
    create_dirs()
    process_hrsc2016()
    process_hrsid()
    generate_dataset_yaml()
    print("所有处理完成！")

if __name__ == "__main__":
    main()