import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_voc_to_yolo(size, box):
    """
    将 VOC 的绝对坐标 (xmin, ymin, xmax, ymax)
    转换为 YOLO 的归一化中心坐标 (x_center, y_center, width, height)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def process_dataset(src_root, dst_root):
    src_path = Path(src_root)
    dst_path = Path(dst_root)

    # 1. 定义子集映射
    subsets = ['train', 'val', 'test']

    # 2. 创建目标目录结构
    for s in subsets:
        (dst_path / 'images' / s).mkdir(parents=True, exist_ok=True)
        (dst_path / 'labels' / s).mkdir(parents=True, exist_ok=True)

    print(f"开始转换数据集: {src_root} -> {dst_root}")

    for subset in subsets:
        split_file = src_path / 'ImageSets/Main' / f"{subset}.txt"
        if not split_file.exists():
            print(f"跳过 {subset}: 找不到划分文件 {split_file}")
            continue

        # 读取该子集的文件名列表
        with open(split_file, 'r') as f:
            file_names = [line.strip() for line in f.readlines() if line.strip()]

        count_processed = 0
        count_filtered = 0

        print(f"正在处理 {subset} 集...")
        for name in tqdm(file_names):
            xml_file = src_path / 'Annotations' / f"{name}.xml"
            img_file = src_path / 'JPEGImages' / f"{name}.jpg"

            if not xml_file.exists() or not img_file.exists():
                continue

            # 解析 XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size_node = root.find('size')
            w = int(size_node.find('width').text)
            h = int(size_node.find('height').text)

            yolo_labels = []
            for obj in root.iter('object'):
                # 提取坐标
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))

                # 转换为 YOLO 格式 (假设飞机类别 ID 为 0)
                bb = convert_voc_to_yolo((w, h), b)
                yolo_labels.append(f"0 {' '.join([f'{a:.6f}' for a in bb])}")

            # 过滤掉不含目标的图片
            if not yolo_labels:
                count_filtered += 1
                continue

            # 写入目标文件
            # 1. 复制并重定位图片
            shutil.copy(img_file, dst_path / 'images' / subset / f"{name}.jpg")

            # 2. 写入 TXT 标签
            with open(dst_path / 'labels' / subset / f"{name}.txt", 'w') as f_out:
                f_out.write('\n'.join(yolo_labels))

            count_processed += 1

        print(f"{subset} 处理完成: 保留 {count_processed} 张，过滤 {count_filtered} 张空目标图片。")


if __name__ == '__main__':
    # 配置路径
    SOURCE_DIR = '../datasets/SAR-AIRcraft-1.0'  # 原始数据集文件夹
    TARGET_DIR = '../datasets/SAR-AIRcraft'  # 转换后的目标文件夹

    if os.path.exists(SOURCE_DIR):
        process_dataset(SOURCE_DIR, TARGET_DIR)
        print("\n所有任务已完成！")
    else:
        print(f"错误：找不到源目录 {SOURCE_DIR}，请检查路径。")