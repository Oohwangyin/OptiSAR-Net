import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_and_split_dataset(src_dir, dst_dir, split_ratios=(0.7, 0.1, 0.2)):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 1. 创建目标目录结构 CORS-ADD-New/images/{train,val,test} 和 CORS-ADD-New/labels/{train,val,test}
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        (dst_path / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (dst_path / 'labels' / subset).mkdir(parents=True, exist_ok=True)

    # 2. 收集所有包含目标的 有效图片-标签对
    valid_pairs = []
    folders_to_use = ['train2017', 'val2017']

    print("正在扫描数据集并过滤无目标图片...")
    for folder in folders_to_use:
        img_folder = src_path / 'images' / folder
        lbl_folder = src_path / 'labels' / folder

        if not img_folder.exists() or not lbl_folder.exists():
            print(f"警告: 找不到文件夹 {folder}，将跳过。")
            continue

        # 遍历该文件夹下的所有 .tif 图片
        for img_file in img_folder.glob('*.tif'):
            base_name = img_file.stem
            lbl_file = lbl_folder / f"{base_name}.txt"

            # 过滤逻辑：判断对应的txt文件是否存在，且文件大小大于0（里面有标注内容）
            if lbl_file.exists() and lbl_file.stat().st_size > 0:
                with open(lbl_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 确保里面不仅仅是空格或换行符
                        valid_pairs.append((img_file, lbl_file))

    total_valid = len(valid_pairs)
    print(f"扫描完毕！包含目标的有效图片总数为: {total_valid}")

    if total_valid == 0:
        print("未找到有效数据，请检查路径。")
        return

    # 3. 打乱顺序并划分数据集 (7 : 1 : 2)
    random.seed(42)  # 固定随机种子以保证每次划分结果一致
    random.shuffle(valid_pairs)

    train_end = int(total_valid * split_ratios[0])
    val_end = train_end + int(total_valid * split_ratios[1])

    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }

    # 4. 转换图片格式并复制文件
    for subset_name, pairs in splits.items():
        print(f"正在处理 {subset_name} 集 (共 {len(pairs)} 张图片)...")

        for img_src, lbl_src in tqdm(pairs, desc=subset_name):
            # 构建目标路径
            img_dst = dst_path / 'images' / subset_name / f"{img_src.stem}.jpg"
            lbl_dst = dst_path / 'labels' / subset_name / lbl_src.name

            try:
                # 使用 Pillow 打开 .tif 并转换为 .jpg (如果是四通道 RGBA 等需要转为 RGB)
                with Image.open(img_src) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(img_dst, 'JPEG', quality=95)

                # 复制 TXT 标注文件
                shutil.copy2(lbl_src, lbl_dst)

            except Exception as e:
                print(f"处理文件时出错 {img_src.name}: {e}")

    print("\n数据集转换和重组完成！新数据集位于:", dst_path.absolute())


if __name__ == '__main__':
    # 请确保这两个文件夹路径与您的实际情况相符
    SOURCE_DIR = '../datasets/CORS-ADD'
    TARGET_DIR = '../datasets/CORS-ADD-New'

    convert_and_split_dataset(SOURCE_DIR, TARGET_DIR)