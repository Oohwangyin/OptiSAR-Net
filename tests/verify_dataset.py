"""
DOTAv1 数据集类别统计脚本
用于统计数据集中每个类别的实例数量
"""

import os
from pathlib import Path
from collections import Counter, defaultdict
import yaml
from ultralytics.utils import LOGGER


def img2label_paths(img_paths):
    """从图像路径生成对应的标签路径"""
    sa, sb = f'{os.sep}images', f'{os.sep}labels'  # /images, /labels
    return [sb.join(p.rsplit(sa, 1)).rsuffix('.txt') for p in img_paths]


def get_image_paths(data_dir, split='train'):
    """获取指定 split 的所有图像路径"""
    images_dir = Path(data_dir) / 'images' / split

    if not images_dir.exists():
        LOGGER.warning(f"目录不存在：{images_dir}")
        return []

    # 支持常见的图像格式
    image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_paths = []

    for ext in image_formats:
        image_paths.extend(images_dir.glob(f'*{ext}'))
        image_paths.extend(images_dir.glob(f'*{ext.upper()}'))

    return sorted(image_paths)


def parse_label_file(label_path):
    """
    解析 YOLO 格式的标签文件
    返回该文件中所有目标的类别索引列表
    """
    classes = []

    if not Path(label_path).exists():
        return classes

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                parts = line.split()
                if len(parts) >= 5:  # YOLO 格式：class x y w h (至少 5 个值)
                    class_idx = int(parts[0])
                    classes.append(class_idx)
    except Exception as e:
        LOGGER.warning(f"读取标签文件失败 {label_path}: {e}")

    return classes


def count_instances(data_dir, splits=None):
    """
    统计数据集中各类别的实例数量

    Args:
        data_dir (str | Path): 数据集根目录
        splits (list, optional): 要统计的数据集划分，如 ['train', 'val', 'test']
                                默认为 None，自动检测存在的划分

    Returns:
        dict: 包含统计结果的字典
    """
    data_dir = Path(data_dir)

    # 加载 dataset YAML 配置（如果存在）
    yaml_files = list(data_dir.glob('*.yaml'))
    class_names = {}
    if yaml_files:
        try:
            with open(yaml_files[0], 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                if 'names' in data_config:
                    class_names = data_config['names']
                    LOGGER.info(f"从配置文件加载类别名称：{len(class_names)} 类")
        except Exception as e:
            LOGGER.warning(f"加载 YAML 配置文件失败：{e}")

    # 默认统计所有可能的划分
    if splits is None:
        splits = []
        for split in ['train', 'val', 'test']:
            if (data_dir / 'images' / split).exists():
                splits.append(split)

    if not splits:
        LOGGER.error("未找到任何有效的数据集划分 (train/val/test)")
        return {}

    LOGGER.info(f"开始统计以下数据集划分：{splits}")

    # 统计结果
    total_counter = Counter()  # 总计
    split_counters = {}  # 各划分的统计

    for split in splits:
        LOGGER.info(f"\n处理 {split} 集...")

        # 获取所有图像路径
        image_paths = get_image_paths(data_dir, split)

        if not image_paths:
            LOGGER.warning(f"{split} 集没有找到图像文件")
            continue

        label_paths = img2label_paths([str(p) for p in image_paths])

        counter = Counter()
        valid_images = 0
        no_label_images = 0

        for img_path, lb_path in zip(image_paths, label_paths):
            if Path(lb_path).exists():
                classes = parse_label_file(lb_path)
                if classes:
                    counter.update(classes)
                    valid_images += 1
                else:
                    no_label_images += 1
            else:
                no_label_images += 1

        split_counters[split] = {
            'counter': counter,
            'total_images': len(image_paths),
            'valid_images': valid_images,
            'no_label_images': no_label_images
        }

        # 累加到总计
        total_counter.update(counter)

        LOGGER.info(f"{split} 集：{len(image_paths)} 张图像，"
                   f"{valid_images} 张有标注，{no_label_images} 张无标注")
        LOGGER.info(f"{split} 集实例总数：{sum(counter.values())}")

    # 打印统计结果
    print("\n" + "="*80)
    print("DOTAv1 数据集类别统计结果".center(80))
    print("="*80)

    # 表头
    print(f"\n{'类别 ID':<10} {'类别名称':<25} {'训练集':<12} {'验证集':<12} {'测试集':<12} {'总计':<10} {'占比':<10}")
    print("-"*80)

    total_instances = sum(total_counter.values())

    # 按类别 ID 排序输出
    all_classes = sorted(total_counter.keys())

    for class_idx in all_classes:
        class_name = class_names.get(class_idx, f"class_{class_idx}")

        train_count = split_counters.get('train', {}).get('counter', Counter()).get(class_idx, 0)
        val_count = split_counters.get('val', {}).get('counter', Counter()).get(class_idx, 0)
        test_count = split_counters.get('test', {}).get('counter', Counter()).get(class_idx, 0)
        total_count = total_counter.get(class_idx, 0)

        percentage = (total_count / total_instances * 100) if total_instances > 0 else 0

        print(f"{class_idx:<10} {class_name:<25} {train_count:<12} {val_count:<12} {test_count:<12} {total_count:<10} {percentage:>9.2f}%")

    print("-"*80)
    print(f"{'总计':<37} {split_counters.get('train', {}).get('counter', Counter()).sum():<12} "
          f"{split_counters.get('val', {}).get('counter', Counter()).sum():<12} "
          f"{split_counters.get('test', {}).get('counter', Counter()).sum():<12} "
          f"{total_instances:<10} {100.0:>9.2f}%")
    print("="*80)

    # 输出图像数量统计
    print("\n图像数量统计:")
    for split in splits:
        if split in split_counters:
            info = split_counters[split]
            print(f"  {split:8s}: {info['total_images']:4d} 张图像，"
                  f"{info['valid_images']:4d} 张有标注，"
                  f"{info['no_label_images']:4d} 张无标注")

    print("="*80 + "\n")

    # 返回统计结果
    return {
        'class_names': class_names,
        'total_counter': dict(total_counter),
        'split_counters': {k: v['counter'] for k, v in split_counters.items()},
        'image_stats': {k: {kk: vv for kk, vv in v.items() if kk != 'counter'}
                       for k, v in split_counters.items()},
        'total_instances': total_instances
    }


def main():
    """主函数"""
    # 配置数据集路径
    # 相对路径（相对于当前工作目录或项目根目录）
    dataset_path = Path(__file__).parent.parent / 'datasets' / 'DOTAv1'

    # 如果上面路径不存在，可以尝试绝对路径
    if not dataset_path.exists():
        # Windows 系统示例
        dataset_path = Path(r'D:\ProjectCode\PycharmProjects\OptiSAR-Net\datasets\DOTAv1')

    # 检查数据集是否存在
    if not dataset_path.exists():
        LOGGER.error(f"数据集路径不存在：{dataset_path}")
        LOGGER.info("请修改 dataset_path 变量指向正确的 DOTAv1 数据集路径")
        return

    LOGGER.info(f"数据集路径：{dataset_path}")

    # 执行统计
    results = count_instances(dataset_path, splits=['train', 'val', 'test'])

    if not results:
        LOGGER.error("统计失败，请检查数据集路径和格式")
        return

    # 可选：保存统计结果到文件
    save_path = dataset_path / 'dataset_statistics.yaml'
    try:
        stats_to_save = {
            'class_names': results['class_names'],
            'total_instances': results['total_instances'],
            'total_per_class': results['total_counter'],
            'per_split': {
                split: dict(counter)
                for split, counter in results['split_counters'].items()
            },
            'image_stats': results['image_stats']
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats_to_save, f, allow_unicode=True, default_flow_style=False)

        LOGGER.info(f"统计结果已保存到：{save_path}")
    except Exception as e:
        LOGGER.error(f"保存统计结果失败：{e}")


if __name__ == '__main__':
    main()
