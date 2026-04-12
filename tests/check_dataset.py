#!/usr/bin/env python3
"""
检查 CORS-SAR 数据集的完整性：
1. 检查光学图像与 SAR 图像是否配对
2. 检查图像是否有对应的标注文件
3. 统计各数据集中的实例个数（目标检测框数量）

用法:
    python check_dataset.py --data_root ../datasets/CORS-SAR
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict


def get_image_pairs(img_dir):
    """
    获取图像目录中的所有 optical/sar 图像对。
    返回：
        optical_set: 光学图像基础名集合
        sar_set: SAR 图像基础名集合
        optical_files: 光学图像文件列表
        sar_files: SAR 图像文件列表
    """
    if not img_dir.exists():
        return set(), set(), [], []

    optical_files = []
    sar_files = []

    for fname in os.listdir(img_dir):
        if fname.startswith('optical_') and (fname.endswith('.jpg') or fname.endswith('.png')):
            optical_files.append(fname)
        elif fname.startswith('sar_') and (fname.endswith('.jpg') or fname.endswith('.png')):
            sar_files.append(fname)

    # 提取基础名（去掉 optical_/sar_ 前缀和图像扩展名）
    optical_set = {os.path.splitext(f)[0][8:] for f in optical_files}  # 去掉 'optical_' 和 .jpg/.png
    sar_set = {os.path.splitext(f)[0][4:] for f in sar_files}  # 去掉 'sar_' 和 .jpg/.png

    return optical_set, sar_set, optical_files, sar_files


def get_label_files(label_dir):
    """
    获取标签目录中的所有标签文件。
    返回：
        optical_labels: 光学标签基础名集合
        sar_labels: SAR 标签基础名集合
        optical_label_files: 光学标签文件列表
        sar_label_files: SAR 标签文件列表
    """
    if not label_dir.exists():
        return set(), set(), [], []

    optical_label_files = []
    sar_label_files = []

    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            if fname.startswith('optical_'):
                optical_label_files.append(fname)
            elif fname.startswith('sar_'):
                sar_label_files.append(fname)

    # 提取基础名（去掉 optical_/sar_ 前缀和 .txt 扩展名）
    optical_labels = {os.path.splitext(f)[0][8:] for f in optical_label_files}  # 去掉 'optical_' 和 .txt
    sar_labels = {os.path.splitext(f)[0][4:] for f in sar_label_files}  # 去掉 'sar_' 和 .txt

    return optical_labels, sar_labels, optical_label_files, sar_label_files


def count_instances_in_label(label_path):
    """
    统计标签文件中的实例数量（非空行数）。
    返回：(总实例数, 类别0实例数, 类别1实例数)
    """
    total = 0
    class_0_count = 0
    class_1_count = 0
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    total += 1
                    try:
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            class_0_count += 1
                        elif cls_id == 1:
                            class_1_count += 1
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  ⚠️  读取标签文件失败 {label_path}: {e}")
    
    return total, class_0_count, class_1_count


def check_split(split_name, img_dir, label_dir, verbose=True):
    """
    检查单个 split 的完整性。
    返回检查结果字典
    """
    result = {
        'split': split_name,
        'optical_images': 0,
        'sar_images': 0,
        'optical_labels': 0,
        'sar_labels': 0,
        'paired_images': 0,
        'total_instances': 0,
        'optical_aircraft_instances': 0,  # 类别0：光学飞机
        'sar_aircraft_instances': 0,      # 类别1：SAR飞机
        'missing_sar_images': [],
        'missing_optical_images': [],
        'missing_optical_labels': [],
        'missing_sar_labels': [],
        'extra_optical_labels': [],
        'extra_sar_labels': [],
        'images_without_labels': [],
        'label_content_errors': [],
    }

    # 获取图像对
    optical_imgs, sar_imgs, optical_file_list, sar_file_list = get_image_pairs(img_dir)
    result['optical_images'] = len(optical_imgs)
    result['sar_images'] = len(sar_imgs)

    # 获取标签文件
    optical_labels, sar_labels, optical_label_list, sar_label_list = get_label_files(label_dir)
    result['optical_labels'] = len(optical_labels)
    result['sar_labels'] = len(sar_labels)

    # 1. 检查图像配对
    paired = optical_imgs & sar_imgs
    result['paired_images'] = len(paired)

    # 只有光学图没有 SAR 图
    missing_sar = optical_imgs - sar_imgs
    result['missing_sar_images'] = sorted(list(missing_sar))

    # 只有 SAR 图没有光学图
    missing_optical = sar_imgs - optical_imgs
    result['missing_optical_images'] = sorted(list(missing_optical))

    # 2. 检查图像与标签的对应关系
    # 光学图像缺少标签
    missing_opt_label = optical_imgs - optical_labels
    result['missing_optical_labels'] = sorted(list(missing_opt_label))

    # SAR 图像缺少标签
    missing_sar_label = sar_imgs - sar_labels
    result['missing_sar_labels'] = sorted(list(missing_sar_label))

    # 有标签但没有对应图像（多余的标签）
    extra_opt_label = optical_labels - optical_imgs
    result['extra_optical_labels'] = sorted(list(extra_opt_label))

    extra_sar_label = sar_labels - sar_imgs
    result['extra_sar_labels'] = sorted(list(extra_sar_label))

    # 3. 检查有图像但无标签的情况（综合检查）
    all_imgs = optical_imgs | sar_imgs
    all_labels = optical_labels | sar_labels
    without_labels = all_imgs - all_labels
    result['images_without_labels'] = sorted(list(without_labels))

    # 4. 统计实例数量并检查标签内容
    if verbose:
        # 统计光学标签实例
        for label_file in optical_label_list:
            label_path = label_dir / label_file
            total, cls0, cls1 = count_instances_in_label(label_path)
            result['total_instances'] += total
            result['optical_aircraft_instances'] += cls0
            result['sar_aircraft_instances'] += cls1
            
            # 检查格式错误
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            result['label_content_errors'].append(
                                f"{label_file}: 第{i}行格式错误 (期望 5 个值，实际{len(parts)}个)"
                            )
                        else:
                            # 检查数值范围
                            try:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:])
                                if cls not in [0, 1]:
                                    result['label_content_errors'].append(
                                        f"{label_file}: 第{i}行类别ID错误 (应为0或1，实际{cls})"
                                    )
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    result['label_content_errors'].append(
                                        f"{label_file}: 第{i}行数值超出 [0,1] 范围"
                                    )
                            except ValueError:
                                result['label_content_errors'].append(
                                    f"{label_file}: 第{i}行数值解析错误"
                                )
            except Exception as e:
                result['label_content_errors'].append(f"{label_file}: 读取错误 - {str(e)}")
        
        # 统计SAR标签实例
        for label_file in sar_label_list:
            label_path = label_dir / label_file
            total, cls0, cls1 = count_instances_in_label(label_path)
            result['total_instances'] += total
            result['optical_aircraft_instances'] += cls0
            result['sar_aircraft_instances'] += cls1
            
            # 检查格式错误
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            result['label_content_errors'].append(
                                f"{label_file}: 第{i}行格式错误 (期望 5 个值，实际{len(parts)}个)"
                            )
                        else:
                            # 检查数值范围
                            try:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:])
                                if cls not in [0, 1]:
                                    result['label_content_errors'].append(
                                        f"{label_file}: 第{i}行类别ID错误 (应为0或1，实际{cls})"
                                    )
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    result['label_content_errors'].append(
                                        f"{label_file}: 第{i}行数值超出 [0,1] 范围"
                                    )
                            except ValueError:
                                result['label_content_errors'].append(
                                    f"{label_file}: 第{i}行数值解析错误"
                                )
            except Exception as e:
                result['label_content_errors'].append(f"{label_file}: 读取错误 - {str(e)}")

    return result


def print_report(results, show_details=True):
    """打印检查报告"""
    print("=" * 80)
    print("📊 CORS-SAR 数据集完整性检查报告")
    print("=" * 80)

    total_issues = 0
    grand_total_instances = 0
    grand_optical_instances = 0
    grand_sar_instances = 0

    for result in results:
        split = result['split']
        print(f"\n{'='*80}")
        print(f"📁 Split: {split.upper()}")
        print(f"{'='*80}")

        # 基本统计
        print(f"\n📈 基本统计:")
        print(f"  光学图像：{result['optical_images']} 张")
        print(f"  SAR 图像：{result['sar_images']} 张")
        print(f"  光学标签：{result['optical_labels']} 个")
        print(f"  SAR 标签：{result['sar_labels']} 个")
        print(f"  配对图像：{result['paired_images']} 对")

        # 实例统计
        print(f"\n🎯 实例统计:")
        print(f"  总实例数：{result['total_instances']} 个")
        print(f"  光学飞机 (类别0)：{result['optical_aircraft_instances']} 个")
        print(f"  SAR 飞机 (类别1)：{result['sar_aircraft_instances']} 个")
        
        grand_total_instances += result['total_instances']
        grand_optical_instances += result['optical_aircraft_instances']
        grand_sar_instances += result['sar_aircraft_instances']

        # 问题统计
        issues = []

        if result['missing_sar_images']:
            issues.append(f"  ❌ 缺少 SAR 配对：{len(result['missing_sar_images'])} 张光学图")
            if show_details and len(result['missing_sar_images']) <= 10:
                for fname in result['missing_sar_images'][:10]:
                    issues.append(f"     - optical_{fname}")
            elif len(result['missing_sar_images']) > 10:
                issues.append(f"     (前 10 个：{', '.join('optical_' + f for f in result['missing_sar_images'][:10])})")

        if result['missing_optical_images']:
            issues.append(f"  ❌ 缺少光学配对：{len(result['missing_optical_images'])} 张 SAR 图")
            if show_details and len(result['missing_optical_images']) <= 10:
                for fname in result['missing_optical_images'][:10]:
                    issues.append(f"     - sar_{fname}")

        if result['missing_optical_labels']:
            issues.append(f"  ❌ 缺少光学标签：{len(result['missing_optical_labels'])} 张图像")
            if show_details and len(result['missing_optical_labels']) <= 10:
                for fname in result['missing_optical_labels'][:10]:
                    issues.append(f"     - optical_{fname}.jpg")

        if result['missing_sar_labels']:
            issues.append(f"  ❌ 缺少 SAR 标签：{len(result['missing_sar_labels'])} 张图像")
            if show_details and len(result['missing_sar_labels']) <= 10:
                for fname in result['missing_sar_labels'][:10]:
                    issues.append(f"     - sar_{fname}.jpg")

        if result['extra_optical_labels']:
            issues.append(f"  ⚠️  多余光学标签：{len(result['extra_optical_labels'])} 个")

        if result['extra_sar_labels']:
            issues.append(f"  ⚠️  多余 SAR 标签：{len(result['extra_sar_labels'])} 个")

        if result['label_content_errors']:
            issues.append(f"  ⚠️  标签内容错误：{len(result['label_content_errors'])} 个")
            if show_details and len(result['label_content_errors']) <= 5:
                for err in result['label_content_errors'][:5]:
                    issues.append(f"     - {err}")

        total_issues += len(issues)

        if issues:
            print(f"\n🚨 发现的问题:")
            for issue in issues:
                print(issue)
        else:
            print(f"\n✅ 未发现问题！")

    # 总结
    print(f"\n{'='*80}")
    print("📋 总体总结")
    print(f"{'='*80}")
    print(f"\n🎯 全数据集实例统计:")
    print(f"  总实例数：{grand_total_instances} 个")
    print(f"  光学飞机 (类别0)：{grand_optical_instances} 个")
    print(f"  SAR 飞机 (类别1)：{grand_sar_instances} 个")
    
    if total_issues == 0:
        print(f"\n✅ 所有 split 均无问题！CORS-SAR 数据集完整性良好。")
    else:
        print(f"\n⚠️  共发现 {total_issues} 个问题项，请检查上述详细信息。")

    return total_issues


def main():
    parser = argparse.ArgumentParser(description='检查 CORS-SAR 数据集完整性')
    parser.add_argument('--data_root', type=str, default='../datasets/CORS-SAR',
                        help='数据集根目录')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='要检查的 splits')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，只显示统计信息，不显示详细文件列表')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()

    if not data_root.exists():
        print(f"❌ 错误：数据集根目录不存在：{data_root}")
        return

    print(f"数据集根目录：{data_root}")
    print(f"检查 splits: {', '.join(args.splits)}")

    results = []

    for split in args.splits:
        img_dir = data_root / 'images' / split
        label_dir = data_root / 'labels' / split

        # 检查目录是否存在
        if not img_dir.exists() and not label_dir.exists():
            print(f"\n⚠️  跳过 {split}: images 和 labels 目录均不存在")
            continue

        result = check_split(split, img_dir, label_dir, verbose=not args.quiet)
        results.append(result)

    # 打印报告
    print_report(results, show_details=not args.quiet)


if __name__ == '__main__':
    main()
