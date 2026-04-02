#!/usr/bin/env python3
"""
Convert OBB annotations in M4-SAR-sampled to horizontal bounding boxes (YOLO format).
Usage: python convert_obb_to_hbb.py --src /path/to/M4-SAR-sampled --dst /path/to/output --overwrite
If --overwrite is used, original files are replaced. Otherwise, new files are saved in a separate directory.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_obb_to_hbb(line):
    """
    Convert a line of OBB format to HBB YOLO format.
    Input line: class_id x1 y1 x2 y2 x3 y3 x4 y4 (all normalized)
    Output: "class_id x_center y_center width height"
    """
    parts = line.strip().split()
    if len(parts) != 9:
        # Not OBB format, skip or raise error
        return line.strip()  # keep original if unexpected
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    # Ensure values are within [0,1] (may be slightly out due to rounding)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_file(input_path, output_path, overwrite=False):
    """Convert a single file."""
    if not os.path.exists(input_path):
        return
    with open(input_path, 'r') as f:
        lines = f.readlines()
    converted_lines = []
    for line in lines:
        if line.strip():
            converted_lines.append(convert_obb_to_hbb(line))
        else:
            converted_lines.append('')
    # Write output
    if overwrite:
        # Write directly to input file
        out_path = input_path
    else:
        out_path = output_path
    with open(out_path, 'w') as f:
        f.write('\n'.join(converted_lines))
        if converted_lines and converted_lines[-1] != '':
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Convert OBB to HBB YOLO format for M4-SAR-sampled')
    parser.add_argument('--src', default='../datasets/M4-SAR-sampled', help='Source dataset directory')
    parser.add_argument('--dst', default=None, help='Output directory for converted labels (if not overwriting)')
    parser.add_argument('--overwrite', action='store_true',default=True, help='Overwrite original files instead of saving to new dir')
    args = parser.parse_args()

    src_root = Path(args.src)
    labels_dir = src_root / 'labels'
    if not labels_dir.exists():
        print(f"Error: labels directory not found at {labels_dir}")
        return

    if args.overwrite:
        output_dir = labels_dir  # will overwrite in place
    else:
        if args.dst:
            output_dir = Path(args.dst)
        else:
            output_dir = src_root / 'labels_hbb'
        print(f"Saving converted labels to {output_dir}")

    # Walk through splits
    for split in ['train', 'val', 'test']:
        src_split = labels_dir / split
        if not src_split.exists():
            continue
        if not args.overwrite:
            dst_split = output_dir / split
            dst_split.mkdir(parents=True, exist_ok=True)
        else:
            dst_split = src_split  # for overwrite, we'll write directly
        files = list(src_split.glob('*.txt'))
        if not files:
            continue
        print(f"Processing {split}...")
        for f in tqdm(files):
            if args.overwrite:
                process_file(f, None, overwrite=True)
            else:
                out_file = dst_split / f.name
                process_file(f, out_file, overwrite=False)
    print("Done.")

if __name__ == '__main__':
    main()