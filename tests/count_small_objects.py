"""Count COCO-style object scales in a YOLO dataset.

AP_S/AP_M/AP_L use object area in original image pixels. The default thresholds
follow COCO: small < 32^2, medium is [32^2, 96^2), and large >= 96^2.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Count small/medium/large object ratios in a YOLO dataset.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/CORS-SAR.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", default="all", choices=("train", "val", "test", "all"), help="Dataset split.")
    parser.add_argument("--area-thr", type=float, default=32.0**2, help="Small-object area threshold in pixels.")
    parser.add_argument("--medium-thr", type=float, default=96.0**2, help="Large-object area threshold in pixels.")
    parser.add_argument("--root", default="", help="Optional dataset root override.")
    parser.add_argument("--save-csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def resolve_data_yaml(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()

    candidate = Path("ultralytics/cfg/datasets") / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Dataset YAML not found: {path}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_root(data: dict, yaml_path: Path, root_override: str = "") -> Path:
    if root_override:
        return Path(root_override).resolve()

    raw = Path(data.get("path", "."))
    if raw.is_absolute():
        return raw

    candidates = [Path.cwd() / raw, yaml_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_source(source: str | Path, root: Path) -> Path:
    source = Path(source)
    if source.is_absolute():
        return source
    return (root / source).resolve()


def iter_images(source, root: Path):
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from iter_images(item, root)
        return

    src = resolve_source(source, root)
    if src.is_dir():
        for path in sorted(src.rglob("*")):
            if path.suffix.lower() in IMG_EXTS:
                yield path.resolve()
    elif src.is_file() and src.suffix.lower() == ".txt":
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                yield (path if path.is_absolute() else (root / path)).resolve()
    elif src.is_file() and src.suffix.lower() in IMG_EXTS:
        yield src.resolve()
    else:
        raise FileNotFoundError(f"Image source not found: {src}")


def label_path_for(image_path: Path, root: Path) -> Path:
    try:
        rel = image_path.resolve().relative_to(root.resolve())
        parts = list(rel.parts)
        base = root
    except ValueError:
        parts = list(image_path.parts)
        base = Path(parts[0])
        parts = parts[1:]

    if "images" in parts:
        parts[parts.index("images")] = "labels"
    else:
        parts.insert(-1, "labels")
    return (base / Path(*parts)).with_suffix(".txt")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def object_area(parts: list[str], img_w: int, img_h: int) -> tuple[int, float] | None:
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))

    if len(parts) == 5:
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h
    elif (len(parts) - 1) % 2 == 0:
        coords = [float(x) for x in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        bw = (max(xs) - min(xs)) * img_w
        bh = (max(ys) - min(ys)) * img_h
    else:
        return None

    return cls, max(bw, 0.0) * max(bh, 0.0)


def empty_stats():
    return dict(images=0, missing_labels=0, objects=0, small=0, medium=0, large=0, skipped=0)


def add_stats(dst, src):
    for key, value in src.items():
        dst[key] += value


def class_name(names, cls: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls, names.get(str(cls), cls)))
    if isinstance(names, list) and cls < len(names):
        return str(names[cls])
    return str(cls)


def scale_name(area: float, small_thr: float, medium_thr: float) -> str:
    if area < small_thr:
        return "small"
    if area < medium_thr:
        return "medium"
    return "large"


def count_split(split: str, source, root: Path, small_thr: float, medium_thr: float):
    stats = empty_stats()
    per_class = defaultdict(empty_stats)

    for image_path in iter_images(source, root):
        stats["images"] += 1
        try:
            img_w, img_h = image_size(image_path)
        except Exception:
            stats["skipped"] += 1
            continue

        label_path = label_path_for(image_path, root)
        if not label_path.exists():
            stats["missing_labels"] += 1
            continue

        image_classes = set()
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = object_area(line.split(), img_w, img_h)
                if parsed is None:
                    stats["skipped"] += 1
                    continue
                cls, area = parsed
                scale = scale_name(area, small_thr, medium_thr)
                stats["objects"] += 1
                stats[scale] += 1
                image_classes.add(cls)
                per_class[cls]["objects"] += 1
                per_class[cls][scale] += 1

        for cls in image_classes:
            per_class[cls]["images"] += 1

    return split, stats, per_class


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def print_rows(rows, title):
    print(f"\n{title}")
    print("-" * 118)
    print(
        f"{'name':<18}{'images':>9}{'objects':>10}"
        f"{'small':>10}{'small%':>10}{'medium':>10}{'medium%':>10}"
        f"{'large':>10}{'large%':>10}{'missing':>10}{'skipped':>10}"
    )
    for row in rows:
        print(
            f"{row['name']:<18}{row['images']:>9}{row['objects']:>10}"
            f"{row['small']:>10}{row['small_pct']:>9.2f}%"
            f"{row['medium']:>10}{row['medium_pct']:>9.2f}%"
            f"{row['large']:>10}{row['large_pct']:>9.2f}%"
            f"{row['missing_labels']:>10}{row['skipped']:>10}"
        )


def main():
    args = parse_args()
    yaml_path = resolve_data_yaml(args.data)
    data = load_yaml(yaml_path)
    root = resolve_dataset_root(data, yaml_path, args.root)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)

    print(f"data: {yaml_path}")
    print(f"root: {root}")
    print(f"small object: area < {args.area_thr:g} px^2")
    print(f"medium object: {args.area_thr:g} <= area < {args.medium_thr:g} px^2")
    print(f"large object: area >= {args.medium_thr:g} px^2")

    split_rows = []
    class_totals = defaultdict(empty_stats)
    total = empty_stats()

    for split in splits:
        source = data.get(split)
        if not source:
            continue
        _, stats, per_class = count_split(split, source, root, args.area_thr, args.medium_thr)
        add_stats(total, stats)
        split_rows.append(
            dict(
                name=split,
                **stats,
                small_pct=pct(stats["small"], stats["objects"]),
                medium_pct=pct(stats["medium"], stats["objects"]),
                large_pct=pct(stats["large"], stats["objects"]),
            )
        )
        for cls, cls_stats in per_class.items():
            add_stats(class_totals[cls], cls_stats)

    split_rows.append(
        dict(
            name="all",
            **total,
            small_pct=pct(total["small"], total["objects"]),
            medium_pct=pct(total["medium"], total["objects"]),
            large_pct=pct(total["large"], total["objects"]),
        )
    )
    print_rows(split_rows, "By split")

    names = data.get("names", {})
    class_rows = []
    for cls in sorted(class_totals):
        stats = class_totals[cls]
        class_rows.append(
            dict(
                name=f"{cls}:{class_name(names, cls)}",
                **stats,
                small_pct=pct(stats["small"], stats["objects"]),
                medium_pct=pct(stats["medium"], stats["objects"]),
                large_pct=pct(stats["large"], stats["objects"]),
            )
        )
    print_rows(class_rows, "By class")

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=(
                    "group",
                    "name",
                    "images",
                    "objects",
                    "small",
                    "small_pct",
                    "medium",
                    "medium_pct",
                    "large",
                    "large_pct",
                    "missing_labels",
                    "skipped",
                ),
            )
            writer.writeheader()
            for row in split_rows:
                writer.writerow(dict(group="split", **row))
            for row in class_rows:
                writer.writerow(dict(group="class", **row))
        print(f"\nsaved: {out.resolve()}")


if __name__ == "__main__":
    main()
