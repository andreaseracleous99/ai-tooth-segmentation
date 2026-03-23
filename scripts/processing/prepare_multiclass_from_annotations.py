import argparse
import base64
import json
import random
import shutil
import zlib
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

FDI_CLASSES = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]


def decode_bitmap_mask(bitmap_data_b64: str) -> np.ndarray:
    raw = base64.b64decode(bitmap_data_b64)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass

    img = Image.open(BytesIO(raw)).convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    return (alpha > 0).astype(np.uint8)


def crop_bbox_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def find_image_for_annotation(images_dir: Path, annotation_file: Path) -> Path | None:
    stem = annotation_file.stem
    if stem.lower().endswith(".jpg") or stem.lower().endswith(".jpeg") or stem.lower().endswith(".png"):
        base = Path(stem).stem
    else:
        base = stem

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = images_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def ensure_dirs(out_root: Path):
    for split in ["train", "val"]:
        for fdi in FDI_CLASSES:
            (out_root / split / str(fdi)).mkdir(parents=True, exist_ok=True)


def prepare_dataset(source_root: Path, output_root: Path, train_ratio: float, seed: int, min_size: int, clear_output: bool):
    if clear_output:
        for split in ["train", "val"]:
            split_dir = output_root / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    ensure_dirs(output_root)

    random.seed(seed)
    extracted = defaultdict(list)

    source_sets = [d for d in source_root.iterdir() if d.is_dir()]
    source_sets.sort(key=lambda p: p.name)

    total_ann_files = 0
    total_objects = 0
    total_saved = 0

    for dataset_dir in source_sets:
        ann_dir = dataset_dir / "annotations"
        img_dir = dataset_dir / "images"
        if not ann_dir.exists() or not img_dir.exists():
            continue

        ann_files = sorted(ann_dir.glob("*.json"))
        if not ann_files:
            continue

        print("Processing set", dataset_dir.name, "annotations:", len(ann_files))

        for ann_file in ann_files:
            total_ann_files += 1
            try:
                data = json.loads(ann_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            img_path = find_image_for_annotation(img_dir, ann_file)
            if img_path is None:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_np = np.array(img)
            H, W = img_np.shape[0], img_np.shape[1]

            objects = data.get("objects", [])
            for obj_idx, obj in enumerate(objects):
                total_objects += 1
                if obj.get("geometryType") != "bitmap":
                    continue

                cls_title = str(obj.get("classTitle", "")).strip()
                if not cls_title.isdigit():
                    continue
                fdi = int(cls_title)
                if fdi not in FDI_CLASSES:
                    continue

                bm = obj.get("bitmap", {})
                if "data" not in bm:
                    continue
                origin = bm.get("origin", [0, 0])
                if not isinstance(origin, list) or len(origin) != 2:
                    continue

                try:
                    ox = int(origin[0])
                    oy = int(origin[1])
                    mask01 = decode_bitmap_mask(bm["data"])
                except Exception:
                    continue

                bbox = crop_bbox_from_mask(mask01)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                gx1 = max(0, ox + x1)
                gy1 = max(0, oy + y1)
                gx2 = min(W - 1, ox + x2)
                gy2 = min(H - 1, oy + y2)

                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                # Small padding around tooth crop
                pad = 2
                gx1 = max(0, gx1 - pad)
                gy1 = max(0, gy1 - pad)
                gx2 = min(W - 1, gx2 + pad)
                gy2 = min(H - 1, gy2 + pad)

                crop = img.crop((gx1, gy1, gx2 + 1, gy2 + 1))
                cw, ch = crop.size
                if cw < min_size or ch < min_size:
                    continue

                uid = f"{dataset_dir.name}_{ann_file.stem}_{obj_idx:02d}"
                extracted[fdi].append((crop, uid))
                total_saved += 1

    for fdi in FDI_CLASSES:
        random.shuffle(extracted[fdi])

    train_total = 0
    val_total = 0

    for fdi in FDI_CLASSES:
        items = extracted[fdi]
        if not items:
            continue

        split_idx = int(len(items) * train_ratio)
        if len(items) > 1:
            split_idx = max(1, min(split_idx, len(items) - 1))

        train_items = items[:split_idx]
        val_items = items[split_idx:]

        for i, (crop, uid) in enumerate(train_items):
            out = output_root / "train" / str(fdi) / f"{fdi}_train_{i:06d}_{uid}.jpg"
            crop.save(out, quality=95)
        for i, (crop, uid) in enumerate(val_items):
            out = output_root / "val" / str(fdi) / f"{fdi}_val_{i:06d}_{uid}.jpg"
            crop.save(out, quality=95)

        train_total += len(train_items)
        val_total += len(val_items)

    print("\nDone")
    print("Annotation files scanned:", total_ann_files)
    print("Objects scanned:", total_objects)
    print("Tooth crops extracted:", total_saved)
    print("Train images:", train_total)
    print("Val images:", val_total)
    print("Total images:", train_total + val_total)

    print("\nPer-class counts")
    for fdi in FDI_CLASSES:
        tr = len(list((output_root / "train" / str(fdi)).glob("*.jpg")))
        va = len(list((output_root / "val" / str(fdi)).glob("*.jpg")))
        if tr > 0 or va > 0:
            print(f"FDI {fdi}: train={tr} val={va} total={tr+va}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FDI multiclass dataset from bitmap annotations")
    parser.add_argument("--source-root", default="datasets/tooth_boxes")
    parser.add_argument("--output-root", default="datasets/tooth_multiclass/images")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=24)
    parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()

    prepare_dataset(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_size=args.min_size,
        clear_output=args.clear_output,
    )


if __name__ == "__main__":
    main()
