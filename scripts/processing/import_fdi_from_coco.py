import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

FDI_CLASSES = {
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
}


def count_existing(target_root: Path):
    counts = {}
    for fdi in sorted(FDI_CLASSES):
        tr = len(list((target_root / "train" / str(fdi)).glob("*.jpg")))
        va = len(list((target_root / "val" / str(fdi)).glob("*.jpg")))
        counts[fdi] = tr + va
    return counts


def resolve_image_path(images_dir: Path, file_name: str) -> Path | None:
    p1 = images_dir / file_name
    if p1.exists():
        return p1

    p2 = images_dir / Path(file_name).name
    if p2.exists():
        return p2

    # final fallback: locate by basename
    matches = list(images_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Import FDI crops from COCO into multiclass dataset")
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--target-root", default="datasets/tooth_multiclass_fdi/images")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=24)
    parser.add_argument("--fill-missing-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    target_root = Path(args.target_root)

    for split in ["train", "val"]:
        for fdi in sorted(FDI_CLASSES):
            (target_root / split / str(fdi)).mkdir(parents=True, exist_ok=True)

    existing = count_existing(target_root)
    missing = {k for k, v in existing.items() if v == 0}

    data = json.loads(coco_json.read_text(encoding="utf-8"))

    category_id_to_fdi = {}
    for cat in data.get("categories", []):
        name = str(cat.get("name", "")).strip()
        if name.isdigit():
            fdi = int(name)
            if fdi in FDI_CLASSES:
                category_id_to_fdi[int(cat["id"])] = fdi

    images = {int(i["id"]): i for i in data.get("images", [])}
    annotations = data.get("annotations", [])

    extracted = defaultdict(list)
    skipped_no_img = 0
    skipped_size = 0

    for ann in annotations:
        cat_id = int(ann.get("category_id", -1))
        if cat_id not in category_id_to_fdi:
            continue

        fdi = category_id_to_fdi[cat_id]
        if args.fill_missing_only and fdi not in missing:
            continue

        img_meta = images.get(int(ann.get("image_id", -1)))
        if not img_meta:
            continue

        img_path = resolve_image_path(images_dir, str(img_meta.get("file_name", "")))
        if img_path is None:
            skipped_no_img += 1
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size

                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                x1 = max(0, int(x) - 2)
                y1 = max(0, int(y) - 2)
                x2 = min(W - 1, int(x + w) + 2)
                y2 = min(H - 1, int(y + h) + 2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = im.crop((x1, y1, x2 + 1, y2 + 1))
                cw, ch = crop.size
                if cw < args.min_size or ch < args.min_size:
                    skipped_size += 1
                    continue

                uid = f"{Path(img_meta.get('file_name', 'img')).stem}_{ann.get('id', 0)}"
                extracted[fdi].append((crop, uid))
        except Exception:
            continue

    added_train = 0
    added_val = 0

    for fdi in sorted(FDI_CLASSES):
        items = extracted[fdi]
        if not items:
            continue

        random.shuffle(items)
        split = int(len(items) * args.train_ratio)
        if len(items) > 1:
            split = max(1, min(split, len(items) - 1))

        tr_items = items[:split]
        va_items = items[split:]

        train_dir = target_root / "train" / str(fdi)
        val_dir = target_root / "val" / str(fdi)

        offset_tr = len(list(train_dir.glob("*.jpg")))
        offset_va = len(list(val_dir.glob("*.jpg")))

        for i, (crop, uid) in enumerate(tr_items):
            out = train_dir / f"{fdi}_coco_tr_{offset_tr + i:06d}_{uid}.jpg"
            crop.save(out, quality=95)
            added_train += 1

        for i, (crop, uid) in enumerate(va_items):
            out = val_dir / f"{fdi}_coco_va_{offset_va + i:06d}_{uid}.jpg"
            crop.save(out, quality=95)
            added_val += 1

    final_counts = count_existing(target_root)
    final_missing = [f for f, c in final_counts.items() if c == 0]

    print("Import complete")
    print("added_train", added_train)
    print("added_val", added_val)
    print("added_total", added_train + added_val)
    print("skipped_no_img", skipped_no_img)
    print("skipped_size", skipped_size)
    print("missing_before", sorted(missing))
    print("missing_after", final_missing)


if __name__ == "__main__":
    main()
