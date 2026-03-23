import json
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("Datasets/30")
IMG_DIR = DATASET_ROOT / "images"
LAB_DIR = DATASET_ROOT / "labels"
OUT_JSON = DATASET_ROOT / "annotations_coco_bbox.json"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]

SINGLE_CLASS = True

def find_image_for_label(label_path: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def yolo_line_to_xywh(line: str, W: int, H: int):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls, xc, yc, w, h = parts
    xc = float(xc); yc = float(yc); w = float(w); h = float(h)

    bw = w * W
    bh = h * H
    x = (xc * W) - (bw / 2)
    y = (yc * H) - (bh / 2)

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    bw = max(1, min(bw, W - x))
    bh = max(1, min(bh, H - y))

    return int(round(x)), int(round(y)), int(round(bw)), int(round(bh))

def main():
    coco = {"images": [], "annotations": [], "categories": []}

    if SINGLE_CLASS:
        coco["categories"] = [{"id": 1, "name": "tooth"}]

    ann_id = 1
    img_id = 1

    label_files = sorted(LAB_DIR.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label .txt files found in {LAB_DIR}")

    for lp in label_files:
        img_path = find_image_for_label(lp)
        if img_path is None:
            print(f"[WARN] No image found for label {lp.name}")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H
        })

        lines = lp.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            if not line.strip():
                continue

            xywh = yolo_line_to_xywh(line, W, H)
            if xywh is None:
                continue

            x, y, w, h = xywh

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1 if SINGLE_CLASS else int(line.split()[0]) + 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    OUT_JSON.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"[DONE] wrote: {OUT_JSON}")
    print(f"Images: {len(coco['images'])} | Annotations: {len(coco['annotations'])} | Categories: {len(coco['categories'])}")

if __name__ == "__main__":
    main()
