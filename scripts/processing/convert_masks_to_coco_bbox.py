import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

DATASET_ROOT = Path("Datasets/42")
IMG_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "masks"
OUT_JSON = DATASET_ROOT / "annotations_coco_bbox.json"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]

MIN_AREA = 250        
DILATE_ITERS = 0        
ERODE_ITERS = 0        

# -------------------------
def find_mask_for_image(img_path: Path) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        mp = MASK_DIR / f"{img_path.stem}{ext}"
        if mp.exists():
            return mp
    return None

def main():
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "tooth"}]}

    ann_id = 1
    img_id = 1

    img_files = []
    for ext in IMAGE_EXTS:
        img_files += sorted(IMG_DIR.glob(f"*{ext}"))
    img_files = sorted(set(img_files))

    if not img_files:
        raise FileNotFoundError(f"No images found in: {IMG_DIR}")

    for img_path in img_files:
        mask_path = find_mask_for_image(img_path)
        if mask_path is None:
            print(f"[WARN] no mask for {img_path.name}, skipping")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H
        })

        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"[WARN] could not read mask {mask_path}, skipping")
            continue

        _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)

        if DILATE_ITERS > 0:
            m = cv2.dilate(m, None, iterations=DILATE_ITERS)
        if ERODE_ITERS > 0:
            m = cv2.erode(m, None, iterations=ERODE_ITERS)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i].tolist()
            if area < MIN_AREA:
                continue

            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    OUT_JSON.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"[DONE] wrote: {OUT_JSON}")
    print(f"Images: {len(coco['images'])} | Annotations: {len(coco['annotations'])}")

if __name__ == "__main__":
    main()
