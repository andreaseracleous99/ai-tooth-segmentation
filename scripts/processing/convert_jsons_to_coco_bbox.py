import json
import base64
import zlib
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

DATASET_ROOT = Path("Datasets/43_3")
IMG_DIR = DATASET_ROOT / "images"
ANN_DIR = DATASET_ROOT / "ann"    
OUT_JSON = DATASET_ROOT / "annotations_coco_bbox.json"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]

SINGLE_CLASS = True     
MIN_AREA = 50             


def decode_bitmap_mask(bitmap_data_b64: str) -> np.ndarray:
    """
    Supervisely bitmap.data is usually zlib-compressed PNG bytes stored as base64.
    Returns a uint8 mask (0/1) for that small bitmap tile.
    """
    raw = base64.b64decode(bitmap_data_b64)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass 

    img = Image.open(BytesIO(raw)).convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    mask = (alpha > 0).astype(np.uint8)
    return mask


def bbox_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    area = int(mask01.sum())
    return x1, y1, w, h, area


def find_image_for_json(json_path: Path) -> Path | None:
    name = json_path.name

    if name.lower().endswith(".json"):
        base = name[:-5]
    else:
        base = json_path.stem

    candidate = IMG_DIR / base
    if candidate.exists():
        return candidate

    base_stem = Path(base).stem 
    for ext in IMAGE_EXTS:
        p = IMG_DIR / f"{base_stem}{ext}"
        if p.exists():
            return p

    return None

def main():
    coco = {"images": [], "annotations": [], "categories": []}

    cat_map = {}
    if SINGLE_CLASS:
        cat_map["tooth"] = 1
        coco["categories"].append({"id": 1, "name": "tooth"})

    ann_id = 1
    img_id = 1

    json_files = sorted(ANN_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {ANN_DIR}")

    for jp in json_files:
        img_path = find_image_for_json(jp)
        if img_path is None:
            print(f"[WARN] No matching image for {jp.name} (stem must match). Skipping.")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H
        })

        data = json.loads(jp.read_text(encoding="utf-8"))
        objects = data.get("objects", [])

        for obj in objects:
            cls_title = str(obj.get("classTitle", "tooth"))

            if SINGLE_CLASS:
                cat_id = cat_map["tooth"]
            else:
                if cls_title not in cat_map:
                    cat_map[cls_title] = len(cat_map) + 1
                    coco["categories"].append({"id": cat_map[cls_title], "name": cls_title})
                cat_id = cat_map[cls_title]

            if obj.get("geometryType") != "bitmap" or "bitmap" not in obj:
                continue

            bm = obj["bitmap"]
            origin = bm.get("origin", [0, 0])
            ox, oy = int(origin[0]), int(origin[1])

            mask01 = decode_bitmap_mask(bm["data"])
            bb = bbox_from_mask(mask01)
            if bb is None:
                continue

            x1, y1, w, h, area_px = bb
            if area_px < MIN_AREA:
                continue

            x = ox + x1
            y = oy + y1

            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
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
