import json
from pathlib import Path
from collections import defaultdict
import numpy as np

COCO_JSON = Path("../../datasets/train-yolo/27/_annotations.coco.json") 
IMAGES_DIR = Path("../../datasets/train-yolo/27/images")            
OUT_LABELS_DIR = Path("../../datasets/train-yolo/27/labels")      

BINARY_MODE = True     
CATEGORY_ID_TO_YOLO = {}  
ALLOWED_CATEGORY_IDS = None 

def bbox_from_segmentation(segmentation):
    """
    COCO polygon segmentation => bbox [x, y, w, h]
    segmentation can be:
      - list of lists (polygons)
      - RLE dict (not handled here)
    """
    if not segmentation:
        return None

    if isinstance(segmentation, list):
        xs, ys = [], []
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])
        if not xs or not ys:
            return None
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    if isinstance(segmentation, dict):
        return None

    return None

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox: [x_min, y_min, width, height] in pixels
    YOLO: x_center/img_w y_center/img_h w/img_w h/img_h
    """
    x, y, w, h = bbox
    x_c = x + w / 2.0
    y_c = y + h / 2.0

    x_c /= img_w
    y_c /= img_h
    w /= img_w
    h /= img_h

    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return x_c, y_c, w, h

def main():
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))

    img_meta = {}
    for im in coco.get("images", []):
        img_meta[im["id"]] = (im["file_name"], int(im["width"]), int(im["height"]))

    categories = coco.get("categories", [])
    cat_ids = sorted([c["id"] for c in categories])

    if not BINARY_MODE:
        if CATEGORY_ID_TO_YOLO:
            cat_map = dict(CATEGORY_ID_TO_YOLO)
        else:
            cat_map = {cid: i for i, cid in enumerate(cat_ids)}
        print("[INFO] Multiclass mode")
        print("[INFO] Category mapping (COCO -> YOLO):", cat_map)
    else:
        cat_map = None
        print("[INFO] Binary mode: all objects -> class 0 (tooth)")

    ann_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue

        cid = ann.get("category_id")
        if ALLOWED_CATEGORY_IDS is not None and cid not in ALLOWED_CATEGORY_IDS:
            continue

        ann_by_image[ann["image_id"]].append(ann)

    total_imgs = len(img_meta)
    total_anns = sum(len(v) for v in ann_by_image.values())
    print(f"[INFO] Images: {total_imgs}")
    print(f"[INFO] Annotations (after filters): {total_anns}")

    written = 0
    for image_id, (file_name, w, h) in img_meta.items():
        anns = ann_by_image.get(image_id, [])
        lines = []

        for ann in anns:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
              
                bbox = bbox_from_segmentation(ann.get("segmentation"))

            if not bbox:
                continue

            if bbox[2] <= 1 or bbox[3] <= 1:
                continue

            if BINARY_MODE:
                yolo_cls = 0
            else:
                cid = ann["category_id"]
                if cid not in cat_map:
                    continue
                yolo_cls = cat_map[cid]

            x_c, y_c, bw, bh = coco_bbox_to_yolo(bbox, w, h)
            lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        out_txt = OUT_LABELS_DIR / (Path(file_name).stem + ".txt")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        written += 1
        if written % 50 == 0 or written == total_imgs:
            print(f"[PROGRESS] {written}/{total_imgs} label files written. Last: {out_txt.name} ({len(lines)} lines)")

    print(f"\n[DONE] YOLO labels saved to: {OUT_LABELS_DIR}")

if __name__ == "__main__":
    main()
