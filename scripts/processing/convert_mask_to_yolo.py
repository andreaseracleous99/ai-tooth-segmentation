import cv2
import numpy as np
from pathlib import Path

IMAGES_DIR = Path("datasets/train-yolo/42/images")
MASKS_DIR  = Path("datasets/train-yolo/42/masks")
OUT_DIR    = Path("datasets/train-yolo/42/labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MULTICLASS = False
VALUE_TO_CLASS = {}
MIN_AREA = 200 

def find_mask_for_image(img_path: Path):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        candidate = MASKS_DIR / (img_path.stem + ext)
        if candidate.exists():
            return candidate
    return None

def bbox_to_yolo(x, y, w, h, W, H):
    xc = (x + w/2) / W
    yc = (y + h/2) / H
    bw = w / W
    bh = h / H
    return xc, yc, bw, bh

for img_path in sorted(IMAGES_DIR.glob("*")):
    if img_path.suffix.lower() not in [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]:
        continue

    mask_path = find_mask_for_image(img_path)
    if mask_path is None:
        print(f"[SKIP] no mask for {img_path.name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] unreadable image {img_path.name}")
        continue
    H, W = img.shape[:2]

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[SKIP] unreadable mask {mask_path.name}")
        continue

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    lines = []

    if not MULTICLASS:
        bin_mask = (mask > 0).astype(np.uint8) * 255

        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        for i in range(1, num):
            x, y, w_box, h_box, area = stats[i]
            if area < MIN_AREA:
                continue
            xc, yc, bw, bh = bbox_to_yolo(x, y, w_box, h_box, W, H)
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    else:
        values = np.unique(mask)
        values = [v for v in values if v != 0]  
        for v in values:
            if VALUE_TO_CLASS and v not in VALUE_TO_CLASS:
                continue
            cls = VALUE_TO_CLASS.get(int(v), int(v)) 

            bin_mask = (mask == v).astype(np.uint8) * 255
            num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            for i in range(1, num):
                x, y, w_box, h_box, area = stats[i]
                if area < MIN_AREA:
                    continue
                xc, yc, bw, bh = bbox_to_yolo(x, y, w_box, h_box, W, H)
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    out_txt = OUT_DIR / f"{img_path.stem}.txt"
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[OK] {img_path.name} -> {out_txt.name} ({len(lines)} boxes)")
