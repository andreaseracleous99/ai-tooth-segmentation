import json
import base64
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


IMAGES_DIR = Path("datasets/train-yolo/11/images")       
JSON_DIR   = Path("datasets/train-yolo/11/annotations")      
OUT_LABELS = Path("datasets/train-yolo/11/labels")    
OUT_LABELS.mkdir(parents=True, exist_ok=True)

CLASS_ID_BINARY = 0  

def decode_bitmap_to_mask(png_b64: str) -> np.ndarray:
    """
    Supervisely-style 'bitmap.data' is often zlib-compressed PNG bytes then base64 encoded.
    We'll try zlib first; if that fails, treat as raw base64 PNG.
    Returns: mask as uint8 array (H,W), non-zero => tooth pixels
    """
    raw = base64.b64decode(png_b64)

    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass

    im = Image.open(BytesIO(raw)).convert("L")
    return np.array(im)

def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)

def to_yolo_line(x1, y1, x2, y2, img_w, img_h, class_id):

    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2
    yc = y1 + bh / 2

    xc /= img_w
    yc /= img_h
    bw /= img_w
    bh /= img_h

    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def main():
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No json files found in {JSON_DIR}")

    print(f"[INFO] Found {len(json_files)} json files in {JSON_DIR}")

    for i, jf in enumerate(json_files, 1):
        data = json.loads(jf.read_text(encoding="utf-8"))

        img_w = int(data["size"]["width"])
        img_h = int(data["size"]["height"])

        lines = []
        for obj in data.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue

            cls = obj.get("classTitle")  # e.g. "15"
            bmp = obj.get("bitmap", {})
            origin = bmp.get("origin", [0, 0])
            x0, y0 = int(origin[0]), int(origin[1])

            mask = decode_bitmap_to_mask(bmp["data"])
            bb = mask_bbox(mask)
            if bb is None:
                continue

            mx1, my1, mx2, my2 = bb

            x1 = x0 + mx1
            y1 = y0 + my1
            x2 = x0 + mx2
            y2 = y0 + my2

            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h - 1))

            line = to_yolo_line(x1, y1, x2, y2, img_w, img_h, CLASS_ID_BINARY)
            if line:
                lines.append(line)

        out_txt = OUT_LABELS / (jf.stem + ".txt")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if i % 50 == 0 or i == len(json_files):
            print(f"[PROGRESS] {i}/{len(json_files)} done. Last labels: {out_txt.name} ({len(lines)} boxes)")

    print(f"\n[DONE] Labels saved in: {OUT_LABELS}")

if __name__ == "__main__":
    main()
