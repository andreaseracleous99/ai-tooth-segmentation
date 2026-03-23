import json
from pathlib import Path
from PIL import Image

BBOX_JSON = Path("datasets/train-yolo/3/teeth_bbox.json") 
IMAGES_DIR = Path("datasets/train-yolo/3/images") 
OUT_LABELS_DIR = Path("datasets/train-yolo/3/labels")
OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

BBOX_IS_X1Y1X2Y2 = True

def to_yolo_line(x1, y1, x2, y2, img_w, img_h, class_id=0):

    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None 

    xc = x1 + bw / 2
    yc = y1 + bh / 2

    xc /= img_w
    yc /= img_h
    bw /= img_w
    bh /= img_h

    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def main():
    data = json.loads(BBOX_JSON.read_text(encoding="utf-8"))

    possible_name_keys = ["file_name", "filename", "image", "imagePath", "name", "External ID", "external_id"]

    converted = 0
    skipped = 0

    for i, item in enumerate(data):
        img_name = None
        for k in possible_name_keys:
            if k in item and isinstance(item[k], str):
                img_name = Path(item[k]).name
                break

        if img_name is None:
        
            skipped += 1
            print(f"[SKIP] entry {i}: cannot find image filename key")
            continue

        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            skipped += 1
            print(f"[SKIP] image not found: {img_path}")
            continue

        img_w, img_h = Image.open(img_path).size

        label = item.get("Label") or item.get("label") or {}
        objects = label.get("objects", [])

        lines = []
        for obj in objects:
            bb = obj.get("bounding box") or obj.get("bbox") or obj.get("bounding_box")
            if not bb or len(bb) != 4:
                continue

            if BBOX_IS_X1Y1X2Y2:
                x1, y1, x2, y2 = bb
            else:
                x1, y1, w, h = bb
                x2, y2 = x1 + w, y1 + h

            line = to_yolo_line(x1, y1, x2, y2, img_w, img_h, class_id=0)
            if line:
                lines.append(line)

        out_txt = OUT_LABELS_DIR / (Path(img_name).stem + ".txt")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        converted += 1
        if converted % 50 == 0:
            print(f"[PROGRESS] converted {converted} images...")

    print(f"\n[DONE] Converted: {converted}")
    print(f"[DONE] Skipped: {skipped}")
    print(f"[DONE] Labels saved to: {OUT_LABELS_DIR}")

if __name__ == "__main__":
    main()
