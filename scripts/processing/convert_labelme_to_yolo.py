import json
from pathlib import Path
import cv2

IMAGES_DIR = Path("datasets/train-yolo/43_3/images")
JSON_DIR   = Path("datasets/train-yolo/43_3/annotations")
OUT_DIR    = Path("datasets/train-yolo/43_3/labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["molar", "premolar", "canine", "incisor"] 
NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}

def get_image_size(labelme_data, json_path: Path):
    H = labelme_data.get("imageHeight", None)
    W = labelme_data.get("imageWidth", None)
    if H is not None and W is not None:
        return int(W), int(H)

    img_path = None
    if "imagePath" in labelme_data and labelme_data["imagePath"]:
        candidate = (json_path.parent / labelme_data["imagePath"]).resolve()
        if candidate.exists():
            img_path = candidate

    if img_path is None:
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            candidate = IMAGES_DIR / (json_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

    if img_path is None:
        raise FileNotFoundError(f"Could not find image for {json_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")
    H, W = img.shape[:2]
    return int(W), int(H)

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

for json_path in sorted(JSON_DIR.glob("*.json")):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    W, H = get_image_size(data, json_path)

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        shape_type = shape.get("shape_type", "")

        if shape_type != "polygon":
            continue

        if label not in NAME_TO_ID:
            continue

        cls_id = NAME_TO_ID[label]
        pts = shape.get("points", [])

        if not pts or len(pts) < 3:
            continue

        coords = []
        for x, y in pts:
            nx = clamp01(float(x) / W)
            ny = clamp01(float(y) / H)
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")

        lines.append(f"{cls_id} " + " ".join(coords))

    out_txt = OUT_DIR / f"{json_path.stem}.txt"
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[OK] {json_path.name} -> {out_txt.name} ({len(lines)} segments)")
