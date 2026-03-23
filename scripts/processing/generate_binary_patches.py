import json
import random
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("dataset")
IMG_DIR = DATASET_ROOT / "images"
ANN_PATH = DATASET_ROOT / "annotations" / "_annotations.coco.json"
OUT_DIR = DATASET_ROOT / "binary_patches"

(OUT_DIR / "tooth").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "non_tooth").mkdir(parents=True, exist_ok=True)

with ANN_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = {im["id"]: im for im in coco["images"]}

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB-xA)*(yB-yA)
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / union

tooth_boxes = {}
for ann in coco["annotations"]:
    tooth_boxes.setdefault(ann["image_id"], []).append(ann["bbox"])

tooth_count = 0
non_tooth_count = 0

for img_id, img_info in images.items():
    img = Image.open(IMG_DIR / img_info["file_name"]).convert("RGB")
    W, H = img.size

    for box in tooth_boxes.get(img_id, []):
        x, y, w, h = map(int, box)
        patch = img.crop((x, y, x+w, y+h))
        patch.save(OUT_DIR / "tooth" / f"tooth_{tooth_count}.jpg")
        tooth_count += 1

    for _ in range(5):
        bw, bh = 120, 120
        rx = random.randint(0, W - bw)
        ry = random.randint(0, H - bh)
        rand_box = [rx, ry, bw, bh]

        if all(iou(rand_box, b) < 0.05 for b in tooth_boxes.get(img_id, [])):
            patch = img.crop((rx, ry, rx+bw, ry+bh))
            patch.save(OUT_DIR / "non_tooth" / f"bg_{non_tooth_count}.jpg")
            non_tooth_count += 1

print("Tooth patches:", tooth_count)
print("Non-tooth patches:", non_tooth_count)
