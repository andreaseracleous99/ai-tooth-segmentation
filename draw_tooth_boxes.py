import cv2
from ultralytics import YOLO
from pathlib import Path

MODEL = "models/tooth_boxes_v2.pt"
IMG = "datasets/tooth_vs_nontooth/merged/binary_patches/non_tooth/orange.jpg"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "tiled_pred.jpg"

model = YOLO(MODEL)

img = cv2.imread(IMG)
H, W = img.shape[:2]

tile = 1024
overlap = 256

all_boxes = []

for y in range(0, H, tile - overlap):
    for x in range(0, W, tile - overlap):
        x2 = min(x + tile, W)
        y2 = min(y + tile, H)
        crop = img[y:y2, x:x2]

        results = model.predict(crop, imgsz=1024, conf=0.5, iou=0.35, verbose=False)[0]
        if results.boxes is None:
            continue

        for b in results.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            
            xyxy[0] += x; xyxy[2] += x
            xyxy[1] += y; xyxy[3] += y
            all_boxes.append((xyxy, conf))

out = img.copy()
for (xyxy, conf) in all_boxes:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imwrite(OUT, out)
print("Saved:", OUT)