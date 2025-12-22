import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from torchvision.ops import nms as tv_nms
from torch import nn
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = Path("models/tooth_vs_nontooth_resnet18_best.pth")
INPUT_IMAGE = Path("datasets/train/11/images/1.jpg")
OUT_IMAGE = Path("outputs/panoramic_predictions.jpg")
OUT_IMAGE.parent.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 224
STRIDE = 56

THRESH = 0.90
NMS_IOU = 0.25
TOPK = 80

ROI_Y_MIN = 0.35
ROI_Y_MAX = 0.80

# ----------------------------
# LOAD MODEL
# ----------------------------
ckpt = torch.load(MODEL_PATH, map_location="cpu")
classes = ckpt["classes"]
if "tooth" not in classes:
    raise ValueError(f"'tooth' not found in classes: {classes}")
tooth_idx = classes.index("tooth")

mean = ckpt.get("mean", (0.485, 0.456, 0.406))
std = ckpt.get("std", (0.229, 0.224, 0.225))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE)
model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),  # IMPORTANT
])

# ----------------------------
# LOAD IMAGE
# ----------------------------
img_bgr = cv2.imread(str(INPUT_IMAGE))
if img_bgr is None:
    raise FileNotFoundError(f"Could not read: {INPUT_IMAGE}")

h, w = img_bgr.shape[:2]
print(f"[INFO] Loaded: {INPUT_IMAGE} ({w}x{h})")
print(f"[INFO] patch={PATCH_SIZE}, stride={STRIDE}, thresh={THRESH}, ROI={ROI_Y_MIN:.2f}-{ROI_Y_MAX:.2f}")

pad = PATCH_SIZE // 2
img_pad = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
hp, wp = img_pad.shape[:2]

roi_y1 = int((ROI_Y_MIN * h) + pad)
roi_y2 = int((ROI_Y_MAX * h) + pad)

boxes = []
scores = []

# ----------------------------
# SLIDING WINDOW INFERENCE
# ----------------------------
count = 0
kept = 0

for y in range(roi_y1, roi_y2 - PATCH_SIZE + 1, STRIDE):
    for x in range(0, wp - PATCH_SIZE + 1, STRIDE):
        patch = img_pad[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(patch_rgb)

        x_tensor = tf(pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            tooth_prob = float(probs[tooth_idx].item())

        count += 1
        if count % 500 == 0:
            print(f"[PROGRESS] scanned {count} patches | kept {kept}")

        if tooth_prob >= THRESH:
            x1 = x - pad
            y1 = y - pad
            x2 = x1 + PATCH_SIZE
            y2 = y1 + PATCH_SIZE

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            boxes.append([x1, y1, x2, y2])
            scores.append(tooth_prob)
            kept += 1

print(f"[INFO] Raw candidates above threshold: {len(boxes)}")

if len(boxes) == 0:
    print("[DONE] No boxes found. Lower THRESH (e.g. 0.85) or widen ROI.")
    cv2.imwrite(str(OUT_IMAGE), img_bgr)
    print(f"[DONE] Saved: {OUT_IMAGE}")
    raise SystemExit

# ----------------------------
# TOP-K filter BEFORE NMS
# ----------------------------
boxes_np = np.array(boxes, dtype=np.float32)
scores_np = np.array(scores, dtype=np.float32)

order = scores_np.argsort()[::-1]
if len(order) > TOPK:
    order = order[:TOPK]
boxes_np = boxes_np[order]
scores_np = scores_np[order]

# ----------------------------
# NMS
# ----------------------------
boxes_t = torch.tensor(boxes_np, dtype=torch.float32)
scores_t = torch.tensor(scores_np, dtype=torch.float32)

keep_idx = tv_nms(boxes_t, scores_t, NMS_IOU).cpu().numpy().tolist()
boxes_nms = boxes_np[keep_idx]
scores_nms = scores_np[keep_idx]

print(f"[INFO] After TOPK({TOPK}) + NMS(iou={NMS_IOU}): {len(boxes_nms)} boxes")

# ----------------------------
# DRAW + SAVE
# ----------------------------
out = img_bgr.copy()
for (x1, y1, x2, y2), sc in zip(boxes_nms.astype(int), scores_nms):
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"{sc:.2f}", (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite(str(OUT_IMAGE), out)
print(f"[DONE] Saved: {OUT_IMAGE}")