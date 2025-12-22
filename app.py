import io
import cv2
import torch
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torch import nn

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = Path("models/tooth_vs_nontooth_resnet18_best.pth")
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODEL
# -----------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]
tooth_idx = classes.index("tooth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------
# SLIDING WINDOW INFERENCE
# -----------------------------
def detect_teeth(img_bgr, stride, threshold):
    h, w = img_bgr.shape[:2]
    PATCH = IMG_SIZE

    boxes, scores = [], []

    for y in range(0, h - PATCH + 1, stride):
        for x in range(0, w - PATCH + 1, stride):
            patch = img_bgr[y:y+PATCH, x:x+PATCH]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(patch)

            x_tensor = tf(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0]

            score = float(probs[tooth_idx])
            if score >= threshold:
                boxes.append((x, y, x+PATCH, y+PATCH))
                scores.append(score)

    return boxes, scores

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Tooth Detection", layout="wide")
st.title("Panoramic Tooth Detection (Binary)")

uploaded = st.file_uploader("Upload panoramic X-ray", type=["jpg", "jpeg", "png"])

stride = st.slider("Stride (lower = more boxes)", 16, 256, 64, 8)
threshold = st.slider("Tooth confidence threshold", 0.5, 0.99, 0.85, 0.01)

if uploaded:
    img_bytes = uploaded.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(pil)
    st.image(pil, caption="Input Image", use_container_width=True)

    if st.button("Run Detection"):
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        boxes, scores = detect_teeth(img_bgr, stride, threshold)

        out = img_bgr.copy()
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, f"{sc:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        st.image(out_rgb, caption=f"Detected {len(boxes)} tooth regions",
                 use_container_width=True)
