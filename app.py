# ============================
# IMPORTS
# ============================
import io
from pathlib import Path

import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from ultralytics import YOLO


# ============================
# CONFIG (EDIT PATHS HERE)
# ============================
RADIOGRAPH_MODEL_PATH = Path("models/radiograph_binary.pth")
TOOTH_MODEL_PATH      = Path("models/tooth_vs_nontooth_binary.pth")
YOLO_MODEL_PATH       = Path("models/tooth_boxes_v2.pt")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# HELPERS - CLASSIFIERS
# ============================
@st.cache_resource
def load_resnet18_classifier(ckpt_path: str):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("mean", (0.485, 0.456, 0.406))
    std  = ckpt.get("std",  (0.229, 0.224, 0.225))

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, classes, tf


@torch.no_grad()
def predict_classifier(model, classes, tf, img: Image.Image):
    x = tf(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    pred_idx = int(probs.argmax().item())
    pred_class = classes[pred_idx]
    conf = float(probs[pred_idx].item())
    probs_dict = {c: float(p) for c, p in zip(classes, probs.tolist())}
    return pred_class, conf, probs_dict


# ============================
# HELPERS - YOLO
# ============================
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)


def run_yolo_and_save(model: YOLO, pil_img: Image.Image, conf: float, imgsz: int):
    results = model.predict(
        source=pil_img,
        conf=conf,
        imgsz=imgsz,
        save=False,
        verbose=False,
        hide_labels=True,
        hide_conf=True,
    )

    r = results[0]
    plotted = r.plot(labels=False, conf=False)  # numpy BGR

    out_path = OUTPUT_DIR / "tooth_boxes_prediction.jpg"
    plotted_rgb = plotted[..., ::-1]  # BGR -> RGB
    Image.fromarray(plotted_rgb).save(out_path)

    return out_path


# ============================
# UI
# ============================
st.set_page_config(page_title="Tooth AI", layout="wide")
st.title("Tooth AI — Binary Classification + Tooth Boxes")
st.write(f"Device: **{DEVICE}**")

# Sanity checks
missing = []
if not RADIOGRAPH_MODEL_PATH.exists():
    missing.append(str(RADIOGRAPH_MODEL_PATH))
if not TOOTH_MODEL_PATH.exists():
    missing.append(str(TOOTH_MODEL_PATH))
if not YOLO_MODEL_PATH.exists():
    missing.append(str(YOLO_MODEL_PATH))

if missing:
    st.error("Missing required model files:\n" + "\n".join(missing))
    st.stop()


tab1, tab2 = st.tabs([
    "1) Binary: tooth vs non-tooth",
    "2) Boxes: detect teeth",
])

# ============================
# TAB 1 - Binary Classification
# ============================
with tab1:
    uploaded_bin = st.file_uploader(
        "Upload an image for Binary Classification",
        type=["jpg", "jpeg", "png"],
        key="uploader_binary",
    )

    if not uploaded_bin:
        st.info("Upload an image to start.")
        st.stop()

    pil_img = Image.open(io.BytesIO(uploaded_bin.read())).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(pil_img, use_container_width=True)

    with col2:
        skip_gate = st.checkbox("Skip radiograph check")

        if st.button("Run Binary Classification"):
            with st.spinner("Running models..."):
                gate_model, gate_classes, gate_tf = load_resnet18_classifier(
                    str(RADIOGRAPH_MODEL_PATH)
                )
                tooth_model, tooth_classes, tooth_tf = load_resnet18_classifier(
                    str(TOOTH_MODEL_PATH)
                )

                if not skip_gate:
                    gate_pred, gate_conf, gate_probs = predict_classifier(
                        gate_model, gate_classes, gate_tf, pil_img
                    )
                    st.markdown("### Radiograph Gate")
                    st.write("Prediction:", gate_pred)
                    st.write("Confidence:", f"{gate_conf:.4f}")
                    st.json(gate_probs)

                    if gate_pred != "radiograph":
                        st.warning("STOP: Not a radiograph.")
                        st.stop()

                tooth_pred, tooth_conf, tooth_probs = predict_classifier(
                    tooth_model, tooth_classes, tooth_tf, pil_img
                )

                st.markdown("### Tooth vs Non-tooth")
                st.write("Prediction:", tooth_pred)
                st.write("Confidence:", f"{tooth_conf:.4f}")
                st.json(tooth_probs)


# ============================
# TAB 2 - YOLO Boxes
# ============================
with tab2:
    uploaded_yolo = st.file_uploader(
        "Upload an image for Tooth Boxes",
        type=["jpg", "jpeg", "png"],
        key="uploader_yolo",
    )

    if not uploaded_yolo:
        st.info("Upload an image to start.")
        st.stop()

    pil_img = Image.open(io.BytesIO(uploaded_yolo.read())).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(pil_img, use_container_width=True)

    with col2:
        conf = st.slider("Confidence threshold", 0.01, 0.99, 0.25, 0.01)
        imgsz = st.selectbox("Image size (imgsz)", [640, 768, 896, 1024, 1280], index=3)

        if st.button("Draw Tooth Boxes"):
            with st.spinner("Running YOLO..."):
                yolo = load_yolo(str(YOLO_MODEL_PATH))
                out_path = run_yolo_and_save(
                    yolo, pil_img, conf=conf, imgsz=imgsz
                )

            st.image(str(out_path), use_container_width=True)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download result image",
                    data=f.read(),
                    file_name=out_path.name,
                    mime="image/jpeg",
                )
