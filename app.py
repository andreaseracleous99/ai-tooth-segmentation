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
import pandas as pd

from ultralytics import YOLO
from multiclass_pipeline import ToothDetectionPipeline, format_results
from tooth_system import ToothGroup, get_tooth_info


# ============================
# CONFIG (EDIT PATHS HERE)
# ============================
RADIOGRAPH_MODEL_PATH = Path("models/radiograph_binary.pth")
TOOTH_MODEL_PATH      = Path("models/tooth_vs_nontooth_binary.pth")
YOLO_MODEL_PATH       = Path("models/tooth_boxes_v2.pt")
MULTICLASS_MODEL_PATH = Path("models/tooth_multiclass.pth")

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

    arch = ckpt.get("arch", "resnet18")
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(classes))
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {arch}")
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


def classify_patches(pil_img: Image.Image, model, classes, tf, patch_size=120, stride=60):
    W, H = pil_img.size
    tooth_count = 0
    total_patches = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = pil_img.crop((x, y, x + patch_size, y + patch_size))
            pred, _, _ = predict_classifier(model, classes, tf, patch)
            if pred == "tooth":
                tooth_count += 1
            total_patches += 1
    tooth_ratio = tooth_count / total_patches if total_patches > 0 else 0
    return "tooth" if tooth_ratio > 0.5 else "non_tooth", tooth_ratio


# ============================
# HELPERS - YOLO
# ============================
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)


@st.cache_resource
def load_multiclass_pipeline(radiograph_model: str, yolo_model: str, multiclass_model: str):
    """Load the complete multiclass tooth detection pipeline"""
    return ToothDetectionPipeline(
        radiograph_model_path=radiograph_model,
        yolo_model_path=yolo_model,
        multiclass_model_path=multiclass_model,
        device=DEVICE
    )


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
st.title("🦷 Tooth AI — Multiclass Segmentation & Classification")
st.write(f"Device: **{DEVICE}**")

# Sanity checks
missing = []
if not RADIOGRAPH_MODEL_PATH.exists():
    missing.append(str(RADIOGRAPH_MODEL_PATH))
if not TOOTH_MODEL_PATH.exists():
    missing.append(str(TOOTH_MODEL_PATH))
if not YOLO_MODEL_PATH.exists():
    missing.append(str(YOLO_MODEL_PATH))
if not MULTICLASS_MODEL_PATH.exists():
    missing.append(str(MULTICLASS_MODEL_PATH))

if missing:
    st.error("Missing required model files:\n" + "\n".join(missing))
    st.info("Please ensure the following models are available:")
    st.info("- radiograph_binary.pth (radiograph validation)")
    st.info("- tooth_boxes_v2.pt (tooth detection)")
    st.info("- tooth_multiclass.pth (32-class tooth classification)")
    st.stop()


tab1, tab2, tab3 = st.tabs([
    "1) Binary: tooth vs non-tooth",
    "2) Boxes: detect teeth",
    "3) Multiclass: 32-tooth segmentation",
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

                # Patch-based classification
                patch_pred, tooth_ratio = classify_patches(pil_img, tooth_model, tooth_classes, tooth_tf)

                st.markdown("### Tooth vs Non-tooth (Full Image)")
                st.write("Prediction:", tooth_pred)
                st.write("Confidence:", f"{tooth_conf:.4f}")
                st.json(tooth_probs)

                st.markdown("### Tooth vs Non-tooth (Patch-Based)")
                st.write("Prediction:", patch_pred)
                st.write("Tooth Ratio:", f"{tooth_ratio:.2f}")
                st.write("Total Patches:", int(tooth_ratio * 100), "/", 100)  # approximate


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


# ============================
# TAB 3 - Multiclass 32-Tooth Segmentation
# ============================
with tab3:
    st.subheader("🦷 32-Tooth Multiclass Segmentation & Classification (FDI System)")
    st.write("""
    This system identifies and classifies individual teeth using the FDI numbering system:
    - **11-18, 21-28**: Upper teeth (maxillary)
    - **31-38, 41-48**: Lower teeth (mandibular)
    
    Each tooth is classified into groups:
    - **Incisors** (8): Central & lateral incisors for cutting
    - **Canines** (4): Pointed teeth for tearing  
    - **Premolars** (8): Grinding teeth
    - **Molars** (12): Large grinding teeth
    """)
    
    uploaded_multiclass = st.file_uploader(
        "Upload an intraoral photograph",
        type=["jpg", "jpeg", "png"],
        key="uploader_multiclass",
    )

    if not uploaded_multiclass:
        st.info("Upload an intraoral photograph to begin tooth analysis.")
        st.stop()

    pil_img = Image.open(io.BytesIO(uploaded_multiclass.read())).convert("RGB")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("Input Image")
        st.image(pil_img, use_container_width=True)

    with col2:
        st.subheader("Settings")
        conf_threshold = st.slider("Detection Confidence", 0.01, 0.99, 0.25, 0.01)
        imgsz = st.selectbox("Image Size (imgsz)", [640, 768, 896, 1024, 1280], index=3)
        
        run_analysis = st.button("🔍 Run Multiclass Analysis", type="primary", use_container_width=True)

    if run_analysis:
        with st.spinner("Running multiclass tooth segmentation and classification..."):
            try:
                # Load pipeline
                pipeline = load_multiclass_pipeline(
                    str(RADIOGRAPH_MODEL_PATH),
                    str(YOLO_MODEL_PATH),
                    str(MULTICLASS_MODEL_PATH)
                )
                
                # Run detection and classification
                results = pipeline.detect_and_classify(
                    pil_img,
                    conf_threshold=conf_threshold,
                    imgsz=imgsz
                )
                
                # Display results
                st.success(f"Analysis complete! Detected {len(results['detections'])} teeth.")
                
                # Print formatted results
                st.markdown("---")
                st.text(format_results(results))
                
                # Radiograph validation
                col_radio1, col_radio2 = st.columns(2)
                with col_radio1:
                    if results["is_radiograph"]:
                        st.markdown(f"✅ **Valid Intraoral Image** ({results['radiograph_confidence']:.1%} confidence)")
                    else:
                        st.warning(f"⚠️ May not be intraoral ({results['radiograph_confidence']:.1%} confidence)")
                
                # Summary by group
                st.markdown("### Teeth Summary by Group")
                
                group_summary = []
                for group in [ToothGroup.INCISOR, ToothGroup.CANINE, ToothGroup.PREMOLAR, ToothGroup.MOLAR]:
                    teeth = results["teeth_by_group"].get(group, [])
                    if teeth:
                        fdi_nums = sorted([t.fdi_number for t in teeth])
                        avg_conf = sum([t.confidence for t in teeth]) / len(teeth)
                        group_summary.append({
                            "Group": group.value.upper(),
                            "Count": len(teeth),
                            "Teeth": ", ".join(map(str, fdi_nums)),
                            "Avg Confidence": f"{avg_conf:.1%}"
                        })
                
                if group_summary:
                    df = pd.DataFrame(group_summary)
                    st.dataframe(df, use_container_width=True)
                
                # Detailed detections table
                if results["detections"]:
                    st.markdown("### Detailed Detections")
                    
                    detections_data = []
                    for det in sorted(results["detections"], key=lambda x: x.fdi_number):
                        info = det.tooth_info
                        detections_data.append({
                            "FDI": det.fdi_number,
                            "Name": info["name"],
                            "Group": det.group.value,
                            "Arch": info["arch"].value,
                            "Confidence": f"{det.confidence:.3f}",
                            "Position": f"({det.bbox[0]}, {det.bbox[1]})",
                            "Size": f"{det.bbox[2]}×{det.bbox[3]}"
                        })
                    
                    det_df = pd.DataFrame(detections_data)
                    st.dataframe(det_df, use_container_width=True)
                    
                    # Download results as CSV
                    csv = det_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="tooth_segmentation_results.csv",
                        mime="text/csv",
                    )
                
                # Export detection results
                st.markdown("### Export Results")
                st.json({
                    "radiograph": results["is_radiograph"],
                    "radiograph_confidence": float(results["radiograph_confidence"]),
                    "total_teeth": len(results["detections"]),
                    "detections": [det.to_dict() for det in results["detections"]],
                })
                
            except FileNotFoundError as e:
                st.error(f"❌ Model file not found: {e}")
                st.info("Please ensure 'tooth_multiclass.pth' is trained and available in the models/ directory.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

