from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import argparse

IMAGE_PATH = Path("datasets/tooth_vs_nontooth/merged/binary_patches/tooth/tooth_00000000.jpg")
RADIOGRAPH_MODEL_PATH = Path("models/radiograph_binary.pth")
TOOTH_MODEL_PATH = Path("models/tooth_vs_nontooth_binary.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Tooth classification pipeline")
    parser.add_argument(
        "--skip-radiograph-check",
        action="store_true",
        help="Skip radiograph vs non-radiograph check"
    )
    return parser.parse_args()

def load_resnet18_classifier(ckpt_path: Path):
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
def predict(model, classes, tf, img: Image.Image):
    x = tf(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    pred_idx = int(probs.argmax().item())
    return classes[pred_idx], float(probs[pred_idx].item()), {c: float(p) for c, p in zip(classes, probs.tolist())}

def main():
    args = parse_args()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH.resolve())
    if not RADIOGRAPH_MODEL_PATH.exists():
        raise FileNotFoundError(RADIOGRAPH_MODEL_PATH.resolve())
    if not TOOTH_MODEL_PATH.exists():
        raise FileNotFoundError(TOOTH_MODEL_PATH.resolve())

    print("Device:", DEVICE)
    print("Image:", IMAGE_PATH)
    print("Skip Radiograph check:", args.skip_radiograph_check)

    radiograph_model, radiograph_classes, radiograph_tf = load_resnet18_classifier(RADIOGRAPH_MODEL_PATH)
    tooth_model, tooth_classes, tooth_tf = load_resnet18_classifier(TOOTH_MODEL_PATH)

    img = Image.open(IMAGE_PATH).convert("RGB")

    # 1) Radiograph/non-radiograph
    if not args.skip_radiograph_check:
        radiograph_pred, radiograph_conf, radiograph_probs = predict(radiograph_model, radiograph_classes, radiograph_tf, img)
        print("\n--- RADIOGRAPH VS NON-RADIOGRAPH RESULT ---")
        print("Prediction:", radiograph_pred)
        print("Confidence:", f"{radiograph_conf:.4f}")
        print("All probs:", radiograph_probs)

        if radiograph_pred != "radiograph":
            print("\n[STOP] This image is NOT a radiograph patch. Tooth classifier will NOT run.")
            return
    else:
        print("\n[SKIPPED] Radiograph check")

    # 2) Tooth/non-tooth
    tooth_pred, tooth_conf, tooth_probs = predict(tooth_model, tooth_classes, tooth_tf, img)
    print("\n--- TOOTH VS NON-TOOTH RESULT ---")
    print("Prediction:", tooth_pred)
    print("Confidence:", f"{tooth_conf:.4f}")
    print("All probs:", tooth_probs)

if __name__ == "__main__":
    main()