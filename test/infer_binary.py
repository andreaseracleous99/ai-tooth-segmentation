from pathlib import Path
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

MODEL_PATH = Path("models/tooth_vs_nontooth_resnet18.pth")
TEST_IMAGE = Path("/datasets/train/11/binary_patches/tooth/tooth_0000000.jpg") 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    return ckpt, classes, img_size

def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE.resolve()}")

    ckpt, classes, img_size = load_checkpoint(MODEL_PATH)

    print("Loaded model:", MODEL_PATH)
    print("Classes:", classes)
    print("Image size:", img_size)
    print("Device:", DEVICE)

    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(TEST_IMAGE).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)  # shape: [1, 3, H, W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # shape: [num_classes]
        pred_idx = int(torch.argmax(probs).item())
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx].item())

    print("\n--- RESULT ---")
    print("Image:", TEST_IMAGE)
    print("Prediction:", pred_class)
    print("Confidence:", round(confidence, 4))
    print("All probs:", {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))})

if __name__ == "__main__":
    main()
