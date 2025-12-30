import argparse
from pathlib import Path
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to image")
parser.add_argument("--model", default="models/tooth_vs_nontooth_binary_resnet18.pth", help="Path to .pth model")
args = parser.parse_args()

IMAGE_PATH = Path(args.image)
MODEL_PATH = Path(args.model)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH.resolve()}")

ckpt = torch.load(MODEL_PATH, map_location="cpu")

classes = ckpt["classes"] 
num_classes = len(classes)
img_size = ckpt.get("img_size", 224)
mean = ckpt.get("mean", (0.485, 0.456, 0.406))
std  = ckpt.get("std",  (0.229, 0.224, 0.225))

print("Loaded model:", MODEL_PATH)
print("Device:", DEVICE)
print("Classes:", classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

img = Image.open(IMAGE_PATH).convert("RGB")
x = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

pred_idx = int(probs.argmax().item())
pred_class = classes[pred_idx]
confidence = float(probs[pred_idx].item())

print("\n--- RESULT ---")
print(f"Image: {IMAGE_PATH}")
print(f"Prediction: {pred_class}")
print(f"Confidence: {confidence:.4f}")

print("\nAll probabilities:")
for cls, p in zip(classes, probs.tolist()):
    print(f"  {cls}: {p:.4f}")
