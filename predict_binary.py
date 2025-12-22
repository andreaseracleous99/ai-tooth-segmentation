import argparse
from pathlib import Path
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

# ----------------------------
# ARGUMENTS
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to image")
args = parser.parse_args()

IMAGE_PATH = Path(args.image)
MODEL_PATH = Path("models/tooth_vs_nontooth_coco.pth")

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD MODEL
# ----------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

classes = ckpt["classes"]          # ['non_tooth', 'tooth']
num_classes = len(classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

# ----------------------------
# TRANSFORMS
# ----------------------------
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ----------------------------
# LOAD IMAGE
# ----------------------------
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = Image.open(IMAGE_PATH).convert("RGB")
x = tf(img).unsqueeze(0).to(DEVICE)

# ----------------------------
# PREDICTION
# ----------------------------
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

pred_idx = probs.argmax().item()
pred_class = classes[pred_idx]
confidence = probs[pred_idx].item()

# ----------------------------
# OUTPUT
# ----------------------------
print("\n--- RESULT ---")
print(f"Image: {IMAGE_PATH}")
print(f"Prediction: {pred_class}")
print(f"Confidence: {confidence:.4f}")

print("\nAll probabilities:")
for cls, p in zip(classes, probs.tolist()):
    print(f"  {cls}: {p:.4f}")
