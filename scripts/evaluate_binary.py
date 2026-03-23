from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from torchvision import models
from collections import Counter

TEST_DIR = Path("../datasets/tooth_vs_nontooth/split/test")
MODEL_PATH = Path("../models/tooth_vs_nontooth_binary.pth")

ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
classes = ckpt["classes"]
img_size = ckpt.get("img_size", 224)
mean = ckpt.get("mean", (0.485, 0.456, 0.406))
std = ckpt.get("std", (0.229, 0.224, 0.225))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model.eval()

tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dataset = datasets.ImageFolder(TEST_DIR, transform=tf)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())

# Compute accuracy
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true)

# Confusion matrix
cm = [[0, 0], [0, 0]]
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

# Precision, recall, f1 for each class
def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

# For class 0 (non_tooth)
tp0 = cm[0][0]
fp0 = cm[1][0]
fn0 = cm[0][1]
prec0, rec0, f10 = calc_metrics(tp0, fp0, fn0)

# For class 1 (tooth)
tp1 = cm[1][1]
fp1 = cm[0][1]
fn1 = cm[1][0]
prec1, rec1, f11 = calc_metrics(tp1, fp1, fn1)

print("Overall Accuracy:", accuracy)
print("Confusion Matrix:")
print("[[TN, FP], [FN, TP]] =", cm)
print(f"Non-tooth: Precision={prec0:.3f}, Recall={rec0:.3f}, F1={f10:.3f}")
print(f"Tooth: Precision={prec1:.3f}, Recall={rec1:.3f}, F1={f11:.3f}")

# Save to file
with open("binary_evaluation_results.txt", "w") as f:
    f.write(f"Overall Accuracy: {accuracy}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"[[TN, FP], [FN, TP]] = {cm}\n")
    f.write(f"Non-tooth: Precision={prec0:.3f}, Recall={rec0:.3f}, F1={f10:.3f}\n")
    f.write(f"Tooth: Precision={prec1:.3f}, Recall={rec1:.3f}, F1={f11:.3f}\n")

print("Results saved to binary_evaluation_results.txt")