"""
Multiclass Tooth Classifier Evaluation

Evaluate trained multiclass tooth classifier on test dataset.
Computes per-class and overall accuracy, confusion matrix, and detailed metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tooth_system import get_tooth_info


def build_classifier_model(arch: str, num_classes: int):
    arch = (arch or "resnet18").lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported checkpoint architecture: {arch}")


def evaluate_model(model_path: str, 
                  test_dir: str,
                  batch_size: int = 32,
                  device: str = "cuda"):
    """
    Evaluate multiclass model
    
    Args:
        model_path: Path to trained model checkpoint
        test_dir: Directory with test images organized by tooth class
        batch_size: Evaluation batch size
        device: "cuda" or "cpu"
    """
    
    device = device if torch.cuda.is_available() else "cpu"
    model_path = Path(model_path)
    test_dir = Path(test_dir)
    
    print("\n" + "=" * 70)
    print("MULTICLASS TOOTH CLASSIFIER EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Test data: {test_dir}")
    print(f"Device: {device}\n")
    
    # Load checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    classes = ckpt.get("classes", [])
    class_to_idx = ckpt.get("class_to_idx", None)
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("mean", (0.485, 0.456, 0.406))
    std = ckpt.get("std", (0.229, 0.224, 0.225))
    num_classes = ckpt.get("num_classes", len(classes))
    arch = ckpt.get("arch", "resnet18")
    
    # Build model
    model = build_classifier_model(arch, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    
    print(f"Model Classes: {num_classes}")
    print(f"Architecture: {arch}")
    print(f"Input Size: {img_size}×{img_size}")
    print(f"Normalization: mean={mean}, std={std}\n")
    
    # Transforms
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load dataset
    print("[1] Loading test dataset...")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   Test images: {len(test_dataset)}")
    print(f"   Test batches: {len(test_loader)}")

    # Resolve class labels with robust fallback when checkpoint metadata is stale.
    if isinstance(class_to_idx, dict) and len(class_to_idx) == num_classes:
        class_labels = [""] * num_classes
        for class_name, idx in class_to_idx.items():
            if isinstance(idx, int) and 0 <= idx < num_classes:
                class_labels[idx] = str(class_name)
    elif isinstance(classes, list) and len(classes) == num_classes:
        class_labels = [str(c) for c in classes]
    elif len(test_dataset.classes) == num_classes:
        class_labels = [str(c) for c in test_dataset.classes]
        print("   Warning: checkpoint class metadata mismatch; using test dataset classes.")
    else:
        class_labels = [str(i) for i in range(num_classes)]
        print("   Warning: class labels unavailable; using class indices.")
    
    # Check class distribution
    class_counts = Counter(test_dataset.targets)
    print(f"   Unique classes in test set: {len(class_counts)}")

    if num_classes != len(test_dataset.classes):
        raise ValueError(
            f"Model outputs {num_classes} classes ({class_labels}), but test dataset has "
            f"{len(test_dataset.classes)} classes ({test_dataset.classes}). "
            "This model was likely trained on the wrong dataset root. Retrain using "
            "--data-dir datasets/tooth_multiclass_fdi so the trainer reads images/train and images/val."
        )
    
    # Evaluate
    print("\n[2] Evaluating model...\n")
    
    all_preds = []
    all_targets = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Per-class accuracy
            for pred, target in zip(predicted.cpu().numpy(), targets.cpu().numpy()):
                class_total[target] += 1
                if pred == target:
                    class_correct[target] += 1
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    overall_acc = (all_preds == all_targets).mean() * 100
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")
    print(f"Total Predictions: {len(all_preds)}")
    print(f"Correct: {(all_preds == all_targets).sum()}")
    print(f"Incorrect: {(all_preds != all_targets).sum()}")
    
    # Per-class metrics
    print("\n" + "-" * 70)
    print("Per-Class Accuracy:")
    print("-" * 70)
    print(f"{'FDI':<6} {'Name':<25} {'Accuracy':<12} {'Count':<8} {'Correct':<8}")
    print("-" * 70)
    
    tooth_accuracies = []

    for class_idx in sorted(class_total.keys()):
        label = class_labels[class_idx] if class_idx < len(class_labels) else str(class_idx)
        try:
            fdi = int(label)
        except (TypeError, ValueError):
            fdi = None
        
        total = class_total.get(class_idx, 0)
        correct = class_correct.get(class_idx, 0)
        
        if total > 0:
            acc = (correct / total) * 100
            tooth_display = str(fdi) if fdi is not None else label
            tooth_accuracies.append((tooth_display, acc))
            
            try:
                tooth_info = get_tooth_info(fdi) if fdi is not None else None
                name = tooth_info["name"]
            except Exception:
                name = "Unknown"
            
            print(f"{tooth_display:<6} {name:<25} {acc:>6.2f}%{'':<5} {total:<8} {correct:<8}")
    
    # Best and worst performing teeth
    if tooth_accuracies:
        tooth_accuracies.sort(key=lambda x: x[1])
        
        print("\n" + "-" * 70)
        print("Worst Performing Teeth (Bottom 5):")
        print("-" * 70)
        for fdi, acc in tooth_accuracies[:5]:
            print(f"  Tooth {fdi}: {acc:.2f}%")
        
        print("\nBest Performing Teeth (Top 5):")
        print("-" * 70)
        for fdi, acc in tooth_accuracies[-5:]:
            print(f"  Tooth {fdi}: {acc:.2f}%")
    
    # Confusion matrix elements
    print("\n" + "-" * 70)
    print("Confusion Statistics:")
    print("-" * 70)
    
    # Find most common misclassifications
    misclass = []
    for pred, target in zip(all_preds, all_targets):
        if pred != target:
            misclass.append((int(target), int(pred)))
    
    if misclass:
        misclass_counts = Counter(misclass)
        print("\nMost Common Misclassifications (predicted as):")
        for (true_class, pred_class), count in misclass_counts.most_common(10):
            true_label = class_labels[true_class] if true_class < len(class_labels) else str(true_class)
            pred_label = class_labels[pred_class] if pred_class < len(class_labels) else str(pred_class)
            try:
                true_fdi = int(true_label)
                pred_fdi = int(pred_label)
                true_name = get_tooth_info(true_fdi)["name"]
                pred_name = get_tooth_info(pred_fdi)["name"]
            except Exception:
                true_fdi = true_label
                pred_fdi = pred_label
                true_name = "Unknown"
                pred_name = "Unknown"
            
            print(f"  FDI {true_fdi} ({true_name}) → "
                  f"FDI {pred_fdi} ({pred_name}): {count} times")
    
    # Export results
    results = {
        "overall_accuracy": float(overall_acc),
        "total_samples": int(len(all_preds)),
        "correct": int((all_preds == all_targets).sum()),
        "incorrect": int((all_preds != all_targets).sum()),
        "per_class_accuracy": {
            (int(class_labels[class_idx]) if class_idx < len(class_labels) and class_labels[class_idx].isdigit()
             else (class_labels[class_idx] if class_idx < len(class_labels) else str(class_idx))): {
                "accuracy": float((class_correct[class_idx] / class_total[class_idx] * 100) 
                                if class_total[class_idx] > 0 else 0),
                "correct": int(class_correct[class_idx]),
                "total": int(class_total[class_idx])
            }
            for class_idx in sorted(class_total.keys())
        }
    }
    
    print("\n" + "=" * 70)
    print(f"Evaluation complete!")
    print("=" * 70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiclass tooth classifier")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/tooth_multiclass.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="datasets/tooth_multiclass/images/val",
        help="Directory with test images (organized by class subdirectories)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        batch_size=args.batch_size
    )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
