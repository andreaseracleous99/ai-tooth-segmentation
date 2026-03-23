"""
Multiclass Tooth Classifier Training Script

Trains a ResNet18 model to classify 32 different tooth types (FDI numbering system).
The model can then segment and identify individual teeth in dental photographs.
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from pathlib import Path
import argparse
from collections import Counter
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tooth_system import get_all_fdi_numbers, get_tooth_info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_classifier_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}. Use one of: resnet18, resnet50, efficientnet_v2_s")


def get_targets_from_dataset(ds):
    if isinstance(ds, Subset):
        base_targets = ds.dataset.targets
        return [base_targets[i] for i in ds.indices]
    return list(ds.targets)


def main():
    # ============================
    # CONFIG
    # ============================
    parser = argparse.ArgumentParser(description="Train multiclass tooth classifier (32 classes)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/tooth_multiclass_fdi",
        help="Root dataset directory with images/ and labels/ subdirs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=300,
        help="Image size for model input",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Total epochs to train",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Warmup epochs before cosine schedule",
    )
    parser.add_argument(
        "--lr-warmup",
        type=float,
        default=1e-4,
        help="Learning rate for warmup phase",
    )
    parser.add_argument(
        "--lr-finetune",
        type=float,
        default=3e-4,
        help="Learning rate for finetuning phase",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet_v2_s",
        choices=["resnet18", "resnet50", "efficientnet_v2_s"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Cross entropy label smoothing",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine schedule",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop if validation accuracy does not improve for N epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        action="store_true",
        help="Disable class-balanced weighted sampling",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss",
    )

    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(exist_ok=True)

    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    WARMUP_EPOCHS = args.warmup_epochs
    LR_WARMUP = args.lr_warmup
    LR_FINETUNE = args.lr_finetune
    VAL_RATIO = args.val_ratio
    NUM_WORKERS = args.num_workers
    ARCH = args.arch
    WEIGHT_DECAY = args.weight_decay
    LABEL_SMOOTHING = args.label_smoothing
    MIN_LR = args.min_lr
    EARLY_STOP_PATIENCE = args.early_stop_patience
    USE_WEIGHTED_SAMPLER = not args.no_weighted_sampler
    USE_CLASS_WEIGHTS = not args.no_class_weights

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False

    set_seed(args.seed)

    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(2)

    print("\n" + "=" * 60)
    print("MULTICLASS TOOTH CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Data directory    : {DATA_DIR.resolve()}")
    print(f"Output directory  : {OUT_DIR.resolve()}")
    print(f"Device            : {DEVICE}")
    print(f"Image size        : {IMG_SIZE}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Total epochs      : {NUM_EPOCHS}")
    print(f"Warmup epochs     : {WARMUP_EPOCHS} @ lr={LR_WARMUP}")
    print(f"Finetune epochs   : {NUM_EPOCHS - WARMUP_EPOCHS} @ lr={LR_FINETUNE}")
    print(f"Validation ratio  : {VAL_RATIO}")
    print(f"Architecture      : {ARCH}")
    print(f"Label smoothing   : {LABEL_SMOOTHING}")
    print(f"Weight decay      : {WEIGHT_DECAY}")
    print(f"Weighted sampler  : {USE_WEIGHTED_SAMPLER}")
    print(f"Class weights     : {USE_CLASS_WEIGHTS}")
    print("=" * 60 + "\n")

    # Verify dataset exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR.resolve()}")

    # Setup normalization (ImageNet pretrained)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.25)],
            p=0.6,
        ),
        transforms.RandomAutocontrast(p=0.35),
        transforms.RandomEqualize(p=0.20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Resolve dataset layout and load data.
    print("[1] Loading dataset...")
    train_dir = None
    val_dir = None

    if (DATA_DIR / "images" / "train").exists() and (DATA_DIR / "images" / "val").exists():
        train_dir = DATA_DIR / "images" / "train"
        val_dir = DATA_DIR / "images" / "val"
    elif (DATA_DIR / "train").exists() and (DATA_DIR / "val").exists():
        train_dir = DATA_DIR / "train"
        val_dir = DATA_DIR / "val"

    if train_dir and val_dir:
        train_base = datasets.ImageFolder(train_dir, transform=train_tf)
        val_base = datasets.ImageFolder(val_dir, transform=val_tf)

        classes = train_base.classes
        num_classes = len(classes)

        if train_base.classes != val_base.classes:
            raise ValueError("Train/val class folders do not match. Please ensure both splits use identical FDI class names.")

        train_subset = train_base
        val_subset = val_base

        print(f"Train directory: {train_dir}")
        print(f"Val directory  : {val_dir}")
    else:
        full_train_base = datasets.ImageFolder(DATA_DIR, transform=train_tf)
        classes = full_train_base.classes
        num_classes = len(classes)

        val_len = int(len(full_train_base) * VAL_RATIO)
        train_len = len(full_train_base) - val_len

        train_subset, tmp_val_subset = random_split(
            full_train_base,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

        full_val_base = datasets.ImageFolder(DATA_DIR, transform=val_tf)
        val_subset = Subset(full_val_base, tmp_val_subset.indices)
        print(f"Dataset directory: {DATA_DIR} (random split with val_ratio={VAL_RATIO})")

    print(f"Total train images: {len(train_subset)}")
    print(f"Total val images  : {len(val_subset)}")
    print(f"Number of classes : {num_classes}")
    print(f"Classes: {classes[:5]}... (showing first 5)")

    # Expected 32 tooth classes
    if num_classes != 32:
        print(f"\n[WARN] Expected 32 tooth classes, but found {num_classes}.")
        print("Make sure your dataset has subdirectories named after FDI tooth numbers (11-18, 21-28, 31-38, 41-48)")

    train_targets = get_targets_from_dataset(train_subset)
    class_counts = Counter(train_targets)

    sampler = None
    class_weights_tensor = None
    if USE_WEIGHTED_SAMPLER:
        class_sample_weights = {c: 1.0 / max(count, 1) for c, count in class_counts.items()}
        sample_weights = [class_sample_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

    if USE_CLASS_WEIGHTS:
        class_weights = []
        total_count = len(train_targets)
        for class_idx in range(num_classes):
            c = class_counts.get(class_idx, 0)
            class_weights.append(total_count / max(c, 1))
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Build model
    print("[2] Building model...")
    model = build_classifier_model(ARCH, num_classes, pretrained=True)
    model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS),
        eta_min=MIN_LR,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    
    # Build checkpoint info
    start_epoch = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[*] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[*] Resuming from epoch {start_epoch}")

    # Training loop
    print("[3] Starting training...\n")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Determine learning rate phase
        is_warmup = epoch < WARMUP_EPOCHS
        lr = LR_WARMUP if is_warmup else LR_FINETUNE
        phase_name = "warmup" if is_warmup else "finetune"

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train epoch
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{phase_name}] Train", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            if hasattr(train_pbar, "set_postfix"):
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{phase_name}] Val  ", leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                if hasattr(val_pbar, "set_postfix"):
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS} [{phase_name:7s}] | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = OUT_DIR / "tooth_multiclass.pth"
            
            ckpt = {
                "model_state": model.state_dict(),
                "classes": classes,
                "class_to_idx": {cls_name: idx for idx, cls_name in enumerate(classes)},
                "arch": ARCH,
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "num_classes": num_classes,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            }
            torch.save(ckpt, checkpoint_path)
            print(f"  -> Saved best model to {checkpoint_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        latest_path = OUT_DIR / "tooth_multiclass_last.pth"
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": classes,
                "class_to_idx": {cls_name: idx for idx, cls_name in enumerate(classes)},
                "arch": ARCH,
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "num_classes": num_classes,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            },
            latest_path,
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping: no val accuracy improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print("\n" + "=" * 60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {OUT_DIR / 'tooth_multiclass.pth'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
