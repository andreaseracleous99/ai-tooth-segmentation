from pathlib import Path
import os
import time
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True       


def main():
    seed_everything(42)

    DATA_DIR = Path("datasets/tooth_vs_nontooth/merged/binary_patches") 
    OUT_DIR = Path("models")
    OUT_DIR.mkdir(exist_ok=True)

    IMG_SIZE = 224
    VAL_RATIO = 0.2

    BATCH_SIZE = 64
    NUM_WORKERS = 0  

    WARMUP_EPOCHS = 1         
    FINETUNE_EPOCHS = 6        
    LR_WARMUP = 1e-3
    LR_FINETUNE = 1e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False

    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(2)

    print("\n=== CONFIG ===")
    print("DATA_DIR     :", DATA_DIR.resolve())
    print("DEVICE       :", DEVICE)
    print("IMG_SIZE     :", IMG_SIZE)
    print("BATCH_SIZE   :", BATCH_SIZE)
    print("VAL_RATIO    :", VAL_RATIO)
    print("WARMUP_EPOCHS:", WARMUP_EPOCHS, "LR:", LR_WARMUP)
    print("FINETUNE_EPOCHS:", FINETUNE_EPOCHS, "LR:", LR_FINETUNE)
    print("NUM_WORKERS  :", NUM_WORKERS)
    print("================\n")

    # Normalization (ResNet pretrained expects this)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.25)],
            p=0.6
        ),
        transforms.RandomAutocontrast(p=0.35),
        transforms.RandomEqualize(p=0.20),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load dataset (base)
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR.resolve()}")

    print("[1] Loading dataset...")
    full_train_base = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    classes = full_train_base.classes
    num_classes = len(classes)

    if num_classes != 2:
        print("[WARN] Expected 2 classes (tooth/non_tooth). Found:", classes)

    print("Total images:", len(full_train_base))
    print("Classes:", classes)

    # Split indices
    val_len = int(len(full_train_base) * VAL_RATIO)
    train_len = len(full_train_base) - val_len

    train_subset, tmp_val_subset = random_split(
        full_train_base,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    full_val_base = datasets.ImageFolder(DATA_DIR, transform=val_tf)
    val_subset = Subset(full_val_base, tmp_val_subset.indices)

    print(f"[2] Split -> Train: {len(train_subset)} | Val: {len(val_subset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("[3] Train batches:", len(train_loader), "| Val batches:", len(val_loader))

    train_targets = [full_train_base.targets[i] for i in train_subset.indices]
    counts = torch.zeros(num_classes, dtype=torch.long)
    for t in train_targets:
        counts[t] += 1

    weights = (counts.sum().float() / counts.float().clamp(min=1)).to(torch.float32)
    weights = weights / weights.mean()

    print("\n[4] Class counts in TRAIN:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {counts[i].item()}")
    print("[4] Class weights:", {classes[i]: float(weights[i]) for i in range(num_classes)})

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    # Build model
    print("\n[5] Building model (ResNet18 pretrained)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    def set_trainable(stage: str):
        """
        stage='warmup' -> train only fc
        stage='finetune' -> train layer4 + fc
        """
        for p in model.parameters():
            p.requires_grad = False

        if stage == "warmup":
            for p in model.fc.parameters():
                p.requires_grad = True
        elif stage == "finetune":
            for name, p in model.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    p.requires_grad = True
        else:
            raise ValueError("Unknown stage: " + stage)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Stage={stage} | trainable params: {trainable:,} / {total:,}")

    # Evaluation
    @torch.no_grad()
    def evaluate():
        model.eval()
        total, correct = 0, 0
        y_true, y_pred = [], []
        running_loss = 0.0

        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            running_loss += loss.item()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

        acc = correct / total if total else 0.0
        avg_loss = running_loss / max(1, len(val_loader))
        return avg_loss, acc, y_true, y_pred

    # Train loop
    def train_epochs(num_epochs: int, lr: float, stage: str, best_acc: float):
        set_trainable(stage)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )

        for epoch in range(1, num_epochs + 1):
            print(f"=== Epoch {epoch}/{num_epochs} ===")
            global_epoch = train_epochs.epoch_counter + 1
            train_epochs.epoch_counter += 1

            model.train()
            t0 = time.time()
            running_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx == 1 or batch_idx % 200 == 0 or batch_idx == len(train_loader):
                    elapsed = time.time() - t0
                    ips = (batch_idx * BATCH_SIZE) / max(elapsed, 1e-6)
                    print(f"  [train] epoch {global_epoch} | batch {batch_idx}/{len(train_loader)}"
                          f" | loss {loss.item():.4f} | {ips:.1f} img/s")

            train_avg_loss = running_loss / max(1, len(train_loader))
            val_loss, val_acc, y_true, y_pred = evaluate()

            print(f"\n=== EPOCH {global_epoch} DONE (stage={stage}) ===")
            print(f"train_loss={train_avg_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

            cm = confusion_matrix(y_true, y_pred)
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm)

            per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
            for i, cls in enumerate(classes):
                print(f"  {cls} acc: {per_class_acc[i]:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = OUT_DIR / "tooth_vs_nontooth_binary.pth"
                torch.save({
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "img_size": IMG_SIZE,
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                }, best_path)
                print(f"[SAVE] New best model -> {best_path} (val_acc={best_acc:.4f})")

            print("") 

        return best_acc

    train_epochs.epoch_counter = 0

    best_acc = -1.0
    print("\n[6] WARMUP (train only fc)...\n")
    best_acc = train_epochs(WARMUP_EPOCHS, LR_WARMUP, "warmup", best_acc)

    print("\n[7] FINETUNE (unfreeze layer4 + fc)...\n")
    best_acc = train_epochs(FINETUNE_EPOCHS, LR_FINETUNE, "finetune", best_acc)

    final_path = OUT_DIR / "tooth_vs_nontooth_binary.pth"
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }, final_path)

    print("\n[8] Training complete.")
    print("Best val_acc:", best_acc)
    print("Saved final model ->", final_path)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
