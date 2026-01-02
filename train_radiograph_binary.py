from pathlib import Path
import os, time, random

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    seed_everything(42)

    DATA_DIR = Path("datasets/radiograph_vs_not_radiograph")
    OUT_DIR = Path("models")
    OUT_DIR.mkdir(exist_ok=True)

    IMG_SIZE = 224
    VAL_RATIO = 0.2
    BATCH_SIZE = 64
    NUM_WORKERS = 0  # windows safer

    WARMUP_l = 1
    FINETUNE_EPOCHS = 6
    LR_WARMUP = 1e-3
    LR_FINETUNE = 1e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False

    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(2)

    print("\n=== RADIOGRAPH GATE TRAINING ===")
    print("DATA_DIR:", DATA_DIR.resolve())
    print("DEVICE:", DEVICE)

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR.resolve())

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # Augmentations that help radiographs (contrast/etc.)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.3)], p=0.6),
        transforms.RandomAutocontrast(p=0.35),
        transforms.RandomEqualize(p=0.20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load dataset
    full_train_base = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    classes = full_train_base.classes
    num_classes = len(classes)

    print("Total images:", len(full_train_base))
    print("Classes:", classes)
    print("Class -> index:", {c: i for i, c in enumerate(classes)})

    if set(classes) != {"radiograph", "not_radiograph"}:
        print("\n[WARN] Recommended class folder names:")
        print("  radiograph_patch/")
        print("  not_radiograph/\n")

    # Split
    val_len = int(len(full_train_base) * VAL_RATIO)
    train_len = len(full_train_base) - val_len

    train_subset, tmp_val_subset = random_split(
        full_train_base,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    full_val_base = datasets.ImageFolder(DATA_DIR, transform=val_tf)
    val_subset = Subset(full_val_base, tmp_val_subset.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Split -> Train: {len(train_subset)} | Val: {len(val_subset)}")
    print("Train batches:", len(train_loader), "| Val batches:", len(val_loader))

    # Class weights (optional; helps imbalance)
    train_targets = [full_train_base.targets[i] for i in train_subset.indices]
    counts = torch.zeros(num_classes, dtype=torch.long)
    for t in train_targets:
        counts[t] += 1
    weights = (counts.sum().float() / counts.float().clamp(min=1)).to(torch.float32)
    weights = weights / weights.mean()
    print("Train counts:", {classes[i]: int(counts[i]) for i in range(num_classes)})
    print("Class weights:", {classes[i]: float(weights[i]) for i in range(num_classes)})

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    def set_trainable(stage: str):
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
            raise ValueError(stage)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Stage={stage} | trainable params: {trainable:,}/{total:,}")

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

    def train_stage(num_epochs: int, lr: float, stage: str, best_acc: float):
        set_trainable(stage)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        for epoch in range(1, num_epochs + 1):
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
                    print(f"  [train] {stage} epoch {epoch}/{num_epochs} | batch {batch_idx}/{len(train_loader)} "
                          f"| loss {loss.item():.4f} | {ips:.1f} img/s")

            train_loss = running_loss / max(1, len(train_loader))
            val_loss, val_acc, y_true, y_pred = evaluate()

            print(f"\n[{stage}] DONE: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion matrix:\n", cm)

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = OUT_DIR / "radiograph_binary_best.pth"
                torch.save({
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "img_size": IMG_SIZE,
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                }, best_path)
                print("[SAVE] best ->", best_path, "val_acc=", best_acc)

            print("")
        return best_acc, y_true, y_pred

    best_acc = -1.0
    print("\n[STAGE 1] WARMUP (train fc)...")
    best_acc, y_true, y_pred = train_stage(WARMUP_EPOCHS, LR_WARMUP, "warmup", best_acc)

    print("\n[STAGE 2] FINETUNE (layer4+fc)...")
    best_acc, y_true, y_pred = train_stage(FINETUNE_EPOCHS, LR_FINETUNE, "finetune", best_acc)

    print("\n=== FINAL REPORT ===")
    print(classification_report(y_true, y_pred, target_names=classes))

    final_path = OUT_DIR / "radiograph_binary_final.pth"
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }, final_path)

    print("\nTraining complete.")
    print("Best val_acc:", best_acc)
    print("Saved:", final_path)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
