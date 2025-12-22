from pathlib import Path
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # -------------------------
    # Config (FAST CPU)
    # -------------------------
    DATA_DIR = Path("datasets/train/merged/binary_patches")  # tooth/ and non_tooth/
    BATCH_SIZE = 64          # was 16 -> faster on CPU
    EPOCHS = 2               # quick baseline; increase later if needed
    LR = 1e-3
    IMG_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # On Windows, num_workers>0 can be slower / unstable due to spawn.
    NUM_WORKERS = 0
    PIN_MEMORY = True if DEVICE == "cuda" else False

    # Make PyTorch use more CPU threads (often helps)
    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(2)

    print("=== CONFIG ===")
    print("Dataset dir:", DATA_DIR.resolve())
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("LR:", LR)
    print("IMG_SIZE:", IMG_SIZE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("================\n")

    # -------------------------
    # Normalization (pretrained ResNet expects this)
    # -------------------------
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # -------------------------
    # Transforms (FAST: no rotation)
    # -------------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # -------------------------
    # Load dataset
    # -------------------------
    print("[1] Loading dataset...")
    full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    print("Dataset loaded.")
    print("Total images:", len(full_ds))
    print("Classes:", full_ds.classes)

    # -------------------------
    # Split (avoid transform-sharing bug)
    # -------------------------
    val_ratio = 0.2
    val_len = int(len(full_ds) * val_ratio)
    train_len = len(full_ds) - val_len

    print("\n[2] Splitting dataset...")
    print("Train size:", train_len)
    print("Val size  :", val_len)

    train_ds, tmp_val_ds = random_split(
        full_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    # Separate val dataset object so it can use val_tf
    val_base = datasets.ImageFolder(DATA_DIR, transform=val_tf)
    val_ds = torch.utils.data.Subset(val_base, tmp_val_ds.indices)

    print("Split done.")

    # -------------------------
    # Class weights for imbalance
    # -------------------------
    train_targets = [full_ds.targets[i] for i in train_ds.indices]
    num_classes = len(full_ds.classes)

    counts = torch.zeros(num_classes, dtype=torch.long)
    for t in train_targets:
        counts[t] += 1

    weights = (counts.sum().float() / counts.float()).to(torch.float32)

    print("\n[3] Class counts in TRAIN split:")
    for i, cls in enumerate(full_ds.classes):
        print(f"  {cls}: {counts[i].item()}")
    print("[3] Class weights:", {full_ds.classes[i]: float(weights[i]) for i in range(num_classes)})

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    # -------------------------
    # Dataloaders
    # -------------------------
    print("\n[4] Creating dataloaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("Train batches:", len(train_loader))
    print("Val batches  :", len(val_loader))

    # Sanity check
    print("\n[5] Sanity check: loading one training batch...")
    x0, y0 = next(iter(train_loader))
    print("Batch loaded.")
    print("Images shape:", x0.shape)
    print("Labels shape:", y0.shape)

    # -------------------------
    # Model
    # -------------------------
    print("\n[6] Building model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    # FAST: freeze backbone, train only the final layer
    for name, p in model.named_parameters():
        if not name.startswith("fc"):
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    print("Model ready. Trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate():
        model.eval()
        total, correct = 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(dim=1)

                total += y.size(0)
                correct += (pred == y).sum().item()
                y_true.extend(y.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

        acc = correct / total if total else 0.0
        return acc, y_true, y_pred

    # -------------------------
    # Train
    # -------------------------
    print("\n[7] Starting training...\n")
    best_acc = -1.0

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"=== Epoch {epoch}/{EPOCHS} ===")
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [train] Batch {batch_idx+1}/{len(train_loader)}")

        val_acc, y_true, y_pred = evaluate()
        avg_loss = running_loss / max(1, len(train_loader))

        print(f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": full_ds.classes,
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            }, out_dir / "tooth_vs_nontooth_resnet18_best.pth")
            print(f"[INFO] Saved new best model (val_acc={best_acc:.4f})")

    # -------------------------
    # Report
    # -------------------------
    print("\n[8] Validation report:")
    print(classification_report(y_true, y_pred, target_names=full_ds.classes))

    cm = confusion_matrix(y_true, y_pred)
    print("[8] Confusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nTraining complete. Best val_acc:", best_acc)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
