import random
import shutil
from pathlib import Path

IMG_TRAIN = Path("datasets/train-yolo/merged/images/train")
LBL_TRAIN = Path("datasets/train-yolo/merged/labels/train")

IMG_VAL = Path("datasets/train-yolo/merged/images/val")
LBL_VAL = Path("datasets/train-yolo/merged/labels/val")

VAL_RATIO = 0.2
SEED = 42

IMG_VAL.mkdir(parents=True, exist_ok=True)
LBL_VAL.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

random.seed(SEED)

images = [p for p in IMG_TRAIN.iterdir() if p.suffix.lower() in IMG_EXTS]
random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)

moved = 0
for img_path in images[:val_count]:
    lbl_path = LBL_TRAIN / f"{img_path.stem}.txt"

    if not lbl_path.exists():
        print(f"[WARN] Missing label for {img_path.name}")
        continue

    shutil.move(img_path, IMG_VAL / img_path.name)
    shutil.move(lbl_path, LBL_VAL / lbl_path.name)
    moved += 1

print("\n--- SPLIT COMPLETE ---")
print(f"Total images: {len(images)}")
print(f"Validation images: {moved}")
print(f"Training images: {len(images) - moved}")
