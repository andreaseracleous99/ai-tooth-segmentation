import shutil
from pathlib import Path

DATASETS_ROOT = Path("../../datasets/train-yolo")  

OUT_IMG = Path("../../datasets/train-yolo/merged/images/train")
OUT_LBL = Path("../../datasets/train-yolo/merged/labels/train")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

img_count = 0
skipped = 0

for dataset_dir in sorted(DATASETS_ROOT.iterdir()):
    if not dataset_dir.is_dir():
        continue

    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[SKIP] {dataset_dir.name} (missing images/labels)")
        continue

    print(f"[DATASET] {dataset_dir.name}")

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            skipped += 1
            continue

        new_name = f"{dataset_dir.name}_{img_path.name}"
        new_lbl  = f"{dataset_dir.name}_{img_path.stem}.txt"

        shutil.copy2(img_path, OUT_IMG / new_name)
        shutil.copy2(lbl_path, OUT_LBL / new_lbl)

        img_count += 1

print("\n--- DONE ---")
print(f"Copied image-label pairs: {img_count}")
print(f"Skipped images without labels: {skipped}")
