from pathlib import Path
import shutil
import random

DATA_DIR = Path("../datasets/tooth_vs_nontooth/merged/binary_patches")
OUT_DIR = Path("../datasets/tooth_vs_nontooth/split")
(OUT_DIR / "train" / "tooth").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "train" / "non_tooth").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "val" / "tooth").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "val" / "non_tooth").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "test" / "tooth").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "test" / "non_tooth").mkdir(parents=True, exist_ok=True)

for cls in ["tooth", "non_tooth"]:
    cls_dir = DATA_DIR / cls
    files = list(cls_dir.glob("*.jpg"))
    random.shuffle(files)
    n = len(files)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)  # 15% val, 15% test
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    for f in train_files: shutil.copy(f, OUT_DIR / "train" / cls / f.name)
    for f in val_files: shutil.copy(f, OUT_DIR / "val" / cls / f.name)
    for f in test_files: shutil.copy(f, OUT_DIR / "test" / cls / f.name)

print("Dataset split complete.")