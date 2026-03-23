from pathlib import Path
import shutil

SRC_ROOTS = [
    Path("Datasets/3/binary_patches"),
    Path("Datasets/11/binary_patches"),
    Path("Datasets/12/binary_patches"),
    Path("Datasets/20/binary_patches"),
    Path("Datasets/27/binary_patches"),
    Path("Datasets/30/binary_patches"),
    Path("Datasets/42/binary_patches"),
]

OUT_DIR = Path("Datasets/ALL/binary_patches")
CLASSES = ["tooth", "non_tooth"]

for cls in CLASSES:
    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

counters = {cls: 0 for cls in CLASSES}

for src in SRC_ROOTS:
    print(f"[INFO] Processing {src}")
    for cls in CLASSES:
        src_cls = src / cls
        if not src_cls.exists():
            print(f"[WARN] Missing {src_cls}")
            continue

        for img_path in src_cls.iterdir():
            if not img_path.is_file():
                continue

            new_name = f"{cls}_{counters[cls]:08d}{img_path.suffix}"
            dst = OUT_DIR / cls / new_name

            shutil.copy2(img_path, dst)
            counters[cls] += 1

print("\n[DONE]")
print("Merged counts:")
for cls in CLASSES:
    print(f"{cls}: {counters[cls]}")
print("Saved to:", OUT_DIR.resolve())
