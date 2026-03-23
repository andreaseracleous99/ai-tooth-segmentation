# filter_yolo_dataset.py
# Creates a NEW filtered YOLO dataset folder:
# - copies only images that have a non-empty label file
# - rewrites every class id in labels to 0
# - removes (skips) empty label files (and their images)
# - skips images with no corresponding label .txt
#
# Expected structure under INPUT_DIR:
#   images/train, images/val
#   labels/train, labels/val
#
# Output structure is the same under OUTPUT_DIR.

from pathlib import Path
import shutil

# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = Path(r"datasets/train-yolo/merged")          # <-- change if needed
OUTPUT_DIR = Path(r"datasets/train-yolo/merged_filtered")  # <-- new folder

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------
# HELPERS
# ----------------------------
def rewrite_label_to_class0(text: str) -> str:
    """
    YOLO label line format: <class> <x> <y> <w> <h> (detection)
    or: <class> <x1> <y1> <x2> <y2> ... (segmentation polygon)
    We replace the first token (class id) with 0 for every non-empty valid line.
    We also drop invalid lines (e.g. empty, non-numeric coords).
    """
    out_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:   # too short to be YOLO det or seg
            continue

        # Validate numeric coords (skip bad lines)
        try:
            _ = float(parts[1])
        except Exception:
            continue

        parts[0] = "0"
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines).strip() + ("\n" if out_lines else "")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def process_split(split: str):
    img_in = INPUT_DIR / "images" / split
    lbl_in = INPUT_DIR / "labels" / split

    img_out = OUTPUT_DIR / "images" / split
    lbl_out = OUTPUT_DIR / "labels" / split

    ensure_dir(img_out)
    ensure_dir(lbl_out)

    if not img_in.exists() or not lbl_in.exists():
        print(f"[WARN] Missing split folders for '{split}': {img_in} or {lbl_in}")
        return 0, 0, 0

    # Map labels by stem for fast lookup
    labels_map = {p.stem: p for p in lbl_in.glob("*.txt")}

    kept = 0
    skipped_no_label = 0
    skipped_empty = 0

    for img_path in img_in.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        stem = img_path.stem
        lbl_path = labels_map.get(stem)

        # Skip images without labels
        if lbl_path is None or not lbl_path.exists():
            skipped_no_label += 1
            continue

        raw = lbl_path.read_text(encoding="utf-8", errors="ignore")
        rewritten = rewrite_label_to_class0(raw)

        # Skip empty/invalid labels (and skip the image too)
        if rewritten.strip() == "":
            skipped_empty += 1
            continue

        # Copy image
        shutil.copy2(img_path, img_out / img_path.name)

        # Write filtered label
        (lbl_out / lbl_path.name).write_text(rewritten, encoding="utf-8")

        kept += 1

    print(f"[{split}] kept={kept} | skipped_no_label={skipped_no_label} | skipped_empty_or_invalid={skipped_empty}")
    return kept, skipped_no_label, skipped_empty

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR.resolve()}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"OUTPUT_DIR already exists: {OUTPUT_DIR.resolve()}\n"
            f"Pick a new name or delete it first."
        )

    print("INPUT :", INPUT_DIR.resolve())
    print("OUTPUT:", OUTPUT_DIR.resolve())
    print("Creating filtered dataset...")

    # Copy data.yaml if exists (optional but useful)
    data_yaml = INPUT_DIR / "data.yaml"
    if data_yaml.exists():
        ensure_dir(OUTPUT_DIR)
        shutil.copy2(data_yaml, OUTPUT_DIR / "data.yaml")

    total_kept = total_no_label = total_empty = 0

    for split in ["train", "val"]:
        kept, no_label, empty = process_split(split)
        total_kept += kept
        total_no_label += no_label
        total_empty += empty

    print("\n=== DONE ===")
    print(f"Total kept images+labels: {total_kept}")
    print(f"Skipped images (no label file): {total_no_label}")
    print(f"Skipped images (empty/invalid label after filtering): {total_empty}")
    print(f"Filtered dataset saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
