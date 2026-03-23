"""
Data Preparation Script for Multiclass Tooth Classification

This script helps organize and prepare tooth images for the 32-class multiclass
tooth classifier. It can:
1. Organize extracted tooth patches into FDI class directories
2. Generate synthetic data augmentation
3. Create train/val splits
4. Validate dataset integrity
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from typing import Dict, List
from collections import defaultdict


class ToothDatasetPreparer:
    """Prepare and organize tooth dataset for multiclass training"""
    
    # FDI tooth classes (32 permanent teeth)
    FDI_CLASSES = list(range(11, 19)) + list(range(21, 29)) + \
                  list(range(31, 39)) + list(range(41, 49))
    
    def __init__(self, output_dir: str = "datasets/tooth_multiclass"):
        self.output_dir = Path(output_dir)
        self.class_counts = defaultdict(int)
        
    def create_directory_structure(self):
        """Create the directory structure for FDI classes"""
        print("[1] Creating directory structure...")
        
        # Create main directories
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            
            # Create FDI class subdirectories
            for fdi in self.FDI_CLASSES:
                (self.output_dir / "images" / split / str(fdi)).mkdir(parents=True, exist_ok=True)
        
        print(f"   ✓ Created directories for {len(self.FDI_CLASSES)} FDI classes")
    
    def organize_tooth_images(self, source_dir: str, fdi_mapping: Dict[str, int]):
        """
        Organize tooth images from source directory into FDI classes
        
        Args:
            source_dir: Directory containing tooth image patches
            fdi_mapping: Dict mapping original folder names to FDI numbers
                        e.g., {"tooth_0": 11, "tooth_1": 12, ...}
        """
        print(f"\n[2] Organizing tooth images from {source_dir}...")
        
        source_dir = Path(source_dir)
        if not source_dir.exists():
            print(f"   ✗ Source directory not found: {source_dir}")
            return False
        
        total_copied = 0
        
        for src_folder, fdi_number in fdi_mapping.items():
            src_path = source_dir / src_folder
            if not src_path.exists():
                print(f"   ⚠ Source folder not found: {src_path}")
                continue
            
            # Count images
            img_files = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")) + \
                       list(src_path.glob("*.JPG")) + list(src_path.glob("*.PNG"))
            
            if not img_files:
                print(f"   ⚠ No images found in {src_path}")
                continue
            
            # Split into train/val
            random.shuffle(img_files)
            train_count = int(len(img_files) * 0.8)
            train_files = img_files[:train_count]
            val_files = img_files[train_count:]
            
            # Copy train images
            train_dest = self.output_dir / "images" / "train" / str(fdi_number)
            for idx, img_file in enumerate(train_files):
                dest = train_dest / f"{fdi_number}_train_{idx:06d}{img_file.suffix}"
                shutil.copy2(img_file, dest)
            
            # Copy val images
            val_dest = self.output_dir / "images" / "val" / str(fdi_number)
            for idx, img_file in enumerate(val_files):
                dest = val_dest / f"{fdi_number}_val_{idx:06d}{img_file.suffix}"
                shutil.copy2(img_file, dest)
            
            self.class_counts[fdi_number] = len(img_files)
            total_copied += len(img_files)
            print(f"   ✓ FDI {fdi_number}: {len(train_files)} train, {len(val_files)} val")
        
        print(f"\n   Total images copied: {total_copied}")
        return True
    
    def augment_class_images(self, min_images_per_class: int = 50):
        """
        Augment underrepresented classes to have minimum samples
        
        Args:
            min_images_per_class: Minimum number of images needed per class
        """
        print(f"\n[3] Augmenting classes to minimum {min_images_per_class} images...")
        
        total_augmented = 0
        
        for split in ["train", "val"]:
            split_dir = self.output_dir / "images" / split
            
            for fdi_dir in split_dir.iterdir():
                if not fdi_dir.is_dir():
                    continue
                
                img_files = list(fdi_dir.glob("*.jpg")) + list(fdi_dir.glob("*.png"))
                current_count = len(img_files)
                
                if current_count == 0:
                    continue
                
                # If class has fewer images than minimum, augment
                if current_count < min_images_per_class:
                    needed = min_images_per_class - current_count
                    
                    for i in range(needed):
                        # Pick random image and apply light augmentation
                        src_img = Image.open(random.choice(img_files))
                        
                        # Apply random augmentations
                        aug_img = self._augment_image(src_img)
                        
                        # Save augmented image
                        aug_filename = f"{fdi_dir.name}_aug_{i:06d}.jpg"
                        aug_img.save(fdi_dir / aug_filename)
                        total_augmented += 1
                    
                    print(f"   ✓ FDI {fdi_dir.name} ({split}): Added {needed} augmented images")
        
        print(f"   Total augmented images created: {total_augmented}")
    
    def _augment_image(self, img: Image.Image) -> Image.Image:
        """Apply light augmentation to image"""
        # Random rotation
        if random.random() > 0.5:
            img = img.rotate(random.randint(-15, 15), expand=False, fillcolor=(255, 255, 255))
        
        # Random flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random brightness/contrast
        arr = np.array(img, dtype=np.float32)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        arr = (arr - 128) * contrast + 128 * brightness
        arr = np.clip(arr, 0, 255)
        
        return Image.fromarray(arr.astype(np.uint8))
    
    def generate_data_yaml(self):
        """Generate data.yaml for YOLO-style training"""
        print("\n[4] Generating data.yaml...")
        
        train_dir = str((self.output_dir / "images" / "train").resolve())
        val_dir = str((self.output_dir / "images" / "val").resolve())
        
        yaml_content = f"""path: {self.output_dir}
train: images/train
val: images/val

nc: {len(self.FDI_CLASSES)}
names:
"""
        
        for fdi in self.FDI_CLASSES:
            yaml_content += f"  {fdi - 10}: '{fdi}'  # FDI tooth {fdi}\n"
        
        yaml_path = self.output_dir / "data.yaml"
        yaml_path.write_text(yaml_content)
        print(f"   ✓ Created {yaml_path}")
    
    def validate_dataset(self):
        """Validate dataset structure and statistics"""
        print("\n[5] Validating dataset...")
        
        total_train = 0
        total_val = 0
        class_stats = defaultdict(dict)
        
        for split, split_name in [("train", "Train"), ("val", "Val")]:
            split_dir = self.output_dir / "images" / split
            
            for fdi_dir in sorted(split_dir.iterdir()):
                if not fdi_dir.is_dir():
                    continue
                
                fdi = fdi_dir.name
                img_count = len(list(fdi_dir.glob("*.jpg"))) + len(list(fdi_dir.glob("*.png")))
                
                if fdi not in class_stats:
                    class_stats[fdi] = {}
                class_stats[fdi][split] = img_count
                
                if split == "train":
                    total_train += img_count
                else:
                    total_val += img_count
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Total Train Images: {total_train}")
        print(f"  Total Val Images:   {total_val}")
        print(f"  Total Images:       {total_train + total_val}")
        print(f"  Classes with data:  {len(class_stats)} / {len(self.FDI_CLASSES)}")
        
        # Find classes with no data
        empty_classes = [fdi for fdi in self.FDI_CLASSES if fdi not in class_stats]
        if empty_classes:
            print(f"\n  ⚠ Empty classes: {empty_classes}")
        
        print("\nPer-class breakdown:")
        print(f"  {'FDI':<5} {'Train':<10} {'Val':<10} {'Total':<10}")
        print("  " + "-" * 35)
        for fdi in sorted(class_stats.keys(), key=lambda x: int(x)):
            train = class_stats[fdi].get("train", 0)
            val = class_stats[fdi].get("val", 0)
            total = train + val
            print(f"  {fdi:<5} {train:<10} {val:<10} {total:<10}")
        
        return class_stats
    
    def get_summary(self):
        """Print dataset summary"""
        print("\n" + "=" * 60)
        print("DATASET PREPARATION SUMMARY")
        print("=" * 60)
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"Total FDI classes: {len(self.FDI_CLASSES)}")
        print(f"Classes: {self.FDI_CLASSES}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare multiclass tooth dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/tooth_multiclass",
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source directory with tooth images (optional for organizing existing data)",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=50,
        help="Minimum images per class for augmentation",
    )
    parser.add_argument(
        "--create-structure-only",
        action="store_true",
        help="Only create directory structure without organizing data",
    )

    args = parser.parse_args()

    preparer = ToothDatasetPreparer(args.output_dir)
    
    print("\n" + "=" * 60)
    print("TOOTH DATASET PREPARATION TOOL")
    print("=" * 60 + "\n")
    
    # Create directory structure
    preparer.create_directory_structure()
    
    if args.create_structure_only:
        print("\n[*] Directory structure created. Ready to organize tooth images.")
        print("\nNext steps:")
        print("1. Manually organize tooth images into FDI class directories")
        print("2. Run this script with --source-dir to organize existing data")
        return
    
    if args.source_dir:
        # Example mapping - customize based on your data
        fdi_mapping = {
            "tooth": 11,  # Adjust based on your actual data
            "tooth_0": 11,
            "tooth_1": 12,
            # Add more mappings as needed
        }
        
        print("\n[*] Note: Customize fdi_mapping in this script based on your data structure")
        # preparer.organize_tooth_images(args.source_dir, fdi_mapping)
    
    # Generate data.yaml
    preparer.generate_data_yaml()
    
    # Validate dataset
    preparer.validate_dataset()
    
    # Print summary
    preparer.get_summary()
    
    print("\nDataset preparation complete!")
    print(f"Use 'python scripts/train_multiclass_teeth.py --data-dir {args.output_dir}' to train")


if __name__ == "__main__":
    main()
