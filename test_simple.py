# test_simple.py - Minimal test
import sys
sys.path.append('D:/HEMSAM_Project')

import cv2
import torch
from pathlib import Path

# Test 1: Check if dataset path exists
dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
print(f"Dataset path exists: {dataset_path.exists()}")
print(f"Dataset path: {dataset_path.absolute()}")

if dataset_path.exists():
    # List some class folders
    class_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(class_folders)} class folders")
    for folder in class_folders[:5]:
        print(f"  - {folder.name}")

# Test 2: Load a single image
if dataset_path.exists() and class_folders:
    test_folder = class_folders[0]
    test_images = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.JPG"))
    if test_images:
        img = cv2.imread(str(test_images[0]))
        print(f"Image loaded: shape={img.shape if img is not None else 'None'}")