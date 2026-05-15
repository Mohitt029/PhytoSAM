#!/usr/bin/env python3
"""
Improved PlantVillage Dataset Downloader
Works with Kaggle API token
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm
import shutil
import requests

# Your Kaggle API token
KAGGLE_TOKEN = "KGAT_79cedf62921f2e15420f2f64c38eb333"

def setup_kaggle_credentials():
    """Setup Kaggle credentials from token"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Method 1: Create kaggle.json
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Extract key from token (token format: KGAT_<key>)
    api_key = KAGGLE_TOKEN.replace("KGAT_", "")
    
    kaggle_data = {
        "username": "kaggle",  # You might need your actual username
        "key": api_key
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(kaggle_data, f)
    
    # Set permissions (Windows)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        print("Note: Permission setting skipped on Windows")
    
    print(f"✅ Kaggle credentials saved to {kaggle_json}")
    
    # Method 2: Also set environment variable
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    
    return True

def download_from_kaggle_cli(dataset_path: str, download_dir: Path):
    """Download using kaggle CLI"""
    cmd = ["kaggle", "datasets", "download", dataset_path, "-p", str(download_dir)]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Success: {result.stdout}")
    return True

def download_from_alternative_url(download_dir: Path):
    """
    Alternative download from HuggingFace (more reliable)
    """
    print("\nTrying alternative download from HuggingFace...")
    
    # HuggingFace dataset
    url = "https://huggingface.co/datasets/plantvillage/plantvillage/resolve/main/plantvillage.zip"
    
    zip_path = download_dir / "plantvillage.zip"
    
    # Download with progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    return zip_path

def extract_dataset(zip_path: Path, extract_to: Path):
    """Extract zip file"""
    print(f"\nExtracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        with tqdm(total=len(files), desc="Extracting") as pbar:
            for file in files:
                zip_ref.extract(file, extract_to)
                pbar.update(1)
    
    print(f"Extracted to: {extract_to}")
    return extract_to

def organize_dataset(raw_dir: Path):
    """
    Organize extracted dataset into class folders
    """
    print("\nOrganizing dataset...")
    
    # Look for class folders
    possible_locations = [
        raw_dir / "plantvillage" / "PlantVillage",
        raw_dir / "PlantVillage",
        raw_dir / "Color",
        raw_dir,
    ]
    
    source_dir = None
    for loc in possible_locations:
        if loc.exists() and loc.is_dir():
            # Check if it has subdirectories (class folders)
            subdirs = [d for d in loc.iterdir() if d.is_dir()]
            if subdirs:
                source_dir = loc
                break
    
    if source_dir is None:
        print("Could not auto-organize. Manual organization may be needed.")
        return False
    
    print(f"Found organized dataset at: {source_dir}")
    
    # Count classes
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    total_images = 0
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))
        total_images += len(images)
        print(f"  {class_dir.name}: {len(images)} images")
    
    print(f"\nTotal: {len(class_dirs)} classes, {total_images} images")
    return True

def create_sample_dataset(source_dir: Path, sample_dir: Path, samples_per_class: int = 20):
    """Create a small sample dataset for quick testing"""
    print(f"\nCreating sample dataset ({samples_per_class} images per class)...")
    
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Find class directories
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))
        
        if not images:
            continue
        
        # Create class subdirectory
        target_class_dir = sample_dir / class_dir.name
        target_class_dir.mkdir(exist_ok=True)
        
        # Copy samples
        for img_path in images[:samples_per_class]:
            shutil.copy(img_path, target_class_dir / img_path.name)
        
        print(f"  {class_dir.name}: {min(len(images), samples_per_class)} images")
    
    print(f"\n✅ Sample dataset created at: {sample_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage Dataset")
    parser.add_argument("--output_dir", type=str, default="./data/plantvillage",
                        help="Output directory")
    parser.add_argument("--sample", action="store_true",
                        help="Create sample dataset only")
    parser.add_argument("--samples_per_class", type=int, default=20,
                        help="Samples per class for sample dataset")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, just organize existing data")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    sample_dir = output_dir / "sample"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PlantVillage Dataset Downloader v2")
    print("="*60)
    
    # Setup Kaggle credentials
    setup_kaggle_credentials()
    
    # Check if data already exists
    existing_data = False
    for check_dir in [raw_dir, processed_dir]:
        if check_dir.exists() and any(check_dir.iterdir()):
            existing_data = True
            print(f"\nFound existing data in {check_dir}")
            break
    
    if args.sample:
        # Find source for sample creation
        source = None
        for src in [processed_dir, raw_dir]:
            if src.exists() and any(src.iterdir()):
                source = src
                break
        
        if source:
            create_sample_dataset(source, sample_dir, args.samples_per_class)
        else:
            print("No source data found. Please download full dataset first.")
        return
    
    if not args.skip_download and not existing_data:
        print("\nDownloading PlantVillage dataset...")
        
        # Try Kaggle first
        print("\nAttempt 1: Kaggle download...")
        datasets = [
            "emmarex/plantdisease",
            "vipoooool/new-plant-diseases-dataset",
            "abdallahalidev/plantvillage-dataset"
        ]
        
        downloaded = False
        for dataset in datasets:
            print(f"\nTrying: {dataset}")
            if download_from_kaggle_cli(dataset, raw_dir):
                downloaded = True
                break
        
        if not downloaded:
            print("\nAttempt 2: Alternative download from HuggingFace...")
            zip_path = download_from_alternative_url(raw_dir)
            if zip_path and zip_path.exists():
                extract_dataset(zip_path, raw_dir)
                downloaded = True
        
        if not downloaded:
            print("\n❌ Download failed. Please download manually from:")
            print("https://www.kaggle.com/datasets/emmarex/plantdisease")
            print(f"\nThen extract to: {raw_dir}")
            return
    
    # Extract any zip files
    zip_files = list(raw_dir.glob("*.zip"))
    for zip_file in zip_files:
        extract_dataset(zip_file, raw_dir)
        # Optionally delete zip after extraction
        # zip_file.unlink()
    
    # Organize dataset
    organize_dataset(raw_dir)
    
    # Create sample dataset from downloaded data
    create_sample_dataset(raw_dir, sample_dir, samples_per_class=20)
    
    print("\n" + "="*60)
    print("✅ Dataset setup complete!")
    print("="*60)
    print(f"Raw data: {raw_dir}")
    print(f"Sample data: {sample_dir}")
    print(f"Processed data: {processed_dir}")

if __name__ == "__main__":
    main()