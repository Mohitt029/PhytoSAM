#!/usr/bin/env python3
"""
Download SAM (Segment Anything Model) weights
"""

import os
import urllib.request
from pathlib import Path
import argparse
from tqdm import tqdm

# SAM model URLs
SAM_WEIGHTS = {
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

class DownloadProgressBar(tqdm):
    """Custom progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_sam_weights(weights_dir: Path, model_type: str = "sam_vit_b"):
    """
    Download SAM weights
    
    Args:
        weights_dir: Directory to save weights
        model_type: "sam_vit_b", "sam_vit_l", or "sam_vit_h"
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type not in SAM_WEIGHTS:
        print(f"Invalid model type. Choose from: {list(SAM_WEIGHTS.keys())}")
        return None
    
    url = SAM_WEIGHTS[model_type]
    file_path = weights_dir / f"{model_type}.pth"
    
    if file_path.exists():
        print(f"Weights already exist at: {file_path}")
        return str(file_path)
    
    print(f"Downloading {model_type} weights...")
    print(f"URL: {url}")
    print(f"Destination: {file_path}")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=model_type) as t:
            urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)
        
        print(f"✅ Download complete: {file_path}")
        print(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        return str(file_path)
    
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

def download_all_weights(weights_dir: Path):
    """Download all SAM model variants"""
    print("Downloading all SAM model variants...")
    print("-" * 40)
    
    for model_type in SAM_WEIGHTS.keys():
        print(f"\nProcessing {model_type}...")
        download_sam_weights(weights_dir, model_type)
    
    print("\n✅ All weights downloaded!")

def main():
    parser = argparse.ArgumentParser(description="Download SAM weights")
    parser.add_argument("--weights_dir", type=str, default="./weights",
                        help="Directory to save weights")
    parser.add_argument("--model_type", type=str, default="sam_vit_b",
                        choices=["sam_vit_b", "sam_vit_l", "sam_vit_h", "all"],
                        help="SAM model type")
    
    args = parser.parse_args()
    
    weights_dir = Path(args.weights_dir)
    
    print("="*60)
    print("SAM Weights Downloader")
    print("="*60)
    
    if args.model_type == "all":
        download_all_weights(weights_dir)
    else:
        download_sam_weights(weights_dir, args.model_type)

if __name__ == "__main__":
    main()