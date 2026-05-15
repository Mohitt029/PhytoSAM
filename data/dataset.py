"""
PlantVillage Dataset Loader with Segmentation Masks
Handles both classification and segmentation tasks
Updated for the actual PlantVillage dataset structure
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

# Import actual classes from your dataset
from config.actual_classes import (
    ACTUAL_PLANTVILLAGE_CLASSES, 
    CLASS_ID_TO_NAME, 
    DATASET_STATS
)
from data.augmentation import get_augmentation


class PlantVillageSegmentationDataset(Dataset):
    """
    PlantVillage Dataset with Synthetic Segmentation Masks
    Updated to work with the actual downloaded dataset structure
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        generate_masks: bool = True,
        mask_type: str = "synthetic",  # "synthetic", "contour", "gaussian"
        use_actual_classes: bool = True,  # Use actual dataset classes
    ):
        """
        Args:
            root_dir: Path to PlantVillage dataset directory
            split: "train", "val", or "test"
            transform: Augmentation pipeline
            generate_masks: Whether to generate segmentation masks
            mask_type: Type of mask generation
            use_actual_classes: Use actual classes from your dataset
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.generate_masks = generate_masks
        self.mask_type = mask_type
        self.use_actual_classes = use_actual_classes
        
        # Use actual classes from your dataset
        if use_actual_classes:
            from config.actual_classes import ACTUAL_PLANTVILLAGE_CLASSES, CLASS_ID_TO_NAME
            self.class_mapping = ACTUAL_PLANTVILLAGE_CLASSES
            self.id_to_name = CLASS_ID_TO_NAME
        else:
            from config.dataset_config import HEMSAM_CLASSES, CLASS_ID_TO_NAME
            self.class_mapping = HEMSAM_CLASSES
            self.id_to_name = CLASS_ID_TO_NAME
        
        # Setup transform
        if transform is None:
            from data.augmentation import get_augmentation
            self.transform = get_augmentation(split)
        else:
            self.transform = transform
        
        # Load all image paths and labels
        self.images = []
        self.labels = []
        self.class_names = []
        
        self._load_dataset()
        
        # Print statistics
        print(f"Loaded {len(self.images)} images for {split} set")
        self._print_class_distribution()
    
    def _load_dataset(self):
        """
        Load all image paths and their corresponding labels
        Handles the specific folder structure from Kaggle download
        """
        # Your dataset structure: root_dir/ClassName/*.jpg
        # Class names like: "Pepper__bell___Bacterial_spot"
        
        for class_name, class_id in self.class_mapping.items():
            # Try multiple possible directory names
            possible_dirs = [
                self.root_dir / class_name,
                self.root_dir / class_name.replace("_", " "),
                self.root_dir / class_name.replace("___", "_"),
                self.root_dir / class_name.replace("__", "_"),
                self.root_dir / class_name.lower(),
                self.root_dir / class_name.replace("_", "/"),
            ]
            
            class_dir = None
            for pdir in possible_dirs:
                if pdir.exists() and pdir.is_dir():
                    class_dir = pdir
                    break
            
            if class_dir is None:
                # Check if there's a directory containing this class name
                for item in self.root_dir.iterdir():
                    if item.is_dir() and class_name.lower() in item.name.lower():
                        class_dir = item
                        break
            
            if class_dir is None:
                print(f"Warning: Class directory not found for {class_name}")
                continue
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(class_dir.glob(ext))
            
            # Add to dataset
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(class_id)
                self.class_names.append(class_name)
            
            if len(image_files) > 0:
                print(f"  Loaded {len(image_files)} images from {class_dir.name}")
    
    def _print_class_distribution(self):
        """Print class distribution statistics"""
        from collections import Counter
        counter = Counter(self.labels)
        print(f"\nClass distribution ({self.split} set):")
        print("-" * 50)
        total = sum(counter.values())
        for class_id, count in sorted(counter.items()):
            class_name = self.id_to_name[class_id]
            # Truncate long class names
            display_name = class_name[:35] + "..." if len(class_name) > 35 else class_name
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"  {display_name:40s}: {count:6d} ({percentage:5.1f}%) {bar}")
    
    def _generate_lesion_mask(
        self, 
        image: np.ndarray, 
        class_id: int
    ) -> np.ndarray:
        """
        Generate synthetic lesion mask based on disease class
        
        Args:
            image: RGB image array (H, W, 3)
            class_id: Disease class ID
            
        Returns:
            Binary mask (H, W) where 1 = lesion, 0 = healthy
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # For healthy class, return all zeros
        class_name = self.id_to_name[class_id]
        if "healthy" in class_name.lower():
            return mask
        
        # Different mask generation strategies
        if self.mask_type == "synthetic":
            mask = self._synthetic_mask_by_disease(image, class_name)
        elif self.mask_type == "contour":
            mask = self._contour_based_mask(image)
        elif self.mask_type == "gaussian":
            mask = self._gaussian_lesion_mask(image)
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8) * 255
        
        return mask
    
    def _synthetic_mask_by_disease(self, image: np.ndarray, disease_name: str) -> np.ndarray:
        """
        Generate disease-specific synthetic masks based on typical lesion patterns
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to HSV for better color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Disease detection based on class name
        disease_lower = disease_name.lower()
        
        if "blight" in disease_lower:
            # Blight causes irregular brown patches
            # Threshold on brown color range
            lower_brown = np.array([10, 50, 50])
            upper_brown = np.array([30, 200, 200])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Add random irregular patches
            num_patches = np.random.randint(3, 8)
            for _ in range(num_patches):
                cx = np.random.randint(w//4, 3*w//4)
                cy = np.random.randint(h//4, 3*h//4)
                radius = np.random.randint(20, min(h, w)//6)
                cv2.circle(brown_mask, (cx, cy), radius, 255, -1)
            
            mask = brown_mask
            
        elif "bacterial_spot" in disease_lower or "bacterial spot" in disease_lower:
            # Bacterial spot causes small angular lesions
            mask = self._generate_circular_lesions(h, w, num=40, min_r=2, max_r=10)
            
            # Add some irregular shapes
            num_irregular = np.random.randint(5, 15)
            for _ in range(num_irregular):
                cx = np.random.randint(10, w-10)
                cy = np.random.randint(10, h-10)
                axes = (np.random.randint(3, 8), np.random.randint(3, 8))
                angle = np.random.randint(0, 180)
                cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
            
        elif "rust" in disease_lower:
            # Rust causes small circular orange/brown spots
            mask = self._generate_circular_lesions(h, w, num=50, min_r=2, max_r=8)
            
        elif "mildew" in disease_lower:
            # Powdery mildew appears as white powdery patches
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Dilate to create patches
            kernel = np.ones((10, 10), np.uint8)
            white_mask = cv2.dilate(white_mask, kernel, iterations=2)
            mask = white_mask
            
        elif "leaf_mold" in disease_lower:
            # Leaf mold causes fuzzy patches on leaf surface
            mask = self._generate_random_patches(h, w, num_patches=8)
            # Add blur to make it fuzzy
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = (mask > 50).astype(np.uint8) * 255
            
        elif "mosaic_virus" in disease_lower or "yellowleaf" in disease_lower:
            # Viral diseases cause mottling/yellowing
            mask = self._generate_mosaic_pattern(h, w)
            
        elif "spider_mites" in disease_lower:
            # Spider mite damage causes stippling (tiny spots)
            mask = self._generate_circular_lesions(h, w, num=200, min_r=1, max_r=3)
            
        elif "target_spot" in disease_lower:
            # Target spot has concentric rings
            mask = self._generate_target_spot_pattern(h, w)
            
        elif "septoria" in disease_lower:
            # Septoria leaf spot causes small circular lesions with dark borders
            mask = self._generate_circular_lesions(h, w, num=60, min_r=3, max_r=12)
            
        else:
            # Default: generate random irregular patches
            mask = self._generate_random_patches(h, w, num_patches=5)
        
        # Post-processing: ensure mask is binary and clean
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Morphological operations to make masks realistic
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Ensure mask isn't empty (add small random patch if needed)
        if np.sum(mask) < 100:
            cx = np.random.randint(w//4, 3*w//4)
            cy = np.random.randint(h//4, 3*h//4)
            r = np.random.randint(20, min(h, w)//8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
        
        return mask
    
    def _generate_circular_lesions(
        self, 
        h: int, 
        w: int, 
        num: int = 30, 
        min_r: int = 3, 
        max_r: int = 15
    ) -> np.ndarray:
        """Generate random circular lesions"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num):
            cx = np.random.randint(10, w-10)
            cy = np.random.randint(10, h-10)
            r = np.random.randint(min_r, max_r)
            cv2.circle(mask, (cx, cy), r, 255, -1)
        
        return mask
    
    def _generate_random_patches(
        self, 
        h: int, 
        w: int, 
        num_patches: int = 5
    ) -> np.ndarray:
        """Generate random irregular patches"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num_patches):
            cx = np.random.randint(w//4, 3*w//4)
            cy = np.random.randint(h//4, 3*h//4)
            
            # Random ellipse
            axes = (
                np.random.randint(20, w//8),
                np.random.randint(20, h//8)
            )
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        
        return mask
    
    def _generate_mosaic_pattern(self, h: int, w: int) -> np.ndarray:
        """Generate mosaic pattern for viral diseases"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create grid of irregular patches
        cell_size = np.random.randint(15, 35)
        for i in range(0, h, cell_size):
            for j in range(0, w, cell_size):
                if np.random.random() > 0.5:
                    mask[i:min(i+cell_size, h), j:min(j+cell_size, w)] = 255
        
        return mask
    
    def _generate_target_spot_pattern(self, h: int, w: int) -> np.ndarray:
        """Generate target spot pattern with concentric rings"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        num_lesions = np.random.randint(3, 8)
        for _ in range(num_lesions):
            cx = np.random.randint(w//4, 3*w//4)
            cy = np.random.randint(h//4, 3*h//4)
            r1 = np.random.randint(15, 25)
            r2 = r1 + np.random.randint(5, 10)
            r3 = r2 + np.random.randint(5, 10)
            
            cv2.circle(mask, (cx, cy), r3, 100, -1)
            cv2.circle(mask, (cx, cy), r2, 200, -1)
            cv2.circle(mask, (cx, cy), r1, 255, -1)
        
        return mask
    
    def _contour_based_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate mask based on image contours (edge-based)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from largest contours
        mask = np.zeros_like(gray)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask
    
    def _gaussian_lesion_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate smooth lesion mask using Gaussian blobs"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        num_blobs = np.random.randint(2, 6)
        for _ in range(num_blobs):
            cx = np.random.rand() * w
            cy = np.random.rand() * h
            sx = np.random.rand() * w * 0.2 + 20
            sy = np.random.rand() * h * 0.2 + 20
            
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            
            gaussian = np.exp(-((X - cx)**2 / (2 * sx**2) + (Y - cy)**2 / (2 * sy**2)))
            mask = np.maximum(mask, gaussian)
        
        # Threshold to get binary mask
        mask = (mask > 0.3).astype(np.uint8) * 255
        return mask
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'image': RGB image tensor (3, H, W)
                - 'mask': Binary mask tensor (1, H, W)
                - 'label': Disease class label
                - 'class_name': String class name
        """
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Return a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        class_name = self.class_names[idx]
        
        # Generate mask if needed
        if self.generate_masks:
            mask = self._generate_lesion_mask(image, label)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply augmentations
        augmented = self.transform(image=image, mask=mask)
        
        # Convert mask to tensor and add channel dimension
        if "mask" in augmented:
            mask_tensor = augmented["mask"].unsqueeze(0).float() / 255.0
        else:
            mask_tensor = torch.zeros((1, 224, 224))
        
        return {
            "image": augmented["image"],
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "image_path": str(img_path),
        }


class PlantVillageClassificationDataset(Dataset):
    """
    Simplified classification-only dataset (no masks)
    Used for evaluating classification accuracy
    """
    
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        from config.actual_classes import ACTUAL_PLANTVILLAGE_CLASSES
        
        self.root_dir = Path(root_dir)
        self.images = []
        self.labels = []
        
        # Load dataset using actual classes
        for class_name, class_id in ACTUAL_PLANTVILLAGE_CLASSES.items():
            class_dir = self.root_dir / class_name
            
            # Try alternative names
            if not class_dir.exists():
                alt_names = [
                    class_name.replace("___", "_"),
                    class_name.replace("__", "_"),
                    class_name.lower(),
                ]
                for alt in alt_names:
                    alt_dir = self.root_dir / alt
                    if alt_dir.exists():
                        class_dir = alt_dir
                        break
            
            if class_dir.exists():
                for img_path in class_dir.glob("*.[jJ][pP][gG]"):
                    self.images.append(img_path)
                    self.labels.append(class_id)
                for img_path in class_dir.glob("*.png"):
                    self.images.append(img_path)
                    self.labels.append(class_id)
        
        if transform is None:
            self.transform = get_augmentation(split)
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.images)} images for classification ({split})")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        augmented = self.transform(image=image)
        
        return {
            "image": augmented["image"],
            "label": torch.tensor(label, dtype=torch.long),
        }


# Utility function to create data loaders
def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    generate_masks: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        root_dir: Path to PlantVillage dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        generate_masks: Whether to generate segmentation masks
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split
    
    # First, create full dataset
    full_dataset = PlantVillageSegmentationDataset(
        root_dir=root_dir,
        split="full",
        generate_masks=generate_masks
    )
    
    # Get all indices
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    
    # Split indices (stratified by class)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    # Update split attribute for proper augmentation
    train_dataset.dataset.split = "train"
    val_dataset.dataset.split = "val"
    test_dataset.dataset.split = "test"
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"\n{'='*50}")
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"{'='*50}")
    
    return train_loader, val_loader, test_loader


# Quick test function
def test_dataset():
    """Quick test to verify dataset loading"""
    print("Testing dataset loading...")
    
    # Path to your dataset
    dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
    
    if not dataset_path.exists():
        # Try alternative path
        dataset_path = Path("./data/plantvillage/raw")
    
    print(f"Looking for dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    try:
        dataset = PlantVillageSegmentationDataset(
            root_dir=dataset_path,
            split="train",
            generate_masks=True
        )
        
        print(f"\n✅ Dataset test passed!")
        print(f"Total images: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"\nSample output shapes:")
        print(f"  Image: {sample['image'].shape}")
        print(f"  Mask: {sample['mask'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Class: {sample['class_name']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dataset()