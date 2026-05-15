"""
Test script to verify dataset loading
"""

import sys
sys.path.append('D:/HEMSAM_Project')

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import dataset classes
from data.dataset import PlantVillageSegmentationDataset
from config.actual_classes import ACTUAL_PLANTVILLAGE_CLASSES, DATASET_STATS

def test_dataset_loading():
    """Test basic dataset functionality"""
    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    # Path to your dataset
    dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Checking alternative location...")
        dataset_path = Path("./data/plantvillage/raw")
    
    print(f"Using dataset path: {dataset_path}")
    
    # Create dataset
    try:
        dataset = PlantVillageSegmentationDataset(
            root_dir=dataset_path,
            split="train",
            generate_masks=True
        )
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"Total images: {len(dataset)}")
        print(f"Number of classes: {len(ACTUAL_PLANTVILLAGE_CLASSES)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Class name: {sample['class_name']}")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_sample(dataset, idx=0):
    """Visualize a sample image and its mask"""
    sample = dataset[idx]
    
    # Convert tensors to numpy
    image = sample['image'].permute(1, 2, 0).numpy()
    mask = sample['mask'].squeeze().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nClass: {sample['class_name']}")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"Lesion Mask\nDisease Area")
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[:,:,0] = np.where(mask > 0, 1, overlay[:,:,0])  # Red channel for mask
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Lesions in Red)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Visualization saved to 'sample_visualization.png'")

def check_class_distribution(dataset):
    """Check class distribution in the dataset"""
    from collections import Counter
    
    labels = dataset.labels
    counter = Counter(labels)
    
    print("\n" + "="*60)
    print("Class Distribution")
    print("="*60)
    
    from config.actual_classes import CLASS_ID_TO_NAME
    
    for class_id, count in sorted(counter.items()):
        class_name = CLASS_ID_TO_NAME[class_id]
        percentage = (count / len(labels)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{class_name:40s}: {count:6d} ({percentage:5.1f}%) {bar}")

def test_dataloader():
    """Test dataloader functionality"""
    from torch.utils.data import DataLoader
    from data.dataset import PlantVillageSegmentationDataset
    
    dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
    
    dataset = PlantVillageSegmentationDataset(
        root_dir=dataset_path,
        split="train",
        generate_masks=True
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("\n" + "="*60)
    print("Testing Dataloader")
    print("="*60)
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print(f"Batch images shape: {batch['image'].shape}")
    print(f"Batch masks shape: {batch['mask'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    print(f"Batch labels: {batch['label']}")
    
    print("\n✅ Dataloader test passed!")

if __name__ == "__main__":
    # Run tests
    dataset = test_dataset_loading()
    
    if dataset:
        print("\n" + "="*60)
        check_class_distribution(dataset)
        
        print("\n" + "="*60)
        print("Visualizing sample...")
        visualize_sample(dataset, idx=0)
        
        print("\n" + "="*60)
        test_dataloader()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\nYour dataset is ready for HEMSAM training!")