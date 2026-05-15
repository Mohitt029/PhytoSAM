"""
Custom Augmentations for Plant Leaf Disease Detection
Using torchvision.transforms for better compatibility
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Dict, Optional, Any, Tuple
from PIL import Image


class ToTensor:
    """Convert numpy arrays to torch tensors"""
    def __call__(self, image, mask=None):
        # Convert to tensor and normalize to [0, 1]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        if mask is not None:
            return image, mask
        return image


class Normalize:
    """Normalize image with mean and std"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, mask=None):
        # Normalize image
        for i in range(3):
            image[i] = (image[i] - self.mean[i]) / self.std[i]
        
        if mask is not None:
            return image, mask
        return image


class Resize:
    """Resize image and mask"""
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, image, mask=None):
        # Convert to PIL for resizing
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_resized = image_pil.resize(self.size, Image.BILINEAR)
        image = np.array(image_resized) / 255.0
        
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().numpy()
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize(self.size, Image.NEAREST)
            mask = np.array(mask_resized) / 255.0
        
        if mask is not None:
            return image, mask
        return image


class RandomHorizontalFlip:
    """Random horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        
        if mask is not None:
            return image, mask
        return image


class RandomVerticalFlip:
    """Random vertical flip"""
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            image = np.flipud(image).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        
        if mask is not None:
            return image, mask
        return image


class RandomRotation:
    """Random rotation"""
    def __init__(self, degrees=45, p=0.7):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            angle = np.random.uniform(-self.degrees, self.degrees)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        if mask is not None:
            return image, mask
        return image


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment"""
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.7):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            brightness = 1 + np.random.uniform(-self.brightness_limit, self.brightness_limit)
            contrast = 1 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            image = image * contrast + brightness
            image = np.clip(image, 0, 1)
        
        if mask is not None:
            return image, mask
        return image


class RandomGamma:
    """Random gamma correction"""
    def __init__(self, gamma_limit=(80, 120), p=0.3):
        self.gamma_limit = gamma_limit
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            gamma = np.random.uniform(self.gamma_limit[0] / 100, self.gamma_limit[1] / 100)
            image = np.power(image, gamma)
            image = np.clip(image, 0, 1)
        
        if mask is not None:
            return image, mask
        return image


class GaussianBlur:
    """Apply Gaussian blur"""
    def __init__(self, blur_limit=(3, 5), p=0.2):
        self.blur_limit = blur_limit
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            ksize = np.random.randint(self.blur_limit[0], self.blur_limit[1] + 1)
            if ksize % 2 == 0:
                ksize += 1
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        if mask is not None:
            return image, mask
        return image


class Cutout:
    """Apply cutout augmentation (simulate occlusion)"""
    def __init__(self, num_holes=8, hole_size=16, p=0.3):
        self.num_holes = num_holes
        self.hole_size = hole_size
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.random() < self.p:
            h, w = image.shape[:2]
            for _ in range(self.num_holes):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                y1 = max(0, y - self.hole_size // 2)
                y2 = min(h, y + self.hole_size // 2)
                x1 = max(0, x - self.hole_size // 2)
                x2 = min(w, x + self.hole_size // 2)
                image[y1:y2, x1:x2] = 0
                if mask is not None:
                    mask[y1:y2, x1:x2] = 0
        
        if mask is not None:
            return image, mask
        return image


def get_augmentation(phase: str = "train", config: Dict = None):
    """
    Get augmentation pipeline for training or validation
    
    Args:
        phase: "train" or "val"
        config: Optional configuration (ignored)
    
    Returns:
        Composed augmentation function
    """
    
    if phase == "train":
        # Training augmentations
        transforms_list = [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=45, p=0.7),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            RandomGamma(gamma_limit=(80, 120), p=0.3),
            GaussianBlur(blur_limit=(3, 5), p=0.2),
            Cutout(num_holes=8, hole_size=16, p=0.3),
            Resize(size=(224, 224)),
        ]
    else:
        # Validation augmentations - only resize
        transforms_list = [
            Resize(size=(224, 224)),
        ]
    
    class Compose:
        """Compose multiple transforms"""
        def __init__(self, transforms, phase):
            self.transforms = transforms
            self.phase = phase
        
        def __call__(self, image, mask=None):
            for t in self.transforms:
                if mask is not None:
                    image, mask = t(image, mask)
                else:
                    image = t(image)
            
            # Convert to tensor and normalize
            if mask is not None:
                image_tensor, mask_tensor = ToTensor()(image, mask)
                image_tensor, mask_tensor = Normalize()(image_tensor, mask_tensor)
                return {"image": image_tensor, "mask": mask_tensor}
            else:
                image_tensor = ToTensor()(image)
                image_tensor = Normalize()(image_tensor)
                return {"image": image_tensor}
    
    return Compose(transforms_list, phase)


class PlantDiseaseAugmentation:
    """Wrapper class for augmentation pipeline"""
    
    def __init__(self, config: Dict = None, phase: str = "train"):
        self.phase = phase
        self.transform = get_augmentation(phase)
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Apply augmentations"""
        return self.transform(image, mask)


def test_augmentation():
    """Test the augmentation pipeline"""
    print("Testing augmentation pipeline...")
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) / 255.0
    dummy_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    
    # Test training augmentation
    train_aug = get_augmentation("train")
    result = train_aug(image=dummy_image, mask=dummy_mask)
    
    print(f"Training augmentation:")
    print(f"  Image shape: {result['image'].shape}")
    print(f"  Mask shape: {result['mask'].shape}")
    print(f"  Image range: [{result['image'].min():.2f}, {result['image'].max():.2f}]")
    
    # Test validation augmentation
    val_aug = get_augmentation("val")
    result = val_aug(image=dummy_image, mask=dummy_mask)
    
    print(f"\nValidation augmentation:")
    print(f"  Image shape: {result['image'].shape}")
    print(f"  Mask shape: {result['mask'].shape}")
    
    print("\n✅ Augmentation test passed!")
    return True


if __name__ == "__main__":
    test_augmentation()