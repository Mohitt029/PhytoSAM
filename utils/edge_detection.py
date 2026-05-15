"""
Edge detection utilities for lesion boundary enhancement
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EdgeDetector:
    """
    Edge detection utilities for plant leaf lesion boundaries
    """
    
    @staticmethod
    def sobel_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect edges using Sobel operator
        
        Args:
            image: RGB image (H, W, 3) in uint8 format (0-255)
        Returns:
            edge_magnitude, edge_direction
        """
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        
        # Normalize magnitude
        if magnitude.max() > magnitude.min():
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude, direction
    
    @staticmethod
    def canny_edges(
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges using Canny edge detector
        
        Args:
            image: RGB image (H, W, 3) in uint8 format
        Returns:
            Binary edge map (0-1 float)
        """
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Canny
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        return edges / 255.0
    
    @staticmethod
    def laplacian_edges(image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Laplacian operator
        """
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Normalize
        laplacian = np.abs(laplacian)
        if laplacian.max() > laplacian.min():
            laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-8)
        
        return laplacian


class BoundaryEnhancer:
    """
    Enhance lesion boundaries using morphological operations
    """
    
    @staticmethod
    def morphological_boundary(
        mask: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Extract boundary using morphological erosion
        
        Args:
            mask: Binary mask (H, W) - values 0 or 1 (float) or 0-255 (uint8)
            kernel_size: Size of structuring element
        Returns:
            Boundary map (float32)
        """
        # Ensure binary mask in uint8 format (0 or 255)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Ensure binary
        mask = (mask > 127).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Erode the mask
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Boundary = original - eroded
        boundary = mask - eroded
        
        return boundary.astype(np.float32) / 255.0
    
    @staticmethod
    def gradient_boundary(mask: np.ndarray) -> np.ndarray:
        """
        Extract boundary using gradient
        """
        # Ensure float32
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        # Compute gradient
        grad_x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
        
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to get boundary
        boundary = (gradient > 0.1).astype(np.float32)
        
        return boundary
    
    @staticmethod
    def enhance_lesion_boundary(
        image: np.ndarray,
        mask: np.ndarray,
        strength: float = 0.3
    ) -> np.ndarray:
        """
        Enhance lesion boundaries in the image
        """
        # Ensure float32 for image
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Extract boundary
        boundary = BoundaryEnhancer.morphological_boundary(mask)
        
        # Dilate boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary_dilated = cv2.dilate(boundary.astype(np.float32), kernel, iterations=1)
        
        # Enhance boundary
        enhanced = image.copy()
        for c in range(3):
            enhanced[:, :, c] = np.where(
                boundary_dilated > 0,
                image[:, :, c] + strength,
                image[:, :, c]
            )
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced


class TorchEdgeDetector(nn.Module):
    """
    PyTorch version of edge detection for GPU processing
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect edges in batch of images
        
        Args:
            x: Input tensor (B, C, H, W) - values in [0, 1]
        Returns:
            Edge map (B, 1, H, W)
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Compute magnitude
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # Normalize
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        
        return edge_mag


def test_edge_detection():
    """Test edge detection utilities"""
    print("Testing edge detection utilities...")
    
    # Create dummy image in uint8 format (0-255)
    dummy_image_uint8 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(dummy_mask, (128, 128), 50, 255, -1)
    
    # Test edge detectors
    detector = EdgeDetector()
    
    sobel_mag, sobel_dir = detector.sobel_edges(dummy_image_uint8)
    print(f"Sobel edges: shape={sobel_mag.shape}, range=[{sobel_mag.min():.3f}, {sobel_mag.max():.3f}]")
    
    canny = detector.canny_edges(dummy_image_uint8)
    print(f"Canny edges: shape={canny.shape}, range=[{canny.min():.3f}, {canny.max():.3f}]")
    
    laplacian = detector.laplacian_edges(dummy_image_uint8)
    print(f"Laplacian edges: shape={laplacian.shape}, range=[{laplacian.min():.3f}, {laplacian.max():.3f}]")
    
    # Test boundary enhancement
    enhancer = BoundaryEnhancer()
    boundary = enhancer.morphological_boundary(dummy_mask)
    print(f"Boundary: shape={boundary.shape}, max={boundary.max():.3f}")
    
    # Test torch version
    torch_detector = TorchEdgeDetector()
    x = torch.randn(2, 3, 224, 224)
    edges = torch_detector(x)
    print(f"\nTorch edge detector output: {edges.shape}")
    
    print("\n✅ Edge detection test passed!")


if __name__ == "__main__":
    test_edge_detection()