"""
Dice Loss for Segmentation
Measures overlap between predicted and ground truth masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Loss = 1 - (2 * |A ∩ B|) / (|A| + |B|)
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted mask (B, 1, H, W) - values in [0, 1]
            target: Ground truth mask (B, 1, H, W) - values in [0, 1]
        
        Returns:
            Dice loss
        """
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss with sigmoid activation built-in"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute Dice
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


def test_dice_loss():
    """Test Dice Loss"""
    print("Testing Dice Loss...")
    
    loss = DiceLoss()
    
    # Perfect prediction
    pred = torch.ones(2, 1, 224, 224)
    target = torch.ones(2, 1, 224, 224)
    loss_val = loss(pred, target)
    print(f"Perfect prediction loss: {loss_val.item():.4f} (should be ~0)")
    
    # Random prediction
    pred = torch.rand(2, 1, 224, 224)
    target = torch.randint(0, 2, (2, 1, 224, 224)).float()
    loss_val = loss(pred, target)
    print(f"Random prediction loss: {loss_val.item():.4f}")
    
    print("✅ Dice Loss test passed!")
    return loss


if __name__ == "__main__":
    test_dice_loss()