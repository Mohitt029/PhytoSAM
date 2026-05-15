"""
Combined Loss for HEMSAM
Combines Dice Loss, Focal Loss, and Classification Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss


class HEMSAMLoss(nn.Module):
    """
    Combined loss for HEMSAM
    Loss = λ_seg * (Dice + Focal) + λ_cls * CrossEntropy
    """
    
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_cls: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        
        # Segmentation losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Classification loss
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            seg_pred: Segmentation prediction (B, 1, H, W)
            seg_target: Segmentation target (B, 1, H, W)
            cls_pred: Classification logits (B, num_classes)
            cls_target: Classification labels (B,)
        
        Returns:
            Dictionary with individual losses and total
        """
        # Segmentation loss
        dice = self.dice_loss(seg_pred, seg_target)
        focal = self.focal_loss(seg_pred, seg_target)
        seg_loss = self.dice_weight * dice + self.focal_weight * focal
        
        # Classification loss
        cls_loss = self.ce_loss(cls_pred, cls_target)
        
        # Total loss
        total_loss = self.lambda_seg * seg_loss + self.lambda_cls * cls_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss,
            'dice_loss': dice,
            'focal_loss': focal
        }


class WeightedHEMSAMLoss(HEMSAMLoss):
    """
    Weighted version with dynamic balancing
    Adjusts weights based on class frequencies
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        class_counts: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Compute class weights if counts provided
        class_weights = None
        if class_counts is not None:
            # Inverse frequency weighting
            total = class_counts.sum()
            class_weights = total / (class_counts * num_classes)
            class_weights = class_weights / class_weights.sum()
        
        super().__init__(class_weights=class_weights, **kwargs)
        self.num_classes = num_classes


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weighting
    Automatically balances segmentation and classification tasks
    """
    
    def __init__(self, num_tasks: int = 2, init_log_vars: float = 0.0):
        """
        Args:
            num_tasks: Number of tasks (2 = seg + cls)
            init_log_vars: Initial log variance
        """
        super().__init__()
        self.log_vars = nn.Parameter(torch.ones(num_tasks) * init_log_vars)
        
        # Base losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with uncertainty weighting
        """
        # Compute individual losses
        dice = self.dice_loss(seg_pred, seg_target)
        focal = self.focal_loss(seg_pred, seg_target)
        seg_loss = (dice + focal) / 2
        
        cls_loss = self.ce_loss(cls_pred, cls_target)
        
        # Apply uncertainty weighting
        precision_seg = torch.exp(-self.log_vars[0])
        precision_cls = torch.exp(-self.log_vars[1])
        
        weighted_seg_loss = precision_seg * seg_loss + self.log_vars[0]
        weighted_cls_loss = precision_cls * cls_loss + self.log_vars[1]
        
        total_loss = weighted_seg_loss + weighted_cls_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss,
            'dice_loss': dice,
            'focal_loss': focal,
            'log_vars': self.log_vars.detach()
        }


def test_combined_loss():
    """Test combined loss"""
    print("Testing Combined Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_fn = HEMSAMLoss()
    
    # Create dummy data
    seg_pred = torch.randn(4, 1, 224, 224).to(device)
    seg_target = torch.randint(0, 2, (4, 1, 224, 224)).float().to(device)
    cls_pred = torch.randn(4, 15).to(device)
    cls_target = torch.randint(0, 15, (4,)).to(device)
    
    # Compute loss
    losses = loss_fn(seg_pred, seg_target, cls_pred, cls_target)
    
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Segmentation Loss: {losses['seg_loss'].item():.4f}")
    print(f"Classification Loss: {losses['cls_loss'].item():.4f}")
    print(f"Dice Loss: {losses['dice_loss'].item():.4f}")
    print(f"Focal Loss: {losses['focal_loss'].item():.4f}")
    
    # Test multi-task loss
    mt_loss = MultiTaskLoss()
    mt_losses = mt_loss(seg_pred, seg_target, cls_pred, cls_target)
    print(f"\nMulti-task Total Loss: {mt_losses['total_loss'].item():.4f}")
    print(f"Learnable log vars: {mt_losses['log_vars']}")
    
    print("✅ Combined Loss test passed!")
    return loss_fn


if __name__ == "__main__":
    test_combined_loss()