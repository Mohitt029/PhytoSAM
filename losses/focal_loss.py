"""
Focal Loss for Handling Class Imbalance
Focuses on hard-to-classify examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, 1, H, W) or (B, C, H, W)
            target: Ground truth (B, H, W) for multiclass or (B, 1, H, W) for binary
        """
        # Handle binary vs multiclass
        if pred.shape[1] == 1:
            return self._binary_focal_loss(pred, target)
        else:
            return self._multiclass_focal_loss(pred, target)
    
    def _binary_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary focal loss"""
        # Apply sigmoid
        probs = torch.sigmoid(pred)
        
        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute modulating factor
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Combine
        loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    def _multiclass_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multiclass focal loss"""
        # Apply softmax
        probs = F.softmax(pred, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 3:
            target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        else:
            target_one_hot = target
        
        # Compute cross entropy
        ce_loss = -torch.sum(target_one_hot * torch.log(probs + 1e-8), dim=1)
        
        # Compute p_t
        p_t = torch.sum(target_one_hot * probs, dim=1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """Simplified binary focal loss"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(probs, target, reduction='none')
        
        # Focal weight
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


def test_focal_loss():
    """Test Focal Loss"""
    print("Testing Focal Loss...")
    
    loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Easy example (prediction matches target)
    pred = torch.ones(2, 1, 224, 224) * 10  # High confidence
    target = torch.ones(2, 1, 224, 224)
    loss_val = loss(pred, target)
    print(f"Easy example loss: {loss_val.item():.4f}")
    
    # Hard example (prediction wrong)
    pred = torch.ones(2, 1, 224, 224) * -10  # Wrong confidence
    loss_val = loss(pred, target)
    print(f"Hard example loss: {loss_val.item():.4f} (should be higher)")
    
    print("✅ Focal Loss test passed!")
    return loss


if __name__ == "__main__":
    test_focal_loss()