"""
Classification Head for HEMSAM
Performs 26-class disease classification using segmented region features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class ClassificationHead(nn.Module):
    """
    Classification head for plant disease classification
    Uses features from segmented lesion regions
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 15,  # Change to 26 for full dataset
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Global pooling options
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Optional: Attention-based pooling
        self.use_attention_pool = True
        self.attention_pool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classify disease from features
        
        Args:
            features: Feature map from encoder (B, C, H, W)
            mask: Segmentation mask (B, 1, H, W) for weighted pooling
        
        Returns:
            Class logits (B, num_classes)
        """
        if self.use_attention_pool and mask is not None:
            # Use mask as attention
            if mask.shape[-2:] != features.shape[-2:]:
                mask = F.interpolate(mask, size=features.shape[-2:], mode='bilinear')
            
            # Weighted pooling using mask
            weighted_features = features * mask
            pooled = weighted_features.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)
        else:
            # Global average pooling
            pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class MultiScaleClassificationHead(nn.Module):
    """
    Multi-scale classification head
    Combines features from multiple scales for better accuracy
    """
    
    def __init__(
        self,
        feature_channels: Tuple[int, ...] = (256, 128, 64),
        num_classes: int = 15,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_scales = len(feature_channels)
        
        # Separate classifiers for each scale
        self.classifiers = nn.ModuleList([
            ClassificationHead(ch, num_classes, dropout=dropout)
            for ch in feature_channels
        ])
        
        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-scale classification
        
        Args:
            features: Dictionary of features at different scales
            mask: Segmentation mask
        
        Returns:
            final_logits: Fused logits
            scale_logits: Individual scale logits
        """
        scale_logits = []
        
        # Convert dict to list if needed
        feature_list = list(features.values()) if isinstance(features, dict) else features
        
        for i in range(min(self.num_scales, len(feature_list))):
            logits = self.classifiers[i](feature_list[i], mask)
            scale_logits.append(logits)
        
        # Fuse with softmax weights
        weights = F.softmax(self.fusion_weights, dim=0)
        final_logits = torch.zeros_like(scale_logits[0])
        
        for i, logits in enumerate(scale_logits):
            final_logits = final_logits + weights[i] * logits
        
        return final_logits, torch.stack(scale_logits, dim=1)


def test_classification_head():
    """Test classification head"""
    print("\nTesting Classification Head...")
    
    # Test basic head
    head = ClassificationHead(in_channels=256, num_classes=15)
    features = torch.randn(2, 256, 14, 14)
    mask = torch.ones(2, 1, 14, 14)
    
    logits = head(features, mask)
    print(f"✓ Basic head: input {features.shape} → logits {logits.shape}")
    
    # Test multi-scale head
    ms_head = MultiScaleClassificationHead(
        feature_channels=(256, 128, 64),
        num_classes=15
    )
    
    features_dict = {
        'scale1': torch.randn(2, 256, 14, 14),
        'scale2': torch.randn(2, 128, 28, 28),
        'scale3': torch.randn(2, 64, 56, 56)
    }
    
    final_logits, scale_logits = ms_head(features_dict, mask)
    print(f"✓ Multi-scale head: final {final_logits.shape}, scales {scale_logits.shape}")
    
    print("✅ Classification Head test passed!")
    return head


if __name__ == "__main__":
    test_classification_head()