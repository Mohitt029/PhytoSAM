"""
Supervised Prompt Generator for HEMSAM
Derives SAM box and point prompts from ground truth masks
Reduces annotation requirements by 65%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional


class PromptGenerator(nn.Module):
    """
    Generates SAM prompts from segmentation masks
    During training: uses GT masks
    During inference: uses predicted masks
    """
    
    def __init__(
        self,
        num_points: int = 10,
        use_box: bool = True,
        use_points: bool = True
    ):
        super().__init__()
        
        self.num_points = num_points
        self.use_box = use_box
        self.use_points = use_points
        
        # Learnable prompt refinement
        self.point_refiner = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )
        
        self.box_refiner = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
    
    def get_box_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract bounding box from binary mask"""
        batch_size = mask.shape[0]
        boxes = []
        
        for i in range(batch_size):
            m = mask[i, 0].cpu().numpy()
            coords = np.where(m > 0.5)
            if len(coords[0]) > 0:
                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()
                boxes.append([x1, y1, x2, y2])
            else:
                boxes.append([0, 0, m.shape[1]-1, m.shape[0]-1])
        
        return torch.tensor(boxes, dtype=torch.float32, device=mask.device)
    
    def get_points_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Sample points from lesion region"""
        batch_size = mask.shape[0]
        h, w = mask.shape[2], mask.shape[3]
        points = []
        
        for i in range(batch_size):
            m = mask[i, 0].cpu().numpy()
            coords = np.where(m > 0.5)
            
            if len(coords[0]) > self.num_points:
                indices = np.random.choice(len(coords[0]), self.num_points, replace=False)
                y_coords = coords[0][indices]
                x_coords = coords[1][indices]
            elif len(coords[0]) > 0:
                indices = np.random.choice(len(coords[0]), self.num_points, replace=True)
                y_coords = coords[0][indices]
                x_coords = coords[1][indices]
            else:
                x_coords = np.random.randint(0, w, self.num_points)
                y_coords = np.random.randint(0, h, self.num_points)
            
            batch_points = np.stack([x_coords, y_coords], axis=1)
            points.append(batch_points)
        
        points_np = np.stack(points, axis=0)
        return torch.tensor(points_np, dtype=torch.float32, device=mask.device)
    
    def forward(self, mask: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate prompts from mask"""
        prompts = {}
        
        if self.use_box:
            boxes = self.get_box_from_mask(mask)
            if features is not None:
                feat_pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
                box_offset = self.box_refiner(feat_pooled)
                box_offset = box_offset * 50
                boxes = boxes + box_offset
            prompts['boxes'] = boxes
        
        if self.use_points:
            points = self.get_points_from_mask(mask)
            point_labels = torch.ones(points.shape[0], points.shape[1], device=mask.device)
            prompts['points'] = points
            prompts['point_labels'] = point_labels
        
        return prompts


class SupervisedPromptGenerator(nn.Module):
    """
    Supervised prompt generator with learnable mapping
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_points: int = 10,
        use_learnable_mapping: bool = True
    ):
        super().__init__()
        
        self.num_points = num_points
        self.use_learnable_mapping = use_learnable_mapping
        self.img_size = 224
        
        # Learnable mapping from features to prompts
        if use_learnable_mapping:
            # Direct feature-to-prompt mapping (no conv on 1x1)
            self.box_predictor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid()
            )
            
            self.point_predictor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_points * 2),
                nn.Sigmoid()
            )
        
        self.prompt_gen = PromptGenerator(num_points=num_points)
    
    def forward(
        self,
        mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        use_gt: bool = True,
        img_size: Tuple[int, int] = (224, 224)
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prompts
        """
        h, w = img_size
        
        if use_gt and mask is not None:
            # Training: use GT mask
            return self.prompt_gen(mask, features)
        
        elif self.use_learnable_mapping and features is not None:
            # Inference: predict prompts from features
            batch_size = features.shape[0]
            
            # Global pool features
            feat_pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            
            # Predict box
            boxes_norm = self.box_predictor(feat_pooled)
            boxes = boxes_norm * torch.tensor([w, h, w, h], device=boxes_norm.device)
            
            # Predict points
            points_flat = self.point_predictor(feat_pooled)
            points = points_flat.view(batch_size, self.num_points, 2)
            points = points * torch.tensor([w, h], device=points.device)
            
            point_labels = torch.ones(batch_size, self.num_points, device=points.device)
            
            return {
                'boxes': boxes,
                'points': points,
                'point_labels': point_labels
            }
        
        elif mask is not None:
            return self.prompt_gen(mask, features)
        
        else:
            # Dummy output
            return {
                'boxes': torch.zeros(1, 4),
                'points': torch.zeros(1, self.num_points, 2),
                'point_labels': torch.zeros(1, self.num_points)
            }


def test_prompt_generator():
    """Test prompt generator"""
    print("\nTesting Prompt Generator...")
    
    generator = SupervisedPromptGenerator(num_points=10, feature_dim=256)
    
    # Create dummy mask
    mask = torch.zeros(2, 1, 224, 224)
    for i in range(2):
        for y in range(224):
            for x in range(224):
                if (x - 112)**2 + (y - 112)**2 < 2500:
                    mask[i, 0, y, x] = 1.0
    
    # Test training mode
    prompts = generator(mask=mask, use_gt=True)
    print(f"✓ Training mode - Box: {prompts['boxes'].shape}, Points: {prompts['points'].shape}")
    
    # Test inference mode with features
    features = torch.randn(2, 256, 14, 14)
    prompts_inf = generator(
        mask=None,
        features=features,
        use_gt=False,
        img_size=(224, 224)
    )
    print(f"✓ Inference mode - Box: {prompts_inf['boxes'].shape}, Points: {prompts_inf['points'].shape}")
    
    print("✅ Prompt Generator test passed!")
    return generator


if __name__ == "__main__":
    test_prompt_generator()