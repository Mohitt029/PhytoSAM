"""
Evaluation Metrics for Segmentation and Classification
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report


class SegmentationMetrics:
    """Compute segmentation metrics: IoU, Dice, Precision, Recall"""
    
    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.intersections = torch.zeros(self.num_classes)
        self.unions = torch.zeros(self.num_classes)
        self.total_pixels = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with batch
        
        Args:
            pred: Prediction mask (B, 1, H, W) - values in [0, 1]
            target: Target mask (B, 1, H, W) - values in [0, 1]
        """
        # Threshold predictions
        pred_binary = (pred > 0.5).float()
        target_binary = target.float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # Compute confusion matrix
        for c in range(self.num_classes):
            pred_c = (pred_flat == c).float()
            target_c = (target_flat == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            self.intersections[c] += intersection
            self.unions[c] += union
        
        self.total_pixels += pred_flat.shape[0]
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        iou = (self.intersections + self.smooth) / (self.unions + self.smooth)
        dice = (2 * self.intersections + self.smooth) / (self.unions + self.intersections + self.smooth)
        
        # Background (class 0) and Foreground (class 1)
        return {
            'iou_background': iou[0].item(),
            'iou_foreground': iou[1].item(),
            'mean_iou': iou.mean().item(),
            'dice_background': dice[0].item(),
            'dice_foreground': dice[1].item(),
            'mean_dice': dice.mean().item()
        }


class ClassificationMetrics:
    """Compute classification metrics: Accuracy, Precision, Recall, F1"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with batch"""
        pred_classes = torch.argmax(pred, dim=1)
        self.predictions.extend(pred_classes.cpu().numpy())
        self.targets.extend(target.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all classification metrics"""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Compute accuracy
        accuracy = (preds == targets).mean()
        
        # Per-class metrics
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        
        # Precision, Recall, F1 per class
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1 = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision[i] = tp / (tp + fp + 1e-8)
            recall[i] = tp / (tp + fn + 1e-8)
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)
        
        return {
            'accuracy': accuracy,
            'mean_precision': precision.mean(),
            'mean_recall': recall.mean(),
            'mean_f1': f1.mean(),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist()
        }


class HEMSAMMetrics:
    """Combined metrics for HEMSAM"""
    
    def __init__(self, num_classes: int = 15):
        self.seg_metrics = SegmentationMetrics(num_classes=2)
        self.cls_metrics = ClassificationMetrics(num_classes=num_classes)
        self.num_classes = num_classes
    
    def reset(self):
        self.seg_metrics.reset()
        self.cls_metrics.reset()
    
    def update(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor
    ):
        self.seg_metrics.update(seg_pred, seg_target)
        self.cls_metrics.update(cls_pred, cls_target)
    
    def compute(self) -> Dict[str, float]:
        seg_results = self.seg_metrics.compute()
        cls_results = self.cls_metrics.compute()
        
        return {
            **seg_results,
            **cls_results
        }


def test_metrics():
    """Test metrics computation"""
    print("Testing Metrics...")
    
    # Test segmentation metrics
    seg_metrics = SegmentationMetrics()
    
    # Perfect prediction
    pred = torch.ones(2, 1, 224, 224)
    target = torch.ones(2, 1, 224, 224)
    seg_metrics.update(pred, target)
    
    results = seg_metrics.compute()
    print(f"Perfect IoU: {results['mean_iou']:.4f}")
    
    # Test classification metrics
    cls_metrics = ClassificationMetrics(num_classes=15)
    pred = torch.randn(10, 15)
    target = torch.randint(0, 15, (10,))
    cls_metrics.update(pred, target)
    
    results = cls_metrics.compute()
    print(f"Classification accuracy: {results['accuracy']:.4f}")
    
    print("✅ Metrics test passed!")
    return seg_metrics


if __name__ == "__main__":
    test_metrics()