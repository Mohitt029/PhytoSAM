#!/usr/bin/env python3
"""
Evaluation script for HEMSAM
Computes detailed metrics and generates visualizations
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hemsam import HEMSAM
from data.dataset import PlantVillageSegmentationDataset
from training.metrics import HEMSAMMetrics
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HEMSAM model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/plantvillage/raw/plantvillage/PlantVillage',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    return parser.parse_args()


class HEMSAMEvaluator:
    """Evaluator for HEMSAM model"""
    
    def __init__(self, model, device, output_dir: str):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = HEMSAMMetrics(num_classes=15)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model on dataloader"""
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_images = []
        all_masks = []
        all_seg_preds = []
        
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, return_attention=True)
            
            # Update metrics
            self.metrics.update(
                outputs['segmentation'],
                masks,
                outputs['classification'],
                labels
            )
            
            # Store for visualization
            all_predictions.extend(torch.argmax(outputs['classification'], dim=1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_seg_preds.append(outputs['segmentation'].cpu())
        
        # Compute metrics
        results = self.metrics.compute()
        
        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return results, cm, all_images, all_masks, all_seg_preds
    
    def generate_report(self, results: dict, cm: np.ndarray) -> str:
        """Generate evaluation report"""
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("HEMSAM Evaluation Report")
        report_lines.append("="*60)
        report_lines.append("")
        
        # Segmentation metrics
        report_lines.append("Segmentation Metrics:")
        report_lines.append(f"  Mean IoU: {results.get('mean_iou', 0):.4f}")
        report_lines.append(f"  Mean Dice: {results.get('mean_dice', 0):.4f}")
        report_lines.append(f"  Foreground IoU: {results.get('iou_foreground', 0):.4f}")
        report_lines.append(f"  Foreground Dice: {results.get('dice_foreground', 0):.4f}")
        report_lines.append("")
        
        # Classification metrics
        report_lines.append("Classification Metrics:")
        report_lines.append(f"  Accuracy: {results.get('accuracy', 0):.4f}")
        report_lines.append(f"  Mean Precision: {results.get('mean_precision', 0):.4f}")
        report_lines.append(f"  Mean Recall: {results.get('mean_recall', 0):.4f}")
        report_lines.append(f"  Mean F1: {results.get('mean_f1', 0):.4f}")
        report_lines.append("")
        
        # Save to file
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        # Also print
        print("\n".join(report_lines))
        
        return "\n".join(report_lines)
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names[:15], yticklabels=class_names[:15])
        plt.title('Confusion Matrix - HEMSAM Classification')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def visualize_predictions(self, images, masks, seg_preds, num_samples: int = 5):
        """Visualize sample predictions"""
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        for i in range(min(num_samples, len(images))):
            # Denormalize image
            img = images[i].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # Ground truth mask
            gt_mask = masks[i, 0].numpy()
            
            # Predicted mask
            pred_mask = (seg_preds[i, 0].numpy() > 0.5).astype(np.float32)
            
            # Overlay
            overlay = img.copy()
            overlay[:, :, 0] = np.where(pred_mask > 0, 1, overlay[:, :, 0])  # Red channel
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("Overlay (Lesions in Red)")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / "sample_predictions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Sample predictions saved to {save_path}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = HEMSAM(
        sam_checkpoint=None,
        num_classes=15,
        use_sam=False,
        use_cdms=True,
        use_uhdam=True
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from {args.checkpoint}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = PlantVillageSegmentationDataset(
        root_dir=args.data_dir,
        split="test",
        generate_masks=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Test set size: {len(test_dataset)} images")
    
    # Evaluate
    print("\nEvaluating...")
    evaluator = HEMSAMEvaluator(model, device, args.output_dir)
    results, cm, images, masks, seg_preds = evaluator.evaluate(test_loader)
    
    # Generate report
    class_names = list(test_dataset.class_mapping.keys())
    evaluator.generate_report(results, cm)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(cm, class_names)
    
    # Visualize predictions
    if args.visualize:
        # Flatten lists for visualization
        all_images = torch.cat(images, dim=0)
        all_masks = torch.cat(masks, dim=0)
        all_seg_preds = torch.cat(seg_preds, dim=0)
        evaluator.visualize_predictions(all_images, all_masks, all_seg_preds)
    
    print(f"\n✅ Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()