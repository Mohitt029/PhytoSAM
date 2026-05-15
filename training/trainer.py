"""
HEMSAM Trainer
Complete training loop with validation, checkpointing, and logging
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from losses.combined_loss import HEMSAMLoss, MultiTaskLoss
from training.metrics import HEMSAMMetrics
from models.hemsam import HEMSAM


class HEMSAMTrainer:
    """
    Trainer for HEMSAM model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        self.log_interval = config.get('log_interval', 10)
        self.eval_interval = config.get('eval_interval', 1)
        
        # Initialize loss
        num_classes = config.get('num_classes', 15)
        use_multi_task = config.get('use_multi_task_loss', False)
        
        if use_multi_task:
            self.criterion = MultiTaskLoss()
        else:
            # Compute class weights if provided
            class_counts = config.get('class_counts', None)
            if class_counts:
                class_weights = torch.tensor(class_counts, dtype=torch.float32)
                class_weights = class_weights.sum() / (class_weights * num_classes)
                class_weights = class_weights / class_weights.sum()
            else:
                class_weights = None
            
            self.criterion = HEMSAMLoss(
                lambda_seg=config.get('lambda_seg', 1.0),
                lambda_cls=config.get('lambda_cls', 0.5),
                dice_weight=config.get('dice_weight', 0.5),
                focal_weight=config.get('focal_weight', 0.5),
                focal_alpha=config.get('focal_alpha', 0.25),
                focal_gamma=config.get('focal_gamma', 2.0),
                class_weights=class_weights
            )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics = HEMSAMMetrics(num_classes=num_classes)
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_accuracy': []
        }
        
        print(f"\n{'='*50}")
        print("HEMSAM Trainer Initialized")
        print(f"{'='*50}")
        print(f"  Device: {device}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Checkpoint Dir: {self.checkpoint_dir}")
        print(f"{'='*50}\n")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups"""
        # Different learning rates for different parts
        param_groups = [
            {
                'params': self.model.encoder.parameters(),
                'lr': self.learning_rate * 0.1  # Lower LR for encoder
            },
            {
                'params': self.model.cdms.parameters(),
                'lr': self.learning_rate
            },
            {
                'params': self.model.uhdam.parameters(),
                'lr': self.learning_rate
            },
            {
                'params': self.model.classifier.parameters(),
                'lr': self.learning_rate * 2  # Higher LR for classifier
            }
        ]
        
        return AdamW(param_groups, weight_decay=self.weight_decay)
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, masks)  # Use GT mask for training
            losses = self.criterion(
                outputs['segmentation'],
                masks,
                outputs['classification'],
                labels
            )
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'dice': f"{losses['dice_loss'].item():.4f}",
                'cls': f"{losses['cls_loss'].item():.4f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.log_interval == 0:
                wandb.log({
                    'train/batch_loss': losses['total_loss'].item(),
                    'train/batch_dice': losses['dice_loss'].item(),
                    'train/batch_focal': losses['focal_loss'].item(),
                    'train/batch_cls': losses['cls_loss'].item()
                })
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass (no GT mask for inference)
            outputs = self.model(images, return_attention=False)
            
            # Compute loss
            losses = self.criterion(
                outputs['segmentation'],
                masks,
                outputs['classification'],
                labels
            )
            
            total_loss += losses['total_loss'].item()
            
            # Update metrics
            self.metrics.update(
                outputs['segmentation'],
                masks,
                outputs['classification'],
                labels
            )
            
            progress_bar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_metric': self.best_metric
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 30)
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if epoch % self.eval_interval == 0:
                val_metrics = self.validate()
                
                for key, value in val_metrics.items():
                    if 'loss' in key:
                        self.history['val_loss'].append(value)
                    elif 'dice' in key.lower():
                        self.history['val_dice'].append(value)
                    elif 'iou' in key.lower():
                        self.history['val_iou'].append(value)
                    elif 'accuracy' in key.lower():
                        self.history['val_accuracy'].append(value)
                
                # Print metrics
                print(f"\nValidation Results:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  Mean IoU: {val_metrics.get('mean_iou', 0):.4f}")
                print(f"  Mean Dice: {val_metrics.get('mean_dice', 0):.4f}")
                print(f"  Accuracy: {val_metrics.get('accuracy', 0):.4f}")
                
                # Check for improvement
                current_metric = val_metrics.get('mean_dice', val_metrics.get('accuracy', 0))
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    print(f"  ✓ New best model! {current_metric:.4f}")
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train/loss': train_metrics['loss'],
                        'val/loss': val_metrics['loss'],
                        'val/mean_iou': val_metrics.get('mean_iou', 0),
                        'val/mean_dice': val_metrics.get('mean_dice', 0),
                        'val/accuracy': val_metrics.get('accuracy', 0),
                        'val/precision': val_metrics.get('mean_precision', 0),
                        'val/recall': val_metrics.get('mean_recall', 0),
                        'val/f1': val_metrics.get('mean_f1', 0),
                        'best_metric': self.best_metric,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # Early stopping
                early_stop_patience = self.config.get('early_stop_patience', 20)
                if self.patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('mean_dice', 0))
                else:
                    self.scheduler.step()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best metric: {self.best_metric:.4f}")
        print(f"Best model saved to: {self.checkpoint_dir}/best_model.pth")
        print("="*50)
        
        return self.history


def test_trainer():
    """Test trainer initialization"""
    print("Testing Trainer...")
    
    # Create dummy model
    from models.hemsam import HEMSAM
    model = HEMSAM(num_classes=15, use_sam=False)
    
    # Create dummy dataloaders
    from torch.utils.data import TensorDataset, DataLoader
    dummy_images = torch.randn(8, 3, 224, 224)
    dummy_masks = torch.randint(0, 2, (8, 1, 224, 224)).float()
    dummy_labels = torch.randint(0, 15, (8,))
    
    dummy_dataset = TensorDataset(dummy_images, dummy_masks, dummy_labels)
    
    # Wrap to match dataset output format
    class DummyDataset:
        def __init__(self, images, masks, labels):
            self.images = images
            self.masks = masks
            self.labels = labels
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return {
                'image': self.images[idx],
                'mask': self.masks[idx],
                'label': self.labels[idx]
            }
    
    dummy_dataset = DummyDataset(dummy_images, dummy_masks, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=4)
    
    config = {
        'num_epochs': 1,
        'learning_rate': 1e-4,
        'num_classes': 15
    }
    
    trainer = HEMSAMTrainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=config,
        device=torch.device('cpu')
    )
    
    print("✓ Trainer initialized successfully")
    print("✅ Trainer test passed!")
    
    return trainer


if __name__ == "__main__":
    test_trainer()