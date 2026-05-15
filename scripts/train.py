#!/usr/bin/env python3
"""
Training script for HEMSAM
"""

import sys
import os
import argparse
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hemsam import HEMSAM
from data.dataset import create_dataloaders
from training.trainer import HEMSAMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train HEMSAM model')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/plantvillage/raw/plantvillage/PlantVillage',
                        help='Path to dataset')
    parser.add_argument('--sam_checkpoint', type=str, default='./weights/sam_vit_b.pth',
                        help='Path to SAM checkpoint')
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM encoder (requires checkpoint)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='hemsam',
                        help='WandB project name')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_path}")
        return config
    else:
        print(f"Config file not found: {config_path}, using defaults")
        return {}


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['use_wandb'] = args.use_wandb
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        generate_masks=True
    )
    
    # Initialize model
    print("\nInitializing HEMSAM model...")
    model = HEMSAM(
        sam_checkpoint=args.sam_checkpoint if args.use_sam else None,
        num_classes=15,
        use_sam=args.use_sam,
        use_cdms=True,
        use_uhdam=True
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = HEMSAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"hemsam_{config.get('num_epochs', 100)}epochs"
        )
        print("✓ WandB initialized")
    
    # Train
    history = trainer.train()
    
    print("\n✅ Training completed!")
    print(f"Best model saved to: ./checkpoints/best_model.pth")
    
    return history


if __name__ == "__main__":
    main()