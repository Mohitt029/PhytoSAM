"""
SAM Encoder Integration for HEMSAM
Loads pretrained SAM ViT-B and extracts features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from pathlib import Path

# Try to import SAM
try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    print("Segment-Anything not installed. Run: pip install segment-anything")
    SAM_AVAILABLE = False


class SAMEncoder(nn.Module):
    """
    SAM ViT-B Encoder wrapper for HEMSAM
    Extracts multi-scale features from images
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        freeze_encoder: bool = True,
        extract_layers: List[int] = [6, 8, 10, 11]  # Multi-scale features
    ):
        super().__init__()
        
        assert SAM_AVAILABLE, "Segment-Anything is required"
        
        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = self.sam.image_encoder
        self.extract_layers = extract_layers
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # Get feature dimensions
        self.embed_dim = 256  # SAM ViT-B output channels
        
        # Store intermediate features
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        def hook_fn(module, input, output, layer_idx):
            self.features[f'layer_{layer_idx}'] = output
        
        # Register hooks for specific transformer layers
        for idx in self.extract_layers:
            if hasattr(self.image_encoder, 'blocks') and idx < len(self.image_encoder.blocks):
                self.image_encoder.blocks[idx].register_forward_hook(
                    lambda m, i, o, idx=idx: hook_fn(m, i, o, idx)
                )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract SAM features
        
        Args:
            x: Input image (B, 3, H, W) - values in [0, 1]
        
        Returns:
            Dictionary containing:
                - 'final': Final SAM features (B, 256, H/16, W/16)
                - 'layer_X': Intermediate features from specified layers
        """
        self.features = {}
        
        # SAM expects input in [0, 1] range
        # Preprocess: normalize to SAM's expected format
        x = x * 255  # SAM expects [0, 255]
        
        # Forward through SAM encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            final_features = self.image_encoder(x)
        
        # Add final features
        self.features['final'] = final_features
        
        return self.features
    
    @property
    def freeze_encoder(self):
        """Check if encoder is frozen"""
        return not any(p.requires_grad for p in self.image_encoder.parameters())


class SAMFeatureAdapter(nn.Module):
    """
    Adapter to convert SAM features to HEMSAM format
    Projects SAM features to desired channel dimensions
    """
    
    def __init__(self, sam_channels: int = 256, hemsam_channels: int = 256):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Conv2d(sam_channels, hemsam_channels, kernel_size=1),
            nn.BatchNorm2d(hemsam_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sam_features: torch.Tensor) -> torch.Tensor:
        """
        Convert SAM features to HEMSAM format
        
        Args:
            sam_features: (B, 256, H/16, W/16)
        
        Returns:
            Adapted features (B, hemsam_channels, H/16, W/16)
        """
        return self.adapter(sam_features)


def test_sam_encoder():
    """Test SAM encoder if available"""
    print("Testing SAM Encoder...")
    
    if not SAM_AVAILABLE:
        print("⚠️ SAM not installed. Please run: pip install segment-anything")
        print("Then download weights from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None
    
    # Check for weights
    weights_path = Path("./weights/sam_vit_b.pth")
    if not weights_path.exists():
        print(f"⚠️ SAM weights not found at {weights_path}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None
    
    try:
        encoder = SAMEncoder(str(weights_path), freeze_encoder=True)
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = encoder(x)
        
        print(f"✓ SAM Encoder loaded successfully")
        print(f"  Final features shape: {features['final'].shape}")
        print(f"  Extracted layers: {[k for k in features.keys() if k != 'final']}")
        
        return encoder
        
    except Exception as e:
        print(f"❌ SAM Encoder error: {e}")
        return None


if __name__ == "__main__":
    test_sam_encoder()