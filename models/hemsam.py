"""
HEMSAM: Hybrid Enhanced Multi-Scale SAM
Complete model assembly for plant disease segmentation and classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from pathlib import Path

# Import HEMSAM components
from models.cdms_rfb import CDMS_RFB, MultiScaleRFBSimple, EdgeAttentionModule
from models.uhd_attention import UHDAM
from models.prompt_generator import SupervisedPromptGenerator
from models.classification_head import ClassificationHead, MultiScaleClassificationHead

# Try to import SAM
try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: Segment-Anything not installed. SAM features will be disabled.")


class HEMSAM(nn.Module):
    """
    Complete HEMSAM model for plant disease segmentation and classification
    
    Architecture:
    1. SAM ViT-B Encoder (pretrained, frozen)
    2. CDMS RFB (multi-scale dilated convs + edge attention)
    3. UHDAM (unified hybrid domain attention)
    4. Prompt Generator (for SAM decoder)
    5. SAM Mask Decoder
    6. Classification Head
    """
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        num_classes: int = 15,  # Change to 26 for full dataset
        feature_dim: int = 256,
        use_sam: bool = True,
        use_cdms: bool = True,
        use_uhdam: bool = True,
        freeze_sam_encoder: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_sam = use_sam and SAM_AVAILABLE and sam_checkpoint is not None
        self.use_cdms = use_cdms
        self.use_uhdam = use_uhdam
        
        # 1. SAM Encoder
        if self.use_sam:
            self._init_sam_encoder(sam_checkpoint, freeze_sam_encoder)
        else:
            # Fallback CNN encoder if SAM not available
            self._init_fallback_encoder()
        
        # 2. CDMS RFB Module (edge-aware multi-scale features)
        if self.use_cdms:
            self.cdms = MultiScaleRFBSimple(
                in_channels=feature_dim,
                out_channels=feature_dim,
                use_edge_attention=True
            )
        
        # 3. UHDAM Module (frequency + spatial + channel attention)
        if self.use_uhdam:
            self.uhdam = UHDAM(
                channels=feature_dim,
                use_frequency=True
            )
        
        # 4. Prompt Generator
        self.prompt_generator = SupervisedPromptGenerator(
            feature_dim=feature_dim,
            num_points=10,
            use_learnable_mapping=True
        )
        
        # 5. SAM Mask Decoder (if SAM available)
        if self.use_sam:
            self.mask_decoder = self.sam.mask_decoder
            # Freeze mask decoder initially
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
        
        # 6. Classification Head
        self.classifier = ClassificationHead(
            in_channels=feature_dim,
            num_classes=num_classes,
            dropout=0.3
        )
        
        # Final mask refinement
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        print(f"\n{'='*50}")
        print("HEMSAM Model Initialized")
        print(f"{'='*50}")
        print(f"  SAM Encoder: {'Enabled' if self.use_sam else 'Disabled (Fallback)'}")
        print(f"  CDMS RFB: {'Enabled' if self.use_cdms else 'Disabled'}")
        print(f"  UHDAM: {'Enabled' if self.use_uhdam else 'Disabled'}")
        print(f"  Number of classes: {num_classes}")
        print(f"{'='*50}\n")
    
    def _init_sam_encoder(self, checkpoint: str, freeze: bool):
        """Initialize SAM encoder"""
        try:
            self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            self.sam_encoder = self.sam.image_encoder
            
            if freeze:
                for param in self.sam_encoder.parameters():
                    param.requires_grad = False
            
            # Adapter to match feature dimensions
            self.sam_adapter = nn.Conv2d(256, self.feature_dim, kernel_size=1)
            
            print(f"✓ SAM Encoder loaded from {checkpoint}")
            
        except Exception as e:
            print(f"⚠️ SAM loading failed: {e}")
            print("Falling back to CNN encoder")
            self._init_fallback_encoder()
            self.use_sam = False
    
    def _init_fallback_encoder(self):
        """Fallback CNN encoder when SAM is not available"""
        self.fallback_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU()
        )
        print("✓ Fallback CNN encoder initialized")
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to features
        
        Args:
            image: Input image (B, 3, H, W)
        
        Returns:
            Features (B, feature_dim, H/16, W/16)
        """
        if self.use_sam:
            # SAM expects [0, 255] range
            if image.max() <= 1.0:
                image_255 = image * 255
            else:
                image_255 = image
            
            features = self.sam_encoder(image_255)
            features = self.sam_adapter(features)
        else:
            features = self.fallback_encoder(image)
        
        return features
    
    def enhance_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply CDMS and UHDAM enhancements
        
        Args:
            features: Input features (B, C, H, W)
        
        Returns:
            enhanced_features: Enhanced features
            attention_maps: Dictionary of attention maps for visualization
        """
        attention_maps = {}
        
        # Apply CDMS RFB
        if self.use_cdms:
            features, edge_attn = self.cdms(features)
            if edge_attn is not None:
                attention_maps['edge_attention'] = edge_attn
        
        # Apply UHDAM
        if self.use_uhdam:
            features, uhdam_attn = self.uhdam(features)
            attention_maps.update(uhdam_attn)
        
        return features, attention_maps
    
    def generate_prompts(
        self,
        mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        use_gt: bool = True
    ) -> Dict:
        """
        Generate SAM prompts
        
        Args:
            mask: Ground truth mask (for training)
            features: Image features (for inference)
            use_gt: Use GT mask (True) or predict (False)
        
        Returns:
            Dictionary with 'boxes', 'points', 'point_labels'
        """
        return self.prompt_generator(mask, features, use_gt=use_gt)
    
    def decode_mask(
        self,
        image_embeddings: torch.Tensor,
        prompts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode mask using SAM decoder
        
        Args:
            image_embeddings: SAM image embeddings
            prompts: Box and point prompts
        
        Returns:
            Predicted mask (B, 1, H, W)
        """
        if self.use_sam and hasattr(self, 'mask_decoder'):
            # Prepare prompts for SAM
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(prompts.get('points'), prompts.get('point_labels')) if 'points' in prompts else None,
                boxes=prompts.get('boxes'),
                masks=None
            )
            
            # Decode mask
            masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            return masks
        
        # Fallback: simple segmentation from features
        else:
            # Simple decoder using features
            mask = torch.sigmoid(torch.mean(features, dim=1, keepdim=True))
            return mask
    
    def forward(
        self,
        image: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            image: Input image (B, 3, H, W)
            gt_mask: Ground truth mask (optional, for training)
            return_attention: Return attention maps for visualization
        
        Returns:
            Dictionary with:
                - 'segmentation': Predicted mask (B, 1, H, W)
                - 'classification': Class logits (B, num_classes)
                - 'attention_maps': Attention maps (if return_attention)
        """
        # 1. Encode image
        features = self.encode(image)
        
        # 2. Enhance features with CDMS and UHDAM
        enhanced_features, attention_maps = self.enhance_features(features)
        
        # 3. Generate prompts
        if self.training and gt_mask is not None:
            prompts = self.generate_prompts(mask=gt_mask, features=enhanced_features, use_gt=True)
        else:
            prompts = self.generate_prompts(mask=gt_mask, features=enhanced_features, use_gt=False)
        
        # 4. Decode mask
        if self.use_sam:
            segmentation = self.decode_mask(features, prompts)
        else:
            # Simple upsampling for fallback
            segmentation = F.interpolate(
                enhanced_features[:, :1, :, :],
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            segmentation = torch.sigmoid(segmentation)
        
        # Refine mask
        segmentation = self.mask_refiner(segmentation)
        
        # 5. Classify
        classification = self.classifier(enhanced_features, segmentation.detach())
        
        output = {
            'segmentation': segmentation,
            'classification': classification,
        }
        
        if return_attention:
            output['attention_maps'] = attention_maps
        
        return output


def test_hemsam():
    """Test complete HEMSAM model"""
    print("\n" + "="*60)
    print("Testing Complete HEMSAM Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = HEMSAM(
        sam_checkpoint=None,  # Set to weights path if available
        num_classes=15,
        use_sam=False,  # Use fallback for testing
        use_cdms=True,
        use_uhdam=True
    ).to(device)
    
    # Test forward pass
    image = torch.randn(2, 3, 224, 224).to(device)
    gt_mask = (torch.rand(2, 1, 224, 224) > 0.5).float().to(device)
    
    model.train()
    output = model(image, gt_mask)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Segmentation shape: {output['segmentation'].shape}")
    print(f"  Classification shape: {output['classification'].shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        output_inf = model(image, return_attention=True)
    
    print(f"\n✓ Inference mode")
    print(f"  Segmentation shape: {output_inf['segmentation'].shape}")
    print(f"  Attention maps: {list(output_inf.get('attention_maps', {}).keys())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model Statistics")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✅ HEMSAM test passed!")
    return model


if __name__ == "__main__":
    test_hemsam()