"""
UHDAM (Unified Hybrid Domain Attention Module)
Combines Spatial, Frequency (FFT-based), and Channel attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    Focuses on "where" the lesions are in the image
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Spatial attention weights (B, 1, H, W)
        """
        # Compute mean and max across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        return attention


class FrequencyAttention(nn.Module):
    """
    Frequency domain attention using FFT
    Captures texture patterns of lesions that are discriminative in frequency space
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.channels = channels
        
        # Learnable frequency filters
        self.freq_processor = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global frequency pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel-wise frequency attention
        self.freq_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def _extract_frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency domain features using FFT
        Returns both magnitude and phase information
        """
        # Apply FFT
        fft = torch.fft.rfft2(x, norm='ortho')
        
        # Get magnitude and phase
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Normalize magnitude
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
        
        # Combine magnitude and phase (as sin and cos for better learning)
        freq_features = torch.cat([
            magnitude,
            torch.sin(phase),
            torch.cos(phase)
        ], dim=1)
        
        # Convert back to spatial domain with learned weights
        freq_weighted = torch.fft.irfft2(
            magnitude * torch.exp(1j * phase),
            s=x.shape[-2:],
            norm='ortho'
        )
        
        return freq_weighted
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Frequency attention weights (B, C, 1, 1)
        """
        # Extract frequency features
        freq_features = self._extract_frequency_features(x)
        
        # Global pooling and MLP for channel attention
        pooled = self.global_pool(freq_features).squeeze(-1).squeeze(-1)
        attention = self.freq_mlp(pooled).unsqueeze(-1).unsqueeze(-1)
        
        return attention


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation)
    Focuses on "which" feature channels are important
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Channel attention weights (B, C, 1, 1)
        """
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out)
        
        # Combine
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return attention


class UHDAM(nn.Module):
    """
    Unified Hybrid Domain Attention Module
    Combines Spatial, Frequency, and Channel attention with learnable fusion
    """
    
    def __init__(
        self,
        channels: int,
        spatial_kernel_size: int = 7,
        reduction: int = 16,
        use_frequency: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.use_frequency = use_frequency
        
        # Initialize attention modules
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        self.channel_attention = ChannelAttention(channels, reduction)
        
        if use_frequency:
            self.frequency_attention = FrequencyAttention(channels, reduction)
        
        # Learnable fusion weights (alpha, beta, gamma)
        self.fusion_weights = nn.Parameter(torch.ones(3 if use_frequency else 2) / 3)
        
        # Attention refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            attended_features: Attention-weighted features (B, C, H, W)
            attention_maps: Dictionary containing individual attention maps
        """
        # Get individual attention maps
        spatial_attn = self.spatial_attention(x)  # (B, 1, H, W)
        channel_attn = self.channel_attention(x)  # (B, C, 1, 1)
        
        # Broadcast channel attention to spatial dimensions
        channel_attn_broadcast = channel_attn.expand_as(x)
        
        # Normalize fusion weights with softmax
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Apply spatial attention
        spatial_weighted = x * spatial_attn
        
        # Apply channel attention
        channel_weighted = x * channel_attn_broadcast
        
        if self.use_frequency:
            freq_attn = self.frequency_attention(x)  # (B, C, 1, 1)
            freq_attn_broadcast = freq_attn.expand_as(x)
            freq_weighted = x * freq_attn_broadcast
            
            # Combine all three with learned weights
            combined = (
                weights[0] * spatial_weighted +
                weights[1] * channel_weighted +
                weights[2] * freq_weighted
            )
            attention_maps = {
                'spatial': spatial_attn,
                'channel': channel_attn,
                'frequency': freq_attn
            }
        else:
            # Combine spatial and channel only
            combined = (
                weights[0] * spatial_weighted +
                weights[1] * channel_weighted
            )
            attention_maps = {
                'spatial': spatial_attn,
                'channel': channel_attn,
            }
        
        # Refine combined attention
        refined_attn = self.refine(combined)
        output = x * refined_attn
        
        return output, attention_maps


class MultiScaleUHDAM(nn.Module):
    """
    Multi-scale UHDAM for capturing lesions at different scales
    """
    
    def __init__(
        self,
        channels: int,
        num_scales: int = 3,
        reduction: int = 16,
        use_frequency: bool = True
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # UHDAM blocks at different scales
        self.uhdam_blocks = nn.ModuleList([
            UHDAM(channels, reduction=reduction, use_frequency=use_frequency)
            for _ in range(num_scales)
        ])
        
        # Scale-specific downsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Cross-scale fusion
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply UHDAM at multiple scales and fuse
        """
        multi_scale_outputs = []
        all_attention_maps = {}
        
        current = x
        for i, uhdam in enumerate(self.uhdam_blocks):
            # Apply UHDAM at current scale
            output, attn_maps = uhdam(current)
            multi_scale_outputs.append(output)
            
            # Store attention maps
            for k, v in attn_maps.items():
                all_attention_maps[f'{k}_scale_{i}'] = v
            
            # Downsample for next scale (except last)
            if i < self.num_scales - 1:
                current = self.downsample(current)
                # Also downsample output for alignment
                multi_scale_outputs[-1] = F.interpolate(
                    output, scale_factor=2**i, mode='bilinear', align_corners=False
                )
        
        # Resize all outputs to original size
        resized_outputs = []
        for i, out in enumerate(multi_scale_outputs):
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            resized_outputs.append(out)
        
        # Fuse multi-scale outputs
        fused = torch.cat(resized_outputs, dim=1)
        fused = self.cross_scale_fusion(fused)
        
        return fused, all_attention_maps


# Test the module
def test_uhdam():
    """Test UHDAM module"""
    print("Testing UHDAM module...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic UHDAM
    model = UHDAM(channels=256)
    model = model.to(device)
    
    x = torch.randn(2, 256, 64, 64).to(device)
    output, attention_maps = model(x)
    
    print(f"Basic UHDAM:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention maps: {list(attention_maps.keys())}")
    
    # Test MultiScaleUHDAM
    model_ms = MultiScaleUHDAM(channels=256, num_scales=3)
    model_ms = model_ms.to(device)
    
    output_ms, attn_maps_ms = model_ms(x)
    print(f"\nMultiScaleUHDAM:")
    print(f"  Output shape: {output_ms.shape}")
    print(f"  Attention maps: {list(attn_maps_ms.keys())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")
    
    print("\n✅ UHDAM test passed!")
    return model


if __name__ == "__main__":
    test_uhdam()