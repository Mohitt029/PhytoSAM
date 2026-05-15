"""
CDMS RFB (Context-aware Dilated Multi-Scale Receptive Field Block)
Fully working version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DilatedConvBlock(nn.Module):
    """Multi-dilation convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EdgeAttentionModule(nn.Module):
    """
    Edge attention module using Sobel edge detection
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Sobel filters
        self.register_buffer(
            'sobel_x', 
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        
        # Edge feature extractor
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edge maps using Sobel operator"""
        gray = x.mean(dim=1, keepdim=True)
        
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # Normalize
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        
        return edge_mag
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_map = self._extract_edges(x)
        edge_attention = self.edge_conv(edge_map)
        return edge_attention


class CDMS_RFB(nn.Module):
    """
    Context-aware Dilated Multi-Scale Receptive Field Block
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        dilations: Tuple[int, ...] = (1, 2, 3, 4),
        use_edge_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edge_attention = use_edge_attention
        
        # Reduce channels
        intermediate_channels = max(out_channels // 4, 32)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-dilation branches
        self.dilated_branches = nn.ModuleList([
            DilatedConvBlock(intermediate_channels, intermediate_channels, d)
            for d in dilations
        ])
        
        # Branch fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(intermediate_channels * len(dilations), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Edge attention
        if use_edge_attention:
            self.edge_attention = EdgeAttentionModule(out_channels)
        
        # Residual
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Residual path
        identity = self.residual_bn(self.residual(x))
        
        # Main path
        x_reduced = self.reduce(x)
        
        branch_outputs = []
        for branch in self.dilated_branches:
            branch_outputs.append(branch(x_reduced))
        
        x_concat = torch.cat(branch_outputs, dim=1)
        x_fused = self.fusion(x_concat)
        
        # Apply edge attention
        edge_weight = None
        if self.use_edge_attention:
            edge_weight = self.edge_attention(x_fused)
            x_fused = x_fused * edge_weight
        
        # Combine
        output = self.relu(x_fused + identity)
        
        return output, edge_weight


class MultiScaleRFBSimple(nn.Module):
    """
    Multi-scale RFB with proper channel handling
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_edge_attention: bool = True
    ):
        super().__init__()
        
        # Calculate per-path channels (ensure divisibility)
        per_path_channels = out_channels // 3
        remainder = out_channels - (per_path_channels * 3)
        
        # Distribute remainder to first path
        self.path1_out = per_path_channels + remainder
        self.path2_out = per_path_channels
        self.path3_out = per_path_channels
        
        # Three parallel paths with different receptive fields
        self.path1 = CDMS_RFB(in_channels, self.path1_out, dilations=(1, 2), use_edge_attention=False)
        self.path2 = CDMS_RFB(in_channels, self.path2_out, dilations=(3, 4), use_edge_attention=False)
        self.path3 = CDMS_RFB(in_channels, self.path3_out, dilations=(5, 6), use_edge_attention=False)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.edge_attention = EdgeAttentionModule(out_channels) if use_edge_attention else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Parallel paths
        out1, _ = self.path1(x)
        out2, _ = self.path2(x)
        out3, _ = self.path3(x)
        
        # Concatenate (channels: path1_out + path2_out + path3_out = out_channels)
        x_fused = torch.cat([out1, out2, out3], dim=1)
        x_fused = self.fusion(x_fused)
        
        # Edge attention
        edge_weight = None
        if self.edge_attention is not None:
            edge_weight = self.edge_attention(x_fused)
            x_fused = x_fused * edge_weight
        
        output = self.relu(x_fused)
        
        return output, edge_weight


def test_cdms_rfb():
    """Test CDMS RFB module"""
    print("Testing CDMS RFB module...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic CDMS_RFB
    model = CDMS_RFB(in_channels=256, out_channels=256)
    model = model.to(device)
    
    x = torch.randn(2, 256, 64, 64).to(device)
    output, edge_attn = model(x)
    
    print(f"Basic CDMS_RFB:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    if edge_attn is not None:
        print(f"  Edge attention shape: {edge_attn.shape}")
    
    # Test MultiScaleRFBSimple
    model_ms = MultiScaleRFBSimple(in_channels=256, out_channels=256)
    model_ms = model_ms.to(device)
    
    output_ms, edge_attn_ms = model_ms(x)
    print(f"\nMultiScaleRFBSimple:")
    print(f"  Output shape: {output_ms.shape}")
    if edge_attn_ms is not None:
        print(f"  Edge attention shape: {edge_attn_ms.shape}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nBasic CDMS_RFB parameters: {total_params:,}")
    
    total_params_ms = sum(p.numel() for p in model_ms.parameters())
    print(f"MultiScaleRFBSimple parameters: {total_params_ms:,}")
    
    # Verify channel consistency
    print(f"\nChannel verification:")
    print(f"  Input channels: {x.shape[1]}")
    print(f"  Output channels: {output.shape[1]}")
    print(f"  Multi-scale output channels: {output_ms.shape[1]}")
    
    print("\n✅ CDMS RFB test passed!")
    return model


if __name__ == "__main__":
    test_cdms_rfb()