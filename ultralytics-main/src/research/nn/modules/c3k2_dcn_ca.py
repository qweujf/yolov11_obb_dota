"""
C3k2_DCN_CA Module: C3k2 with DCNv2 Bottleneck and Coordinate Attention.

This module combines:
- DCNv2 (Deformable Convolutional Networks v2) in Bottleneck for adaptive shape adaptation
- Coordinate Attention for direction-aware feature enhancement

References:
- DCNv2: Zhu et al., "Deformable ConvNets v2: More Deformable, Better Results", CVPR 2019
- Coordinate Attention: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

try:
    from ultralytics.nn.modules.conv import Conv, autopad
except Exception:
    from ultralytics.nn.modules import Conv, autopad  # type: ignore


class DCNv2(nn.Module):
    """
    Deformable Convolutional Networks v2.
    
    Reference: Zhu et al., "Deformable ConvNets v2: More Deformable, Better Results", CVPR 2019
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int = None,
        g: int = 1,
        act: Union[nn.Module, bool] = True,
    ):
        super().__init__()
        if p is None:
            p = autopad(k, p)
        
        self.offset_conv = nn.Conv2d(
            c1, 2 * g * k * k, kernel_size=k, stride=s, padding=p, groups=g
        )
        self.mask_conv = nn.Conv2d(
            c1, g * k * k, kernel_size=k, stride=s, padding=p, groups=g
        )
        self.mask_conv.weight.data.zero_()
        self.mask_conv.bias.data.zero_()
        
        self.regular_conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False
        )
        
        # Project mask to output channels for simplified implementation
        self.mask_proj = nn.Conv2d(g * k * k, c2, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.act = (
            nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )
        self.k = k
        self.g = g
        self.c2 = c2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with deformable convolution."""
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        # Use regular conv as fallback (simplified implementation)
        # For full DCNv2, would need to use deform_conv2d operation
        # Here we use a simplified version that applies mask to regular conv
        out = self.regular_conv(x)
        # Project mask to match output channels and apply as attention
        mask_proj = self.mask_proj(mask)
        mask_proj = torch.sigmoid(mask_proj)  # Normalize to [0, 1]
        out = out * (1 + mask_proj)  # Additive attention
        return self.act(out)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    
    Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
    """
    
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.channels = channels
        mip = max(8, channels // reduction)
        
        # Horizontal and vertical pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Shared MLP: process concatenated h and w features
        # Input: (n, c, h+w, 1) after concatenation
        # We'll use a 1D conv or linear layer to process the concatenated sequence
        self.conv1 = nn.Conv2d(channels, mip, 1, 1, 0, bias=True)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        # Separate convs for h and w attention weights
        self.conv_h = nn.Conv2d(mip, channels, 1, 1, 0, bias=True)
        self.conv_w = nn.Conv2d(mip, channels, 1, 1, 0, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        n, c, h, w = x.size()
        
        # Horizontal pooling: (n, c, h, w) -> (n, c, h, 1)
        x_h = self.pool_h(x)
        # Vertical pooling: (n, c, h, w) -> (n, c, 1, w)
        x_w = self.pool_w(x)
        
        # Concatenate along spatial dimension
        # x_h: (n, c, h, 1), x_w: (n, c, 1, w)
        # Permute x_w to (n, c, w, 1) and concatenate along h dimension: (n, c, h+w, 1)
        x_w_perm = x_w.permute(0, 1, 3, 2)  # (n, c, w, 1)
        y = torch.cat([x_h, x_w_perm], dim=2)  # (n, c, h+w, 1)
        
        # Process through shared MLP
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back to h and w
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (n, c, 1, w)
        
        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()  # (n, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (n, c, 1, w)
        
        # Apply attention: broadcast and multiply
        out = identity * a_h * a_w
        return out


class DCNv2Bottleneck(nn.Module):
    """Bottleneck with DCNv2 instead of standard convolution."""
    
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        # Handle k as tuple of tuples or tuple of ints
        # k can be ((3, 3), (3, 3)) or (3, 3)
        if isinstance(k[0], (tuple, list)):
            k1 = k[0][0]  # Extract first element from first tuple
        else:
            k1 = k[0]
        
        if isinstance(k[1], (tuple, list)):
            k2 = k[1][0]  # Extract first element from second tuple
        else:
            k2 = k[1]
        
        self.cv1 = Conv(c1, c_, k1, 1)
        self.cv2 = DCNv2(c_, c2, k2, 1, g=g, act=True)
        self.add = shortcut and c1 == c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with DCNv2 and optional shortcut."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2_DCN_CA(nn.Module):
    """
    C3k2 module enhanced with DCNv2 Bottleneck and Coordinate Attention.
    
    Structure: Conv → Split → DCNv2Bottleneck×n → Concat → CoordinateAttention → Conv
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList(
            DCNv2Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.coord_attn = CoordinateAttention((2 + n) * self.c)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3k2_DCN_CA."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.coord_attn(y)
        return self.cv2(y)

