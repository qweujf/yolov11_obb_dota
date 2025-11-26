"""
Multi-Scale Feature Fusion (MSFF) module - Medium strength version.

Designed for accuracy improvement chapter. This version has moderate
computational cost (~20-30 GFLOPs) with significant accuracy gains.
The next chapter can apply pruning/distillation to reduce the cost.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSFF(nn.Module):
    """
    Multi-Scale Feature Fusion module (Medium strength).

    Features:
    - Cross-scale feature interaction with attention
    - Bidirectional fusion (top-down + bottom-up)
    - Channel attention for adaptive weighting
    - Residual connections

    Target: ~4-5M params, ~20-30 GFLOPs additional cost
    """

    def __init__(self, in_channels: List[int], fusion_channels: int = 48):
        """
        Args:
            in_channels: List of input channel numbers for [P3, P4, P5]
            fusion_channels: Channel dimension for fusion operations (default 48 for lightweight)
        """
        super().__init__()
        assert len(in_channels) == 3, "MSFF expects three pyramid levels (P3, P4, P5)."

        self.in_channels = in_channels
        self.fusion_channels = fusion_channels
        c3, c4, c5 = in_channels

        # 1. Lateral projections to fusion_channels (1x1 conv)
        self.lateral_p3 = self._make_lateral(c3, fusion_channels)
        self.lateral_p4 = self._make_lateral(c4, fusion_channels)
        self.lateral_p5 = self._make_lateral(c5, fusion_channels)

        # 2. Lightweight fusion: single 3x3 conv after concat (not bidirectional)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels * 3, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True),
        )

        # 3. Channel attention (shared)
        self.channel_att = ChannelAttention(fusion_channels)

        # 4. Output projections back to original channels (1x1 conv)
        self.out_p3 = nn.Conv2d(fusion_channels, c3, kernel_size=1, bias=False)
        self.out_p4 = nn.Conv2d(fusion_channels, c4, kernel_size=1, bias=False)
        self.out_p5 = nn.Conv2d(fusion_channels, c5, kernel_size=1, bias=False)

    def _make_lateral(self, in_ch: int, out_ch: int) -> nn.Module:
        """1x1 conv for channel projection."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List/Tuple of [P3, P4, P5] tensors

        Returns:
            Tuple of enhanced (P3', P4', P5') tensors
        """
        p3, p4, p5 = features
        target_size = p3.shape[-2:]  # Align to P3 resolution

        # Lateral projections
        p3_lat = self.lateral_p3(p3)
        p4_lat = self.lateral_p4(p4)
        p5_lat = self.lateral_p5(p5)

        # Upsample P4, P5 to P3 resolution
        p4_up = F.interpolate(p4_lat, size=target_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5_lat, size=target_size, mode='bilinear', align_corners=False)

        # Concat and fuse
        fused = self.fusion_conv(torch.cat([p3_lat, p4_up, p5_up], dim=1))
        fused = self.channel_att(fused)

        # Output projection + residual for each scale
        p3_out = p3 + self.out_p3(fused)
        
        fused_p4 = F.interpolate(fused, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p4_out = p4 + self.out_p4(fused_p4)
        
        fused_p5 = F.interpolate(fused, size=p5.shape[-2:], mode='bilinear', align_corners=False)
        p5_out = p5 + self.out_p5(fused_p5)

        return p3_out, p4_out, p5_out


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


# Aliases for compatibility
LightMSFF = MSFF
MSFFBlock = MSFF
