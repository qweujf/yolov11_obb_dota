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

    def __init__(self, in_channels: List[int], fusion_channels: int = 128):
        """
        Args:
            in_channels: List of input channel numbers for [P3, P4, P5]
            fusion_channels: Channel dimension for fusion operations
        """
        super().__init__()
        assert len(in_channels) == 3, "MSFF expects three pyramid levels (P3, P4, P5)."

        self.in_channels = in_channels
        self.fusion_channels = fusion_channels
        c3, c4, c5 = in_channels

        # 1. Lateral projections to fusion_channels
        self.lateral_p3 = self._make_lateral(c3, fusion_channels)
        self.lateral_p4 = self._make_lateral(c4, fusion_channels)
        self.lateral_p5 = self._make_lateral(c5, fusion_channels)

        # 2. Top-down pathway (P5 -> P4 -> P3)
        self.td_p5_to_p4 = self._make_fusion_block(fusion_channels)
        self.td_p4_to_p3 = self._make_fusion_block(fusion_channels)

        # 3. Bottom-up pathway (P3 -> P4 -> P5)
        self.bu_p3_to_p4 = self._make_fusion_block(fusion_channels)
        self.bu_p4_to_p5 = self._make_fusion_block(fusion_channels)

        # 4. Channel attention for each scale
        self.ca_p3 = ChannelAttention(fusion_channels)
        self.ca_p4 = ChannelAttention(fusion_channels)
        self.ca_p5 = ChannelAttention(fusion_channels)

        # 5. Output projections back to original channels
        self.out_p3 = self._make_output(fusion_channels, c3)
        self.out_p4 = self._make_output(fusion_channels, c4)
        self.out_p5 = self._make_output(fusion_channels, c5)

    def _make_lateral(self, in_ch: int, out_ch: int) -> nn.Module:
        """1x1 conv for channel projection."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def _make_fusion_block(self, channels: int) -> nn.Module:
        """3x3 conv block for feature fusion."""
        return nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def _make_output(self, in_ch: int, out_ch: int) -> nn.Module:
        """Output projection with refinement."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, features: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List/Tuple of [P3, P4, P5] tensors

        Returns:
            Tuple of enhanced (P3', P4', P5') tensors
        """
        p3, p4, p5 = features

        # Lateral projections
        p3_lat = self.lateral_p3(p3)
        p4_lat = self.lateral_p4(p4)
        p5_lat = self.lateral_p5(p5)

        # Top-down pathway
        p5_up = F.interpolate(p5_lat, size=p4_lat.shape[-2:], mode='bilinear', align_corners=False)
        p4_td = self.td_p5_to_p4(torch.cat([p4_lat, p5_up], dim=1))

        p4_up = F.interpolate(p4_td, size=p3_lat.shape[-2:], mode='bilinear', align_corners=False)
        p3_td = self.td_p4_to_p3(torch.cat([p3_lat, p4_up], dim=1))

        # Bottom-up pathway
        p3_down = F.avg_pool2d(p3_td, kernel_size=2, stride=2)
        if p3_down.shape[-2:] != p4_td.shape[-2:]:
            p3_down = F.interpolate(p3_down, size=p4_td.shape[-2:], mode='bilinear', align_corners=False)
        p4_bu = self.bu_p3_to_p4(torch.cat([p4_td, p3_down], dim=1))

        p4_down = F.avg_pool2d(p4_bu, kernel_size=2, stride=2)
        if p4_down.shape[-2:] != p5_lat.shape[-2:]:
            p4_down = F.interpolate(p4_down, size=p5_lat.shape[-2:], mode='bilinear', align_corners=False)
        p5_bu = self.bu_p4_to_p5(torch.cat([p5_lat, p4_down], dim=1))

        # Channel attention
        p3_att = self.ca_p3(p3_td)
        p4_att = self.ca_p4(p4_bu)
        p5_att = self.ca_p5(p5_bu)

        # Output projection + residual
        p3_out = p3 + self.out_p3(p3_att)
        p4_out = p4 + self.out_p4(p4_att)
        p5_out = p5 + self.out_p5(p5_att)

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
