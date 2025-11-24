"""
Multi-Scale Feature Fusion (MSFF) with dedicated small-object detection head.

This module is designed as an add-on for the baseline YOLOv11-OBB head.
It enhances the feature pyramid without relying on the other custom modules
so it can be plugged directly into baseline experiments as the third innovation point.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSFFBlock(nn.Module):
    """
    Multi-Scale Feature Fusion block.

    Takes the baseline P3/P4/P5 features, performs lateral projection, multi-scale fusion,
    and produces an additional high-resolution P2 branch together with refined P3-P5.
    """

    def __init__(self, in_channels: List[int], mid_channels: int = 256):
        super().__init__()
        assert len(in_channels) == 3, "MSFFBlock expects three pyramid levels (P3, P4, P5)."

        self.mid_channels = mid_channels

        # lateral 1x1 projections to equalize channels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, mid_channels, kernel_size=1, bias=False) for ch in in_channels]
        )
        self.lateral_norms = nn.ModuleList(
            [nn.BatchNorm2d(mid_channels) for _ in in_channels]
        )

        # fuse projected features
        fusion_channels = mid_channels * len(in_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        # residual refinement
        self.residual_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        # downsample layers to rebuild lower-resolution levels
        self.downsample_p4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample_p5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False)

        # generate high-resolution P2 feature
        self.p2_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features

        target_size = p3.shape[-2:]
        proj_feats = []
        for feat, conv, bn in zip((p3, p4, p5), self.lateral_convs, self.lateral_norms):
            aligned = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            proj = conv(aligned)
            proj = bn(proj)
            proj_feats.append(proj)

        fused = torch.cat(proj_feats, dim=1)
        fused = self.fusion_conv(fused)

        fused = fused + self.residual_conv(fused)

        p2 = self.p2_proj(F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False))
        p3_out = fused
        p4_out = self.downsample_p4(p3_out)
        p5_out = self.downsample_p5(p4_out)

        return p2, p3_out, p4_out, p5_out


class SmallObjectHead(nn.Module):
    """
    Lightweight detection head that focuses on high-resolution features produced by MSFF.

    It mixes a local-detail branch with a context branch and can be attached to the Detect layer
    as an extra feature level.
    """

    def __init__(self, in_channels: int, head_channels: int = 128):
        super().__init__()
        inter_channels = max(head_channels, in_channels // 2)

        # local branch captures fine spatial cues
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, head_channels, kernel_size=3, padding=1, groups=head_channels, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )

        # context branch enlarges receptive field
        self.context_branch = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, head_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(head_channels * 2, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        fused = torch.cat([local_feat, context_feat], dim=1)
        gate = self.fusion_gate(fused)
        # gate is channel-wise, split to match local branch channels
        gated = local_feat * gate + context_feat * (1 - gate)
        return self.output_proj(gated)


class EnhancedFPNWithSmallHead(nn.Module):
    """
    Wrapper that bundles MSFF and the dedicated small-object head.

    Usage pattern inside a YOLO YAML:
      - feed the baseline P3/P4/P5 features to this module
      - receive four refined pyramid levels plus a high-resolution feature for Detect
    """

    def __init__(
        self,
        in_channels: List[int],
        mid_channels: int = 256,
        small_head_channels: int = 128,
    ):
        super().__init__()
        self.msff = MSFFBlock(in_channels, mid_channels=mid_channels)
        self.small_head = SmallObjectHead(mid_channels, head_channels=small_head_channels)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        p2, p3, p4, p5 = self.msff(features)
        p2_small = self.small_head(p2)
        # Return pyramid features plus the dedicated small-object map
        return [p2, p3, p4, p5], p2_small


