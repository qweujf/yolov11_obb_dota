"""
Multi-Scale Feature Fusion (MSFF) with dedicated small-object detection head.

This module is designed as an add-on for the baseline YOLOv11-OBB head.
It enhances the feature pyramid without relying on the other custom modules
so it can be plugged directly into baseline experiments as the third innovation point.
"""

from typing import List, Tuple, Union

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


class P2MSFFBranch(nn.Module):
    """
    Lightweight MSFF branch dedicated to P2.

    Takes P2 and P3 features, merges multi-scale context, and produces a small-object-aware P2 feature map.
    """

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        branch_channels: int = 128,
        out_channels: int = 96,
    ):
        super().__init__()
        self.branch_channels = branch_channels

        # lateral projections
        self.p2_proj = nn.Sequential(
            nn.Conv2d(p2_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )
        self.p3_proj = nn.Sequential(
            nn.Conv2d(p3_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )

        # multi-scale context (depthwise + dilated)
        self.context = nn.Sequential(
            nn.Conv2d(branch_channels * 2, branch_channels, kernel_size=3, padding=1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(branch_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("P2MSFFBranch forward expects a list/tuple of [P2, P3].")
        assert len(inputs) == 2, "P2MSFFBranch expects [P2, P3] inputs."
        p2, p3 = inputs

        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2_feat = self.p2_proj(p2)
        p3_feat = self.p3_proj(p3_up)
        fused = torch.cat([p2_feat, p3_feat], dim=1)
        fused = self.context(fused)
        return self.output(fused)


