"""
Lightweight Multi-Scale Feature Fusion (MSFF) module.

This is a simplified version that performs feature fusion across P3/P4/P5
without adding extra detection heads. It enhances small object detection
while keeping computational cost manageable.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightMSFF(nn.Module):
    """
    Lightweight Multi-Scale Feature Fusion module.

    Takes P3/P4/P5 features, performs lightweight fusion, and returns
    enhanced P3'/P4'/P5' features for the original detection heads.

    This design:
    - Does NOT add new detection heads
    - Uses small channel dimensions (64) for fusion
    - Uses learnable weights for adaptive fusion
    - Applies residual connections to preserve original features
    """

    def __init__(self, in_channels: List[int], fusion_channels: int = 64):
        """
        Args:
            in_channels: List of input channel numbers for [P3, P4, P5]
            fusion_channels: Channel dimension for fusion (default 64, keep small)
        """
        super().__init__()
        assert len(in_channels) == 3, "LightMSFF expects three pyramid levels (P3, P4, P5)."

        self.in_channels = in_channels
        self.fusion_channels = fusion_channels

        # 1. Lateral 1x1 convs to project to fusion_channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fusion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.SiLU(inplace=True),
            )
            for ch in in_channels
        ])

        # 2. Learnable fusion weights (softmax normalized)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # 3. Fusion refinement conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True),
        )

        # 4. Output projections back to original channel dimensions
        self.output_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fusion_channels, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
            )
            for ch in in_channels
        ])

        # 5. Downsampling layers for P4 and P5 residual paths
        self.downsample_p4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downsample_p5 = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, features: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List/Tuple of [P3, P4, P5] tensors

        Returns:
            Tuple of enhanced (P3', P4', P5') tensors
        """
        if isinstance(features, (list, tuple)):
            p3, p4, p5 = features
        else:
            raise ValueError("LightMSFF expects a list or tuple of [P3, P4, P5] features")

        # Get target size (P3's spatial size)
        target_size = p3.shape[-2:]

        # 1. Project all features to fusion_channels and align to P3 size
        proj_feats = []
        for feat, lateral in zip((p3, p4, p5), self.lateral_convs):
            proj = lateral(feat)
            if proj.shape[-2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            proj_feats.append(proj)

        # 2. Weighted fusion with learnable weights
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * proj_feats[0] + weights[1] * proj_feats[1] + weights[2] * proj_feats[2]

        # 3. Refine fused features
        fused = self.fusion_conv(fused)

        # 4. Project back and add residual connections
        # P3: direct addition
        p3_enhanced = p3 + self.output_projs[0](fused)

        # P4: downsample fused then add
        fused_p4 = self.downsample_p4(fused)
        if fused_p4.shape[-2:] != p4.shape[-2:]:
            fused_p4 = F.interpolate(fused_p4, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p4_enhanced = p4 + self.output_projs[1](fused_p4)

        # P5: downsample fused then add
        fused_p5 = self.downsample_p5(fused)
        if fused_p5.shape[-2:] != p5.shape[-2:]:
            fused_p5 = F.interpolate(fused_p5, size=p5.shape[-2:], mode='bilinear', align_corners=False)
        p5_enhanced = p5 + self.output_projs[2](fused_p5)

        return p3_enhanced, p4_enhanced, p5_enhanced


# Keep old names for backward compatibility but mark as deprecated
MSFFBlock = LightMSFF
SmallObjectHead = None  # Removed - no longer needed
EnhancedFPNWithSmallHead = None  # Removed - no longer needed
P2MSFFBranch = None  # Removed - no longer needed
