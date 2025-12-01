"""
Adaptive Feature Alignment (AFA) module.

在 FPN 的 Concat 操作之前，对来自不同尺度的特征进行自适应对齐和增强。
针对小目标和旋转目标，在特征融合前先对齐空间信息和通道信息。

位置：FPN 内部（Concat 之前）
与 MSFF 的区别：MSFF 在 FPN 输出后，AFA 在 FPN 内部
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveFeatureAlignment(nn.Module):
    """
    自适应特征对齐模块。
    
    在 FPN 的 Concat 之前使用，对两个不同尺度的特征进行：
    1. 空间对齐（上采样/下采样）
    2. 通道对齐（1x1 conv）
    3. 特征增强（轻量注意力）
    """

    def __init__(self, in_channels_low: int, in_channels_high: int, out_channels: int = None):
        """
        Args:
            in_channels_low: 低分辨率特征（如 P4）的通道数
            in_channels_high: 高分辨率特征（如 P3）的通道数
            out_channels: 输出通道数（默认与 in_channels_low 相同）
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels_low

        # 通道对齐：将两个特征对齐到相同通道数
        self.align_low = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.align_high = nn.Sequential(
            nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # 轻量级特征增强：空间注意力 + 通道注意力
        self.spatial_att = SpatialAttention()
        self.channel_att = ChannelAttention(out_channels)

        # 特征融合：融合对齐后的特征
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features) -> torch.Tensor:
        """
        Args:
            features: List/Tuple of [feat_low, feat_high] tensors
                - feat_low: 低分辨率特征 [B, C_low, H_low, W_low]（如 P4）
                - feat_high: 高分辨率特征 [B, C_high, H_high, W_high]（如 P3）

        Returns:
            对齐并增强后的特征 [B, C_out, H_high, W_high]
        """
        if isinstance(features, (list, tuple)):
            feat_low, feat_high = features
        else:
            raise ValueError("AFA expects a list or tuple of [feat_low, feat_high]")
        
        # 1. 通道对齐
        feat_low_aligned = self.align_low(feat_low)
        feat_high_aligned = self.align_high(feat_high)

        # 2. 空间对齐：将低分辨率特征上采样到高分辨率
        if feat_low_aligned.shape[-2:] != feat_high_aligned.shape[-2:]:
            feat_low_aligned = F.interpolate(
                feat_low_aligned, 
                size=feat_high_aligned.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

        # 3. 特征增强：分别对两个特征应用注意力
        feat_low_enhanced = self.spatial_att(feat_low_aligned) * self.channel_att(feat_low_aligned)
        feat_high_enhanced = self.spatial_att(feat_high_aligned) * self.channel_att(feat_high_aligned)

        # 4. 特征融合
        fused = torch.cat([feat_low_enhanced, feat_high_enhanced], dim=1)
        output = self.fusion(fused)

        # 5. 残差连接（使用高分辨率特征作为基础）
        return feat_high_aligned + output


class SpatialAttention(nn.Module):
    """轻量级空间注意力模块"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """空间注意力：关注重要的空间位置"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(att)
        return x * att


class ChannelAttention(nn.Module):
    """轻量级通道注意力模块（SE-style）"""

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
        """通道注意力：关注重要的通道"""
        w = self.fc(self.pool(x))
        return x * w


# 别名
AFA = AdaptiveFeatureAlignment

