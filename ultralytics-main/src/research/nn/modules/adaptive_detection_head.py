"""
Adaptive Scale-Aware Detection Head (ASADH) module.

针对不同尺度特征（P3/P4/P5）使用不同的特征增强策略：
- P3（小目标）：密集感受野 + 细节增强
- P4（中目标）：标准感受野 + 上下文融合
- P5（大目标）：全局上下文 + 轻量注意力

这个模块在 OBB 检测头之前对特征进行自适应增强。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ScaleAwareEnhancement(nn.Module):
    """
    尺度感知特征增强模块。
    
    针对不同尺度使用不同的增强策略：
    - small_scale: 密集感受野（dilated conv）用于小目标
    - medium_scale: 标准卷积 + 上下文融合
    - large_scale: 全局池化 + 轻量注意力
    """

    def __init__(self, in_channels: int, scale_type: str = "medium"):
        """
        Args:
            in_channels: 输入通道数
            scale_type: 尺度类型 ("small", "medium", "large")
        """
        super().__init__()
        self.scale_type = scale_type
        mid_channels = in_channels // 2

        if scale_type == "small":
            # 小目标：密集感受野 + 细节增强
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(mid_channels * 3, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            )
        elif scale_type == "medium":
            # 中目标：标准卷积 + 上下文融合
            self.local_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.context_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(mid_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            )
        else:  # large
            # 大目标：全局上下文 + 轻量注意力
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.global_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                nn.Sigmoid(),
            )
            self.local_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.scale_type == "small":
            # 多分支密集感受野
            b1 = self.branch1(x)
            b2 = self.branch2(x)
            b3 = self.branch3(x)
            fused = torch.cat([b1, b2, b3], dim=1)
            return x + self.fusion(fused)
        elif self.scale_type == "medium":
            # 局部 + 全局上下文
            local = self.local_conv(x)
            context = self.context_conv(x)
            context = F.interpolate(context, size=x.shape[-2:], mode='nearest')
            fused = torch.cat([local, context], dim=1)
            return x + self.fusion(fused)
        else:  # large
            # 全局注意力加权
            global_weight = self.global_conv(self.global_pool(x))
            global_weight = F.interpolate(global_weight, size=x.shape[-2:], mode='nearest')
            local_feat = self.local_conv(x)
            return x * global_weight + local_feat


class AdaptiveScaleAwareHead(nn.Module):
    """
    自适应尺度感知检测头包装器。
    
    在 OBB 检测头之前，对每个尺度的特征进行自适应增强。
    """

    def __init__(self, in_channels: List[int]):
        """
        Args:
            in_channels: 每个尺度的输入通道数，通常为 [P3_ch, P4_ch, P5_ch]
        """
        super().__init__()
        assert len(in_channels) == 3, "需要三个尺度的特征（P3, P4, P5）"
        
        # 为每个尺度创建对应的增强模块
        self.enhance_p3 = ScaleAwareEnhancement(in_channels[0], scale_type="small")
        self.enhance_p4 = ScaleAwareEnhancement(in_channels[1], scale_type="medium")
        self.enhance_p5 = ScaleAwareEnhancement(in_channels[2], scale_type="large")

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of [P3, P4, P5] tensors

        Returns:
            List of enhanced [P3', P4', P5'] tensors
        """
        assert len(features) == 3, "需要三个尺度的特征"
        
        p3_enhanced = self.enhance_p3(features[0])
        p4_enhanced = self.enhance_p4(features[1])
        p5_enhanced = self.enhance_p5(features[2])
        
        return [p3_enhanced, p4_enhanced, p5_enhanced]


# 别名，方便在 YAML 中使用
ASADH = AdaptiveScaleAwareHead

