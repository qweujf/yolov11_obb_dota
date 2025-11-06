# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Adaptive Feature Pyramid Network (AFPN) for YOLOv11-OBB
自适应特征金字塔网络，专为旋转目标检测优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .conv import Conv, autopad
from .mca_attention import MCAttention, SCAttention


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块
    根据目标尺度自适应选择特征层进行融合
    """
    
    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        super().__init__()
        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        
        # 尺度感知权重生成
        self.scale_aware_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 + c2 + c3, (c1 + c2 + c3) // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((c1 + c2 + c3) // 4, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征对齐
        self.align_conv1 = nn.Conv2d(c1, c4, 1)
        self.align_conv2 = nn.Conv2d(c2, c4, 1)
        self.align_conv3 = nn.Conv2d(c3, c4, 1)
        
        # 多尺度空洞卷积
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(c4, c4, 3, padding=1, dilation=1),
            nn.Conv2d(c4, c4, 3, padding=2, dilation=2),
            nn.Conv2d(c4, c4, 3, padding=4, dilation=4),
        ])
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(c4 * 3, c4, 1)
        
        # 注意力机制
        self.attention = MCAttention(c4, c4)
        
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        """
        自适应特征融合
        
        Args:
            p3: P3层特征 [B, C1, H3, W3]
            p4: P4层特征 [B, C2, H4, W4]  
            p5: P5层特征 [B, C3, H5, W5]
            
        Returns:
            torch.Tensor: 融合后的特征 [B, C4, H3, W3]
        """
        B, _, H3, W3 = p3.shape
        
        # 特征对齐到P3尺寸
        p4_aligned = F.interpolate(p4, size=(H3, W3), mode='bilinear', align_corners=False)
        p5_aligned = F.interpolate(p5, size=(H3, W3), mode='bilinear', align_corners=False)
        
        # 通道对齐
        p3_aligned = self.align_conv1(p3)
        p4_aligned = self.align_conv2(p4_aligned)
        p5_aligned = self.align_conv3(p5_aligned)
        
        # 尺度感知权重计算
        concat_feat = torch.cat([p3_aligned, p4_aligned, p5_aligned], dim=1)
        scale_weights = self.scale_aware_weight(concat_feat)  # [B, 3, 1, 1]
        
        # 加权特征
        weighted_p3 = p3_aligned * scale_weights[:, 0:1, :, :]
        weighted_p4 = p4_aligned * scale_weights[:, 1:2, :, :]
        weighted_p5 = p5_aligned * scale_weights[:, 2:3, :, :]
        
        # 多尺度空洞卷积
        multi_scale_features = []
        for conv in self.dilated_convs:
            multi_scale_features.append(conv(weighted_p3 + weighted_p4 + weighted_p5))
        
        # 特征融合
        fused_feat = torch.cat(multi_scale_features, dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        # 注意力增强
        output = self.attention(fused_feat)
        
        return output


class RotatedAwareFPN(nn.Module):
    """
    旋转感知特征金字塔网络
    专门为旋转目标检测设计
    """
    
    def __init__(self, channels: List[int], out_channels: int = 256):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        
        # 上采样层
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], out_channels, 2, 2)
            for i in range(len(channels) - 1)
        ])
        
        # 下采样层
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            for _ in range(len(channels) - 1)
        ])
        
        # 特征处理层
        self.process_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i] if i == 0 else out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                MCAttention(out_channels, out_channels)
            )
            for i in range(len(channels))
        ])
        
        # 旋转感知卷积
        self.rotated_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels//4)
            for _ in range(len(channels))
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征列表，从高分辨率到低分辨率
            
        Returns:
            List[torch.Tensor]: 处理后的特征列表
        """
        # 自顶向下路径
        top_down_features = []
        for i in range(len(features)):
            if i == 0:
                # 最高层特征
                feat = self.process_layers[i](features[i])
            else:
                # 上采样并融合
                upsampled = self.upsample_layers[i-1](top_down_features[-1])
                if upsampled.shape[-2:] != features[i].shape[-2:]:
                    upsampled = F.interpolate(upsampled, size=features[i].shape[-2:], 
                                            mode='bilinear', align_corners=False)
                feat = features[i] + upsampled
                feat = self.process_layers[i](feat)
            
            # 旋转感知处理
            rotated_feat = self.rotated_convs[i](feat)
            feat = feat + rotated_feat
            
            top_down_features.append(feat)
        
        # 自底向上路径
        bottom_up_features = []
        for i in range(len(top_down_features) - 1, -1, -1):
            if i == len(top_down_features) - 1:
                # 最底层特征
                feat = top_down_features[i]
            else:
                # 下采样并融合
                downsampled = self.downsample_layers[i](bottom_up_features[-1])
                if downsampled.shape[-2:] != top_down_features[i].shape[-2:]:
                    downsampled = F.interpolate(downsampled, size=top_down_features[i].shape[-2:],
                                              mode='bilinear', align_corners=False)
                feat = top_down_features[i] + downsampled
                feat = self.process_layers[i](feat)
            
            bottom_up_features.append(feat)
        
        # 反转顺序以匹配原始顺序
        return bottom_up_features[::-1]


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    使用不同尺度的卷积核提取多尺度特征
    """
    
    def __init__(self, c1: int, c2: int, scales: tuple = (1, 3, 5, 7)):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.scales = scales
        
        # 多尺度卷积分支
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(c1, c2 // len(scales), kernel_size=s, padding=s//2)
            for s in scales
        ])
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(c2, c2, 1)
        
        # 注意力机制
        self.attention = SCAttention(c2, c2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x))
        
        # 特征融合
        fused_feat = torch.cat(scale_features, dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        # 注意力增强
        output = self.attention(fused_feat)
        
        return output


class AFPN(nn.Module):
    """
    自适应特征金字塔网络主模块
    """
    
    def __init__(self, channels: List[int], out_channels: int = 256):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        
        # 特征提取器
        self.feature_extractors = nn.ModuleList([
            MultiScaleFeatureExtractor(ch, out_channels)
            for ch in channels
        ])
        
        # 旋转感知FPN
        self.rotated_fpn = RotatedAwareFPN([out_channels] * len(channels), out_channels)
        
        # 自适应融合
        self.adaptive_fusion = AdaptiveFeatureFusion(
            out_channels, out_channels, out_channels, out_channels
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征列表
            
        Returns:
            List[torch.Tensor]: 处理后的特征列表
        """
        # 多尺度特征提取
        extracted_features = []
        for i, feat in enumerate(features):
            extracted_feat = self.feature_extractors[i](feat)
            extracted_features.append(extracted_feat)
        
        # 旋转感知FPN处理
        fpn_features = self.rotated_fpn(extracted_features)
        
        return fpn_features


# 为了兼容YOLOv11的模块注册
class AdaptiveFPN(nn.Module):
    """AFPN模块的简化接口"""
    
    def __init__(self, channels: List[int], out_channels: int = 256, **kwargs):
        super().__init__()
        self.afpn = AFPN(channels, out_channels)
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.afpn(features)
