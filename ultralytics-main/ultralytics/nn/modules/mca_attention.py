# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Multi-Scale Cross-Axis Attention (MCAttention) Module for YOLOv11-OBB
前沿的多尺度交叉轴注意力机制，专为遥感图像目标检测设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .conv import Conv, autopad


class MultiScaleCrossAxisAttention(nn.Module):
    """
    多尺度交叉轴注意力机制 (MCAttention)
    
    创新点：
    1. 多尺度特征提取：使用不同尺度的卷积核捕获多尺度信息
    2. 交叉轴注意力：分别处理水平和垂直轴的特征
    3. 全局上下文感知：通过全局平均池化捕获全局信息
    4. 自适应权重分配：根据特征重要性动态调整权重
    
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数  
        scales (tuple): 多尺度卷积核大小，默认(3, 5, 7)
        reduction (int): 通道压缩比例，默认16
        num_heads (int): 注意力头数，默认8
    """
    
    def __init__(self, c1: int, c2: int, scales: tuple = (3, 5, 7), 
                 reduction: int = 16, num_heads: int = 8):
        super().__init__()
        assert c1 == c2, "输入输出通道数必须相等"
        
        self.c1 = c1
        self.c2 = c2
        self.scales = scales
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        
        # 多尺度卷积分支
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=s, padding=s//2, groups=c1//4) 
            for s in scales
        ])
        
        # 交叉轴注意力分支
        self.horizontal_conv = nn.Conv2d(c1, c1, kernel_size=(1, 3), padding=(0, 1))
        self.vertical_conv = nn.Conv2d(c1, c1, kernel_size=(3, 1), padding=(1, 0))
        
        # 全局上下文分支
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(c1, c1, 1)
        
        # 注意力权重生成
        self.attention_conv = nn.Conv2d(c1 * (len(scales) + 3), c1, 1)
        
        # 多头注意力
        self.q_conv = nn.Conv2d(c1, c1, 1)
        self.k_conv = nn.Conv2d(c1, c1, 1)
        self.v_conv = nn.Conv2d(c1, c1, 1)
        
        # 输出投影
        self.out_conv = nn.Conv2d(c1, c2, 1)
        
        # 层归一化
        self.norm = nn.LayerNorm(c1)
        
        # 激活函数
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征图 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # 2. 交叉轴注意力
        h_feat = self.horizontal_conv(x)  # 水平轴特征
        v_feat = self.vertical_conv(x)    # 垂直轴特征
        
        # 3. 全局上下文
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 4. 特征融合
        fused_features = torch.cat(multi_scale_features + [h_feat, v_feat, global_feat], dim=1)
        attention_weights = self.attention_conv(fused_features)
        attention_weights = torch.sigmoid(attention_weights)
        
        # 5. 多头注意力机制
        q = self.q_conv(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.k_conv(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v_conv(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        attn_output = torch.matmul(v, attn_weights.transpose(-2, -1))
        attn_output = attn_output.view(B, C, H, W)
        
        # 6. 残差连接和输出
        output = x + attn_output * attention_weights
        output = self.out_conv(output)
        
        return self.activation(output)


class AdaptiveMultiScaleAttention(nn.Module):
    """
    自适应多尺度注意力模块
    根据输入特征自适应选择最合适的尺度
    """
    
    def __init__(self, c1: int, c2: int, scales: tuple = (3, 5, 7, 9)):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.scales = scales
        
        # 尺度选择网络
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, len(scales), 1),
            nn.Softmax(dim=1)
        )
        
        # 多尺度卷积
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=s, padding=s//2, groups=c1//8)
            for s in scales
        ])
        
        # 输出融合
        self.fusion_conv = nn.Conv2d(c1, c2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 计算尺度权重
        scale_weights = self.scale_selector(x)  # [B, num_scales, 1, 1]
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # 加权融合
        weighted_features = []
        for i, feat in enumerate(multi_scale_features):
            weight = scale_weights[:, i:i+1, :, :]
            weighted_features.append(feat * weight)
        
        # 特征融合
        fused_feat = sum(weighted_features)
        output = self.fusion_conv(fused_feat)
        
        return output


class SpatialChannelAttention(nn.Module):
    """
    空间-通道双重注意力机制
    结合空间注意力和通道注意力
    """
    
    def __init__(self, c1: int, c2: int, reduction: int = 16):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c1, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.out_conv = nn.Conv2d(c1, c2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa
        
        # 输出
        output = self.out_conv(x_sa)
        return output


# 为了兼容YOLOv11的模块注册
class MCAttention(nn.Module):
    """MCAttention模块的简化接口"""
    
    def __init__(self, c1: int, c2: int, **kwargs):
        super().__init__()
        self.attention = MultiScaleCrossAxisAttention(c1, c2, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class AdaptiveAttention(nn.Module):
    """自适应注意力模块的简化接口"""
    
    def __init__(self, c1: int, c2: int, **kwargs):
        super().__init__()
        self.attention = AdaptiveMultiScaleAttention(c1, c2, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class SCAttention(nn.Module):
    """空间-通道注意力模块的简化接口"""
    
    def __init__(self, c1: int, c2: int, **kwargs):
        super().__init__()
        self.attention = SpatialChannelAttention(c1, c2, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)
