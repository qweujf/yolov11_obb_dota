# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Multi-Scale Cross-Axis Attention (MCAttention) Module for YOLOv11-OBB
前沿的多尺度交叉轴注意力机制，专为遥感图像目标检测设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
# 从 ultralytics 导入基础模块
# 注意：Conv 和 autopad 在当前实现中未直接使用，但保留导入以保持兼容性
try:
    from ultralytics.nn.modules.conv import Conv, autopad
except ImportError:
    # 如果导入失败，定义占位符（实际代码中未使用）
    Conv = None
    autopad = None


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
        
        # 5. 轻量级注意力机制（优化版本，大幅减少计算量）
        # 使用通道注意力 + 简化的空间注意力，完全避免大矩阵乘法
        q = self.q_conv(x)  # [B, C, H, W]
        k = self.k_conv(x)  # [B, C, H, W]
        v = self.v_conv(x)  # [B, C, H, W]
        
        # 方法1: 通道注意力（轻量级，O(C)复杂度）
        q_pool = F.adaptive_avg_pool2d(q, 1).squeeze(-1).squeeze(-1)  # [B, C]
        k_pool = F.adaptive_avg_pool2d(k, 1).squeeze(-1).squeeze(-1)  # [B, C]
        channel_attn = torch.sigmoid(torch.sum(q_pool * k_pool, dim=1, keepdim=True))  # [B, 1]
        
        # 方法2: 简化的空间注意力（使用逐元素乘积代替矩阵乘法，O(C*H*W)复杂度）
        # 对于所有特征图大小，都使用这种轻量级方法，避免 O(H*W*H*W) 的复杂度
        # 使用逐通道的点积作为空间注意力权重
        spatial_attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / (C ** 0.5))  # [B, 1, H, W]
        
        # 结合通道和空间注意力
        attn_output = v * spatial_attn * channel_attn.view(B, 1, 1, 1)
        
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
    
    def __init__(self, c1: int, c2: int, *args, **kwargs):
        super().__init__()
        # 处理位置参数：scales, reduction, num_heads
        # parse_model 会重新构建 args 为 [c1, c2, *args[1:]]
        # 所以如果 YAML 是 [256, (3,5,7), 16, 8]，实际调用时 args = [c1, 256, [3,5,7], 16, 8]
        # 注意：YAML中的元组会被解析为列表，需要转换
        scales = (3, 5, 7)  # 默认值
        reduction = 16
        num_heads = 8
        
        # 解析位置参数：按顺序查找 scales（列表/元组）、reduction（小整数）、num_heads（小整数）
        for arg in args:
            if isinstance(arg, (list, tuple)) and all(isinstance(x, int) for x in arg):
                # 这是 scales 参数
                scales = tuple(arg)
            elif isinstance(arg, int):
                if arg < 100:
                    # 小整数，可能是 reduction 或 num_heads
                    if reduction == 16:  # 如果 reduction 还是默认值，这个是 reduction
                        reduction = arg
                    elif num_heads == 8:  # 如果 reduction 已设置，这个是 num_heads
                        num_heads = arg
        
        # 从 kwargs 中提取参数（优先级更高）
        if 'scales' in kwargs:
            scales_val = kwargs['scales']
            scales = tuple(scales_val) if isinstance(scales_val, (list, tuple)) else scales
        reduction = kwargs.get('reduction', reduction)
        num_heads = kwargs.get('num_heads', num_heads)
        
        self.attention = MultiScaleCrossAxisAttention(c1, c2, scales=scales, reduction=reduction, num_heads=num_heads)
        
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
    
    def __init__(self, c1: int, c2: int, *args, **kwargs):
        super().__init__()
        # 处理位置参数：reduction
        # YAML格式: [c1, c2, reduction]
        reduction = 16  # 默认值
        
        if len(args) >= 1:
            reduction = args[0] if isinstance(args[0], int) else reduction
        
        # 从 kwargs 中提取参数（优先级更高）
        reduction = kwargs.get('reduction', reduction)
        
        self.attention = SpatialChannelAttention(c1, c2, reduction=reduction)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)

