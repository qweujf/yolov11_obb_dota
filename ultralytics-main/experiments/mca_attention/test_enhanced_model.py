#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MCAttention YOLOv11-OBB 模型
验证多尺度交叉轴注意力机制和自适应特征金字塔网络
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ultralytics.nn.modules import MCAttention, SCAttention, AdaptiveFPN

def test_attention_modules():
    """测试注意力模块"""
    print("🔍 测试注意力模块...")
    
    # 测试MCAttention
    print("测试MCAttention模块...")
    mca = MCAttention(256, 256, scales=(3, 5, 7), reduction=16, num_heads=8)
    x = torch.randn(2, 256, 64, 64)
    output = mca(x)
    print(f"MCAttention输入形状: {x.shape}")
    print(f"MCAttention输出形状: {output.shape}")
    assert output.shape == x.shape, "MCAttention输出形状不匹配"
    
    # 测试SCAttention
    print("测试SCAttention模块...")
    sca = SCAttention(512, 512, reduction=16)
    x = torch.randn(2, 512, 32, 32)
    output = sca(x)
    print(f"SCAttention输入形状: {x.shape}")
    print(f"SCAttention输出形状: {output.shape}")
    assert output.shape == x.shape, "SCAttention输出形状不匹配"
    
    print("✅ 注意力模块测试通过!")

def test_afpn_module():
    """测试自适应特征金字塔网络"""
    print("🔍 测试AFPN模块...")
    
    # 创建模拟特征
    features = [
        torch.randn(2, 1024, 8, 8),   # P5
        torch.randn(2, 512, 16, 16),  # P4
        torch.randn(2, 256, 32, 32),  # P3
    ]
    
    # 测试AdaptiveFPN
    afpn = AdaptiveFPN([1024, 512, 256], 256)
    output_features = afpn(features)
    
    print(f"AFPN输入特征数量: {len(features)}")
    print(f"AFPN输出特征数量: {len(output_features)}")
    for i, feat in enumerate(output_features):
        print(f"特征{i}形状: {feat.shape}")
    
    assert len(output_features) == len(features), "AFPN输出特征数量不匹配"
    print("✅ AFPN模块测试通过!")

def test_enhanced_model():
    """测试 MCAttention YOLOv11-OBB 模型"""
    print("🔍 测试 MCAttention YOLOv11-OBB 模型...")
    
    try:
        # 加载 MCAttention 模型（使用新的配置文件路径）
        model_path = project_root / 'configs' / 'model' / 'yolo11-obb-enhanced.yaml'
        model = YOLO(str(model_path))
        
        print(f"模型加载成功: {model_path}")
        print(f"模型类别数: {model.model[-1].nc}")
        
        # 创建测试输入
        test_input = torch.randn(1, 3, 1024, 1024)
        print(f"测试输入形状: {test_input.shape}")
        
        # 前向传播测试
        with torch.no_grad():
            outputs = model.model(test_input)
            print(f"模型输出类型: {type(outputs)}")
            if isinstance(outputs, (list, tuple)):
                print(f"输出数量: {len(outputs)}")
                for i, out in enumerate(outputs):
                    if hasattr(out, 'shape'):
                        print(f"输出{i}形状: {out.shape}")
        
        print("✅ MCAttention 模型测试通过!")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_parameters():
    """测试模型参数数量"""
    print("🔍 测试模型参数...")
    
    try:
        model_path = project_root / 'configs' / 'model' / 'yolo11-obb-enhanced.yaml'
        model = YOLO(str(model_path))
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        print("✅ 参数统计完成!")
        
    except Exception as e:
        print(f"❌ 参数统计失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 开始测试 MCAttention YOLOv11-OBB 模型")
    print("=" * 50)
    
    # 设置环境
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    try:
        # 测试注意力模块
        test_attention_modules()
        print()
        
        # 测试AFPN模块
        test_afpn_module()
        print()
        
        # 测试增强版模型
        if test_enhanced_model():
            print()
            # 测试模型参数
            test_model_parameters()
        
        print("=" * 50)
        print("🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

