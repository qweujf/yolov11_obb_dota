# C3k2_DCN_CA 实验

## 模块说明

本实验实现了基于 **C3k2 + DCNv2 + Coordinate Attention** 的改进模块，用于提升旋转目标检测性能。

### 模块组成

1. **DCNv2 Bottleneck**: 在 Bottleneck 中使用可变形卷积（DCNv2）替代标准卷积
   - 自适应调整卷积核采样位置
   - 更好地适应旋转目标的形状变化
   - 参考文献: Zhu et al., "Deformable ConvNets v2: More Deformable, Better Results", CVPR 2019

2. **Coordinate Attention**: 在特征融合后添加坐标注意力机制
   - 通过水平和垂直方向的全局池化捕获方向信息
   - 生成通道和空间注意力权重
   - 增强对旋转目标的方向感知能力
   - 参考文献: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

### 模块结构

```
C3k2_DCN_CA:
  Conv → Split → DCNv2Bottleneck×n → Concat → CoordinateAttention → Conv
```

## 文件结构

```
experiments/c3k2_dcn_ca/
├── config.yaml          # 实验配置文件
├── train.py            # 训练脚本
└── README.md           # 本文件
```

## 使用方法

### 1. 训练模型

```bash
cd ultralytics-main/experiments/c3k2_dcn_ca
python train.py
```

### 2. 配置文件说明

编辑 `config.yaml` 可以调整训练参数：

```yaml
model: configs/model/yolo11-obb-c3k2-dcn-ca.yaml

train:
  project: runs/obb/c3k2_dcn_ca
  name: c3k2_dcn_ca_exp
  epochs: 300
  batch: 4
  # ... 其他参数
```

### 3. 模型配置文件

模型定义在 `configs/model/yolo11-obb-c3k2-dcn-ca.yaml`，所有 `C3k2` 模块已替换为 `C3k2_DCN_CA`。

## 预期效果

- **形状适应**: DCNv2 能够自适应调整采样位置，更好地处理旋转目标的形状变化
- **方向感知**: Coordinate Attention 通过水平和垂直方向信息增强方向感知能力
- **特征增强**: 组合设计在特征提取和融合两个阶段同时增强对旋转目标的感知

## 参考文献

1. Zhu, X., et al. (2019). Deformable ConvNets v2: More Deformable, Better Results. CVPR 2019.
2. Hou, Q., et al. (2021). Coordinate Attention for Efficient Mobile Network Design. CVPR 2021.

