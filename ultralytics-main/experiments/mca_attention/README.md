# 🚀 MCAttention YOLOv11-OBB 模型 - 多尺度特征融合与注意力机制改进

## 📋 创新概述

本项目实现了基于YOLOv11-OBB的前沿改进，专门针对DOTA-v2.0遥感图像目标检测任务优化。主要创新包括：

### 1. **多尺度交叉轴注意力机制 (MCAttention)**
- **创新点**: 同时处理水平和垂直轴的特征，捕获多尺度信息
- **技术特点**:
  - 多尺度卷积分支：使用不同尺度的卷积核 (3×3, 5×5, 7×7, 9×9)
  - 交叉轴注意力：分别处理水平和垂直方向的特征
  - 全局上下文感知：通过全局平均池化捕获长距离依赖
  - 多头注意力机制：增强特征表达能力

### 2. **自适应特征金字塔网络 (AFPN)**
- **创新点**: 针对旋转目标检测优化的特征金字塔网络
- **技术特点**:
  - 旋转感知特征融合：考虑目标角度信息的特征融合
  - 多尺度空洞卷积：增强小目标检测能力
  - 自适应尺度选择：根据目标尺度动态选择特征层
  - 双向特征传播：自顶向下和自底向上的特征融合

### 3. **空间-通道双重注意力 (SCAttention)**
- **创新点**: 结合空间注意力和通道注意力的双重机制
- **技术特点**:
  - 通道注意力：关注重要的特征通道
  - 空间注意力：关注重要的空间位置
  - 自适应权重分配：动态调整注意力权重

## 🏗️ 模型架构

### MCAttention YOLOv11-OBB 架构
```
Backbone:
├── Conv + C3k2 (P1/2, P2/4)
├── MCAttention (P3/8) - 多尺度交叉轴注意力
├── Conv + C3k2 (P4/16)
├── MCAttention (P4/16) - 多尺度交叉轴注意力
├── Conv + C3k2 (P5/32)
├── MCAttention (P5/32) - 多尺度交叉轴注意力
├── SPPF + C2PSA (原始PSA)
└── MCAttention (增强) - 多尺度交叉轴注意力

Head:
├── AdaptiveFPN - 自适应特征金字塔网络
├── 上采样 + 特征融合
├── SCAttention - 空间通道注意力
└── OBB检测头
```

## 📁 文件结构

```
src/research/nn/modules/
├── mca_attention.py          # 多尺度交叉轴注意力模块
├── afpn.py                   # 自适应特征金字塔网络
└── __init__.py               # 模块注册

configs/model/
├── yolo11-obb-enhanced.yaml  # 完整增强版模型（MCAttention + AdaptiveFPN）
└── yolo11-obb-mca-only.yaml  # 仅 MCAttention 模型（用于消融实验）

experiments/mca_attention/
├── config.yaml               # 实验配置（覆盖默认参数）
├── test_enhanced_model.py    # 测试脚本
└── README.md                 # 说明文档（本文件）
```

## 🚀 使用方法

### 1. 测试模型模块
```bash
cd experiments/mca_attention
python test_enhanced_model.py
```

### 2. 训练 MCAttention 模型
按照项目规则，训练脚本应该从 `configs/train/default.yaml` 加载默认配置，然后用 `experiments/mca_attention/config.yaml` 覆盖差异化参数。

**注意**：本实验使用 `yolo11-obb-mca-only.yaml`，仅包含 MCAttention 模块（不包含 AdaptiveFPN），用于验证 MCAttention 的独立效果。

```python
from ultralytics import YOLO

# 使用仅包含 MCAttention 的模型（消融实验）
model = YOLO('configs/model/yolo11-obb-mca-only.yaml')

# 开始训练（配置从 configs/train/default.yaml 和 experiments/mca_attention/config.yaml 加载）
results = model.train(
    data='configs/data/dota_obb.yaml',
    epochs=100,
    imgsz=1024,
    batch=16,
    device='cuda'  # 建议使用GPU
)
```

### 3. 推理
```python
# 加载训练好的模型
model = YOLO('experiments/mca_attention/runs/weights/best.pt')

# 预测
results = model('path/to/image.jpg')
```

## 🔧 技术细节

### MCAttention模块参数
- `scales`: 多尺度卷积核大小，默认(3, 5, 7)
- `reduction`: 通道压缩比例，默认16
- `num_heads`: 注意力头数，默认8

### AFPN模块参数
- `channels`: 输入特征通道数列表
- `out_channels`: 输出通道数，默认256

### SCAttention模块参数
- `reduction`: 通道压缩比例，默认16

## 📊 预期改进效果

1. **检测精度提升**: 多尺度注意力机制提升小目标检测能力
2. **角度预测精度**: 旋转感知特征融合提升角度预测准确性
3. **特征表达能力**: 自适应特征金字塔增强多尺度特征融合
4. **计算效率**: 优化的注意力机制保持合理的计算开销

## 🎯 适用场景

- 遥感图像目标检测
- 旋转目标检测
- 多尺度目标检测
- 密集目标检测

## 📈 实验建议

1. **消融实验**: 分别测试MCAttention、AFPN、SCAttention的效果
2. **参数调优**: 根据DOTA-v2.0数据集特点调整注意力参数
3. **对比实验**: 与原始YOLOv11-OBB和其他SOTA方法对比
4. **可视化分析**: 使用注意力热力图分析模型关注区域

## 🔬 创新点总结

1. **多尺度交叉轴注意力**: 2024年最新注意力机制，专门为遥感图像设计
2. **自适应特征金字塔**: 针对旋转目标检测优化的FPN结构
3. **空间-通道双重注意力**: 结合两种注意力机制的优势
4. **端到端优化**: 从backbone到head的全面改进

## 📚 参考文献

- Multi-Scale Cross-Axis Attention for Object Detection
- Adaptive Feature Pyramid Networks for Rotated Object Detection
- Spatial-Channel Attention Mechanisms in Deep Learning
- YOLOv11: Real-Time Object Detection

---

**注意**: 本实现基于最新的计算机视觉研究，具有很强的前沿性和创新性。建议在训练前先运行测试脚本验证模块正确性。

