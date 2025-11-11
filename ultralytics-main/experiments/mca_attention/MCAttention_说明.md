# MCAttention 模块说明

## 模块组成

MCAttention（Multi-Scale Cross-Axis Attention）模块包含以下**6个主要部分**：

### 1. 多尺度特征提取（Multi-Scale Feature Extraction）
- **功能**：使用不同尺度的卷积核（默认 3×3, 5×5, 7×7）捕获多尺度信息
- **实现**：分组卷积（groups=c1//4），减少参数量
- **输出**：多个不同尺度的特征图

### 2. 交叉轴注意力（Cross-Axis Attention）
- **水平轴注意力**：使用 (1, 3) 卷积核捕获水平方向的特征
- **垂直轴注意力**：使用 (3, 1) 卷积核捕获垂直方向的特征
- **作用**：专门处理遥感图像中常见的水平/垂直目标（如车辆、建筑物）

### 3. 全局上下文感知（Global Context Awareness）
- **功能**：通过全局平均池化捕获全局信息
- **实现**：AdaptiveAvgPool2d(1) → Conv(1×1) → 上采样回原始尺寸
- **作用**：提供全局上下文信息，帮助模型理解整体场景

### 4. 特征融合与权重生成（Feature Fusion & Weight Generation）
- **功能**：融合多尺度特征、交叉轴特征和全局特征
- **实现**：拼接所有特征 → 1×1 卷积 → Sigmoid 激活
- **输出**：注意力权重图，用于后续的特征增强

### 5. 轻量级注意力机制（Lightweight Attention）
- **通道注意力**：O(C) 复杂度，通过全局池化计算通道重要性
- **空间注意力**：O(C×H×W) 复杂度，使用逐元素乘积代替大矩阵乘法
- **优化**：避免 O(H×W×H×W) 的复杂度，大幅提升速度

### 6. 残差连接与输出（Residual Connection & Output）
- **残差连接**：x + attention_output × attention_weights
- **输出投影**：1×1 卷积调整通道数
- **激活函数**：SiLU 激活

## 在 YOLO 模型中的位置

### 当前配置（yolo11-obb-mca-only.yaml）

MCAttention 模块被放置在 **Backbone 的 4 个关键位置**：

Backbone 结构（对比原始模型）：

原始 YOLOv11-OBB：
Conv → Conv → C3k2 → Conv → C3k2 → Conv → C3k2 → Conv → C3k2 → SPPF → C2PSA

MCAttention 版本：
Conv → Conv → C3k2 → [MCA #1] → Conv → C3k2 → [MCA #2] → Conv → C3k2 → [MCA #3] 
→ Conv → C3k2 → SPPF → C2PSA → [MCA #4]

```
Backbone 结构：
├── Conv(64) → Conv(128) → C3k2(256)
├── [MCAttention #1] ← 位置 3，通道数 256，特征图尺寸 P3/8
├── Conv(256) → C3k2(512)
├── [MCAttention #2] ← 位置 6，通道数 512，特征图尺寸 P4/16
├── Conv(512) → C3k2(512)
├── [MCAttention #3] ← 位置 9，通道数 512，特征图尺寸 P4/16
├── Conv(1024) → C3k2(1024) → SPPF → C2PSA
└── [MCAttention #4] ← 位置 14，通道数 1024，特征图尺寸 P5/32
```

### 详细位置说明

| 位置 | 层索引 | 通道数 | 特征图尺寸 | 多尺度卷积核 | 说明 |
|------|--------|--------|------------|--------------|------|
| MCA #1 | 3 | 256 | P3/8 (较大) | [3, 5, 7] | 在 P3 层之前，捕获中等尺度特征 |
| MCA #2 | 6 | 512 | P4/16 (中等) | [3, 5, 7] | 在 P4 层之前，捕获中尺度特征 |
| MCA #3 | 9 | 512 | P4/16 (中等) | [3, 5, 7] | 在 P4 层之后，进一步增强特征 |
| MCA #4 | 14 | 1024 | P5/32 (较小) | [3, 5, 7, 9] | 在 P5 层，增强多尺度特征（使用4个卷积核） |

### 与原始模型的对比

**原始 YOLOv11-OBB (yolov11_obb.yaml)**：
- Backbone 中没有 MCAttention
- 只有标准的 Conv、C3k2、SPPF、C2PSA 模块

**MCAttention 版本 (yolo11-obb-mca-only.yaml)**：
- 在 Backbone 的 4 个位置插入 MCAttention
- 保持 Head 部分不变（标准 FPN，不使用 AdaptiveFPN）
- 用于消融实验，验证 MCAttention 的独立效果

## 参数配置

每个 MCAttention 模块的参数：
- **输入/输出通道数**：与所在层的通道数相同（256, 512, 或 1024）
- **多尺度卷积核**：前3个使用 [3, 5, 7]，最后一个使用 [3, 5, 7, 9]
- **通道压缩比例 (reduction)**：16
- **注意力头数 (num_heads)**：8

## 计算复杂度优化

- **原始设计**：O(H×W×H×W) 的空间注意力矩阵
- **优化后**：O(C×H×W) 的轻量级注意力
- **速度提升**：对于 1024×1024 的特征图，计算量减少约 1000 倍

