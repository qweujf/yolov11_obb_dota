# GhostNetV2 Backbone 实验

## 实验说明

本实验将 YOLO_DDBC 的骨干网络整体替换为 GhostNetV2 轻量级骨干网络，用于验证轻量化优化的效果。

## 模型改动

### 骨干网络替换

**原 YOLO_DDBC 骨干网络结构：**
```
Conv → Conv → C3k2_DCA → Conv → C3k2_DCA → Conv → C3k2_DCA → Conv → SPPF → C2PSA
```

**替换后的 GhostNetV2 骨干网络结构：**
```
Conv → GhostConv → C3Ghost → DFC Attention → 
GhostConv → C3Ghost → DFC Attention → 
GhostConv → C3Ghost → DFC Attention → 
GhostConv → SPPF → C2PSA
```

### 保持不变的部分

- **Neck**: DDPG + BiFPN_DWC（保持不变）
- **Head**: 标准检测头（保持不变）
- **Loss**: CSL_ProbIoU（保持不变）

## 核心组件

### GhostNetV2 骨干网络

- **GhostConv**: Ghost 卷积，通过主卷积和廉价操作生成特征图
- **C3Ghost**: 基于 Ghost 卷积的 C3 模块
- **DFC Attention**: 解耦全连接注意力机制，硬件友好的长距离依赖捕获

### 输出特征尺度

- P2/4, P3/8, P4/16, P5/32（与原始 YOLO_DDBC 保持一致）

## 运行

```bash
cd ultralytics-main/experiments/ablation_ghostnetv2_backbone/ghostnetv2
python train.py
```

## 配置文件说明

- `config.yaml`: 实验配置，指定使用的模型配置文件
- `train.py`: 训练脚本，自动加载配置并开始训练

## 模型配置文件

模型配置文件位于：`configs/model/yolo11-obb-ddbc-ghostnetv2.yaml`

**注意**：该配置文件需要：
1. 将 YOLO_DDBC 的骨干网络替换为 GhostNetV2 结构
2. 保持 Neck（DDPG + BiFPN_DWC）和 Head 不变
3. 确保输出特征尺度与原始模型兼容

## 预期效果

- **参数量减少**: GhostNetV2 通过 Ghost 卷积显著减少参数量
- **计算量降低**: DFC Attention 相比标准注意力机制计算量更低
- **检测精度**: 在保持检测精度的前提下实现轻量化

