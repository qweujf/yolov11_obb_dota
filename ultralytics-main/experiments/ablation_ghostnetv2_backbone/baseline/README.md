# YOLO_DDBC Baseline 实验

## 实验说明

本实验使用完整的 YOLO_DDBC 模型作为基线（Baseline），用于后续轻量化消融实验的对比。

## 模型组成

YOLO_DDBC 包含以下四个核心改进模块：

- **C3k2_DCA**: 基于 DCNv4 可变形卷积和坐标注意力的特征提取模块（骨干网络）
- **DDPG**: 膨胀深度可分离金字塔门控融合模块（Neck）
- **BiFPN_DWC**: 动态权重修正双向特征金字塔网络（Neck）
- **CSL_ProbIoU**: 基于 CSL 角度编码和 ProbIoU 的损失函数优化

## 骨干网络结构

```
Conv → Conv → C3k2_DCA → Conv → C3k2_DCA → Conv → C3k2_DCA → Conv → SPPF → C2PSA
```

输出特征尺度：P2/4, P3/8, P4/16, P5/32

## 运行

```bash
cd ultralytics-main/experiments/ablation_ghostnetv2_backbone/baseline
python train.py
```

## 配置文件说明

- `config.yaml`: 实验配置，指定使用的模型配置文件
- `train.py`: 训练脚本，自动加载配置并开始训练

## 模型配置文件

模型配置文件位于：`configs/model/yolo11-obb-ddbc.yaml`

**注意**：如果该配置文件不存在，需要先创建包含所有 YOLO_DDBC 模块的配置文件。

