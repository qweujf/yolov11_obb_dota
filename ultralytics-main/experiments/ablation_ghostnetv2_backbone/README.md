# GhostNetV2 Backbone 消融实验

## 实验目的

验证将 YOLO_DDBC 的骨干网络替换为 GhostNetV2 轻量级骨干网络对模型性能的影响。

## 实验设计

### 实验 1: Baseline（YOLO_DDBC 完整模型）

- **目录**: `baseline/`
- **模型**: 完整的 YOLO_DDBC 模型
- **组件**: C3k2_DCA（骨干）+ DDPG（Neck）+ BiFPN_DWC（Neck）+ CSL_ProbIoU（Loss）

### 实验 2: GhostNetV2 Backbone

- **目录**: `ghostnetv2/`
- **模型**: YOLO_DDBC 骨干网络替换为 GhostNetV2
- **改动**: 
  - Backbone: 标准卷积 → GhostNetV2（GhostConv + C3Ghost + DFC Attention）
  - Neck: DDPG + BiFPN_DWC（保持不变）
  - Head: 标准检测头（保持不变）
  - Loss: CSL_ProbIoU（保持不变）

## 对比指标

- **参数量** (Parameters)
- **计算量** (FLOPs)
- **检测精度** (mAP50, mAP50:95)
- **推理速度** (FPS)

## 运行实验

### 1. 运行 Baseline 实验

```bash
cd baseline
python train.py
```

### 2. 运行 GhostNetV2 Backbone 实验

```bash
cd ghostnetv2
python train.py
```

## 注意事项

1. **模型配置文件**: 需要确保以下配置文件存在：
   - `configs/model/yolo11-obb-ddbc.yaml` (YOLO_DDBC 完整模型)
   - `configs/model/yolo11-obb-ddbc-ghostnetv2.yaml` (GhostNetV2 骨干版本)

2. **模块注册**: 确保所有自定义模块（C3k2_DCA, DDPG, BiFPN_DWC, GhostConv, C3Ghost, DFC Attention）已在 `ultralytics/nn/modules/__init__.py` 中注册

3. **预训练权重**: 建议使用预训练权重进行初始化训练

## 实验结果

实验结果将保存在 `runs/obb/ablation_ghostnetv2_backbone/` 目录下，包括：
- 训练日志
- 模型权重
- 验证结果
- 性能指标

