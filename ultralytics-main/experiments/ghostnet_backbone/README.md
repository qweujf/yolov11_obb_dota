# GhostNet Backbone 实验

## 模型组成

本实验将整个backbone替换为GhostNet，实现轻量化设计：

- **Backbone**: GhostNet (GhostConv + C3Ghost)
  - 完全替换标准Conv和C3k2为GhostConv和C3Ghost
  - 移除C3k2_ROAM和SFE_DRB创新点模块
  - 保留SPPF和C2PSA注意力模块
- **Head**: 标准FPN (C3k2)
- **Loss**: RAL Loss（旋转感知损失函数，训练时自动启用）

## 运行

```bash
python experiments/ghostnet_backbone/train.py
```

## 配置文件说明

- `config.yaml`: 实验配置，指定使用的模型配置文件
- `train.py`: 训练脚本，自动加载配置并开始训练

## 模型配置文件

模型配置文件位于：`configs/model/yolo11-obb-ghostnet.yaml`

该配置文件：
- Backbone: 完全使用GhostNet (GhostConv + C3Ghost)
- Head: 标准FPN (C3k2)
- Loss: RAL Loss（在 `ultralytics/utils/loss.py` 中实现，训练时自动使用）

## 预期效果

- **参数量**: ~2.1M (vs baseline 2.7M, -22%; vs full_enhanced 6.6M, -68%)
- **GFLOPs**: ~5.5 (vs baseline 6.9, -20%; vs full_enhanced 18.2, -70%)
- **速度**: ~55 FPS (vs baseline 45 FPS, +22%; vs full_enhanced 18 FPS, +200%)
- **精度**: mAP50 ~0.70-0.72 (vs baseline 0.726, -1~2%; vs full_enhanced ~0.75-0.78, -5~8%)

## 注意事项

1. RAL Loss 已经在 `ultralytics/utils/loss.py` 中实现，训练时会自动使用，无需额外配置
2. GhostConv和C3Ghost模块已在Ultralytics中实现，无需额外实现
3. 建议使用预训练权重 `yolo11n-obb.pt` 进行初始化训练
4. 由于移除了C3k2_ROAM和SFE_DRB，小目标和旋转目标的检测能力会有所下降

