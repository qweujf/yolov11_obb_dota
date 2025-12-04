# Full Enhanced Model 实验

## 模型组成

本实验组合了所有改进模块，构建完整的增强模型：

- **Baseline**: YOLOv11-OBB 基准模型
- **C3k2_ROAM**: 旋转感知特征提取模块（替换 Backbone 中的 C3k2）
- **SFE_DRB**: 空间特征增强与深度残差瓶颈（在 Backbone 中）
- **AFA**: 自适应特征对齐（在 Head/FPN 中，Concat 之前）
- **RAL Loss**: 旋转感知损失函数（训练时自动启用）

## 运行

```bash
python train.py --config config.yaml --seed 42 --device 0
```

## 配置文件说明

- `config.yaml`: 实验配置，指定使用的模型配置文件
- `train.py`: 训练脚本，自动加载配置并开始训练

## 模型配置文件

模型配置文件位于：`configs/model/yolo11-obb-full-enhanced.yaml`

该配置文件组合了：
- Backbone: C3k2_ROAM + SFE_DRB
- Head: AFA 特征对齐 + 标准 FPN
- Loss: RAL Loss（在 `ultralytics/utils/loss.py` 中实现，训练时自动使用）

## 注意事项

1. RAL Loss 已经在 `ultralytics/utils/loss.py` 中实现，训练时会自动使用，无需额外配置
2. 确保所有自定义模块（C3k2_ROAM, SFE_DRB, AFA）已在 `ultralytics/nn/modules/__init__.py` 中注册
3. 建议使用预训练权重 `yolo11n-obb.pt` 进行初始化训练

