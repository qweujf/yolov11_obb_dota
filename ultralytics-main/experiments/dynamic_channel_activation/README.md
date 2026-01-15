# 动态通道激活阈值消融实验

## 实验目的

通过消融实验评估不同动态通道激活阈值 τ 对模型性能（mAP@0.5, mAP@0.5-0.95）和计算量（FLOPs）的影响。

## 实验配置

- **基准模型**: YOLOv11n-OBB
- **数据集**: DOTA-v2.0
- **阈值设置**: 
  - 不使用动态激活（baseline）
  - τ = 0.10
  - τ = 0.15
  - τ = 0.20
  - τ = 0.25
  - τ = 0.30

## 实验结构

```
dynamic_channel_activation/
├── README.md
├── baseline/          # 不使用动态激活（基准）
│   ├── config.yaml
│   └── train.py
├── threshold_0.10/    # τ = 0.10
│   ├── config.yaml
│   └── train.py
├── threshold_0.15/    # τ = 0.15
│   ├── config.yaml
│   └── train.py
├── threshold_0.20/    # τ = 0.20
│   ├── config.yaml
│   └── train.py
├── threshold_0.25/    # τ = 0.25
│   ├── config.yaml
│   └── train.py
└── threshold_0.30/    # τ = 0.30
    ├── config.yaml
    └── train.py
```

## 运行方式

进入对应的阈值文件夹，运行：

```bash
python train.py
```

## 结果保存

所有实验结果保存在 `runs/obb/dynamic_activation/` 目录下，每个阈值对应一个子目录。

## 注意事项

1. **动态通道激活实现**: 需要确保模型代码中已实现动态通道激活机制，并能通过参数控制阈值 τ。
2. **训练一致性**: 所有实验使用相同的训练设置（epochs, lr, batch size 等），确保结果可比较。
3. **参数量**: 动态通道激活不改变模型结构，因此所有配置的参数量应该相同。

