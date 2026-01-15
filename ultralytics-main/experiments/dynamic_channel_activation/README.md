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

## 计算 GFLOPs

使用 `calculate_gflops.py` 脚本可以计算模型的 GFLOPs：

```bash
# 计算 baseline 模型的 GFLOPs
cd ultralytics-main/experiments/dynamic_channel_activation
python calculate_gflops.py
```

在脚本中修改配置：
- `MODEL_PATH`: 模型路径（.yaml 或 .pt）
- `IMG_SIZE`: 输入图像尺寸（默认 1024）
- `DYNAMIC_ACTIVATION_THRESHOLD`: 动态激活阈值（None 表示不使用）
- `METHOD`: 计算方法（'thop' 或 'torch_profiler'）

**注意**：
- 需要安装 `thop` 库：`pip install thop`
- 动态激活在推理时生效，会降低实际 FLOPs，但 `thop` 可能无法准确统计动态跳过的计算

## 注意事项

1. **动态通道激活实现**: 已实现动态通道激活机制，通过环境变量 `DYNAMIC_ACTIVATION_THRESHOLD` 控制阈值 τ。
2. **训练一致性**: 所有实验使用相同的训练设置（epochs, lr, batch size 等），确保结果可比较。
3. **参数量**: 动态通道激活会增加少量参数量（门控分支），但主要影响 FLOPs（运行时计算量）。

