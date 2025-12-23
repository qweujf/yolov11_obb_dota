# Faster R-CNN 对比实验

本文件夹用于运行 Faster R-CNN 在 DOTAv2.0 数据集上的对比实验。

## 实验目的

作为对比实验的基线方法，验证本文提出的 YOLO_DDBC 算法相对于经典两阶段检测方法 Faster R-CNN 的性能优势。

## 数据集

- **数据集**：DOTAv2.0
- **输入尺寸**：1024×1024 像素
- **训练/验证/测试集划分**：与 YOLO_DDBC 实验保持一致

## 实验配置

- **训练轮数**：300 epochs
- **优化器**：SGD
- **初始学习率**：0.01
- **批次大小**：4
- **其他超参数**：见 `config.yaml`

## 运行方式

### 训练
```bash
python train.py --config config.yaml --seed 42 --device 0
```

### 评估
```bash
python eval.py --config config.yaml --ckpt <checkpoint_path>
```

## 结果记录

实验结果将记录在 `results/` 目录下，包括：
- 训练日志
- 模型检查点
- 评估结果（mAP@0.5, mAP@0.5:0.95 等）

## 注意事项

- 确保使用与 YOLO_DDBC 相同的训练和测试设置，以保证对比的公平性
- 建议使用相同的随机种子，确保数据划分一致

