# YOLOv8n-OBB 对比实验

本文件夹用于运行 YOLOv8n-OBB 在 DOTAv2.0 数据集上的对比实验。

## 实验目的

作为对比实验的基线方法之一，验证本文提出的 YOLO_DDBC 算法相对于 YOLOv8-OBB 的性能优势。

## 数据集

- **数据集**：DOTAv2.0
- **输入尺寸**：1024×1024 像素
- **训练/验证/测试集划分**：与 YOLO_DDBC 实验保持一致

## 实验配置

- **模型**：YOLOv8n-OBB
- **训练轮数**：300 epochs
- **优化器**：SGD
- **初始学习率**：0.01
- **批次大小**：4
- **其他超参数**：见 `config.yaml` 或使用默认配置

## 运行方式

### 训练
```bash
python train.py
```

### 评估
```bash
# 使用 ultralytics 的评估功能
python -c "from ultralytics import YOLO; model = YOLO('runs/obb/yolov8n_obb_dota/weights/best.pt'); model.val()"
```

## 结果记录

实验结果将记录在 `runs/obb/yolov8n_obb_dota/` 目录下，包括：
- 训练日志
- 模型检查点
- 评估结果（mAP@0.5, mAP@0.5:0.95 等）

## 注意事项

- 确保使用与 YOLO_DDBC 相同的训练和测试设置，以保证对比的公平性
- 建议使用相同的随机种子，确保数据划分一致
- 首次运行会自动下载 YOLOv8n-OBB 预训练权重

