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
# 使用默认配置文件（当前目录下的 config.yaml）
python train.py

# 或指定配置文件
python train.py --config config.yaml --seed 42 --device 0
```

### 评估
```bash
python eval.py --ckpt <checkpoint_path>
```

## 环境要求

### 安装 mmdetection

```bash
# 方式1：使用 pip（推荐）
pip install mmdet mmengine mmcv

# 方式2：使用 conda
conda install -c conda-forge mmdet
```

### 验证安装

```bash
python -c "from mmdet.apis import init_detector; print('✅ mmdetection 安装成功')"
```

## ⚠️ 重要提示

### 关于 DOTA 数据集

**Faster R-CNN 是水平框检测器，而 DOTA 数据集使用旋转框标注。**

在使用 Faster R-CNN 进行对比实验时，需要：

1. **将旋转框转换为水平框**：使用旋转框的最小外接矩形（MBR）作为水平框
2. **或使用 mmrotate**：如果需要真正的旋转目标检测，建议使用 mmrotate 框架中的 Oriented R-CNN

### 数据集格式转换

脚本会自动创建 mmdetection 配置文件，但需要确保：

1. DOTA 数据集已转换为 COCO 格式（水平框）
2. 或者修改配置文件中的数据路径和格式

### 自动配置

运行 `train.py` 时，如果 `faster_rcnn_config.py` 不存在，脚本会自动创建一个基础配置文件。你可以根据需要修改该配置文件。

## 结果记录

实验结果将记录在 `results/` 目录下，包括：
- 训练日志
- 模型检查点
- 评估结果（mAP@0.5, mAP@0.5:0.95 等）

## 注意事项

- 确保使用与 YOLO_DDBC 相同的训练和测试设置，以保证对比的公平性
- 建议使用相同的随机种子，确保数据划分一致

