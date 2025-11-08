# Baseline vs MCAttention 对比实验

本目录包含 Baseline 和 MCAttention 模型的对比实验，用于验证 MCAttention 模块的有效性。

## 实验设计

### 实验变体

1. **baseline** - 原始 YOLO11-OBB 模型（无 MCAttention）
   - 模型配置：`configs/model/yolov11_obb.yaml`
   - 实验目录：`experiments/baseline/`

2. **mca_attention** - 使用完整 MCAttention 的增强版模型
   - 模型配置：`configs/model/yolo11-obb-enhanced.yaml`
   - 实验目录：`experiments/mca_attention/`

## 使用方法

### 1. 运行 Baseline 实验

```bash
cd experiments/baseline
python train.py
```

### 2. 运行 MCAttention 实验

```bash
cd experiments/mca_attention
python train.py
```

### 3. 对比分析结果

训练完成后，运行对比分析脚本：

```bash
cd experiments
python compare_baseline_enhanced.py
```

该脚本会自动：
- 加载两个实验的训练结果（mAP50, mAP50-95）
- 分析模型复杂度（参数量、GFLOPs）
- 测量推理速度（推理时间、FPS）
- 生成对比表格和可视化图表
- 计算改进幅度

## 评价指标

对比分析包含以下所有评价指标：

### 1. 检测精度
- **mAP50**: 在 IoU=0.5 时的平均精度
- **mAP50-95**: 在 IoU=0.5:0.95 时的平均精度

### 2. 模型复杂度
- **参数量**: 模型参数总数（M）
- **GFLOPs**: 浮点运算次数（十亿次）

### 3. 推理性能
- **推理时间**: 单张图像的平均推理时间（ms）
- **FPS**: 每秒处理的图像帧数

## 输出结果

运行对比分析后，会生成：

1. **comparison_results.csv** - 详细对比表格（CSV格式）
2. **comparison_plots.png** - 可视化对比图表（包含6个子图）

结果保存在 `experiments/` 目录下。

## 注意事项

1. **训练权重**: 对比分析脚本会优先使用训练好的权重（`runs/train/weights/best.pt`），如果不存在则使用模型配置。

2. **推理速度测试**: 需要训练好的模型权重，如果权重不存在会跳过速度测试。

3. **设备要求**: 推理速度测试默认使用 CUDA（如果可用），否则使用 CPU。

4. **图像尺寸**: 默认使用 1024x1024 进行复杂度分析和速度测试，与训练配置一致。

## 预期结果

通过对比实验，我们可以验证：
- MCAttention 模块是否提升了检测精度
- 增加的参数量和计算量是否合理
- 推理速度的影响是否可接受

