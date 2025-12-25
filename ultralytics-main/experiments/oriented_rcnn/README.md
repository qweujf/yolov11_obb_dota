# Oriented R-CNN 对比实验

本文件夹用于运行 Oriented R-CNN 在 DOTAv2.0 数据集上的对比实验。

## 实验目的

作为对比实验的经典两阶段旋转目标检测方法，验证本文提出的 YOLO_DDBC 算法相对于 Oriented R-CNN 的性能优势。

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

## 环境要求

### ⚠️ 重要：确保使用正确的 Python 环境

如果你使用 conda 环境，请确保：
1. **激活正确的环境**后再安装和运行：
   ```bash
   # PowerShell 中
   conda activate yolov8_seg
   
   # 或者 CMD 中
   conda activate yolov8_seg
   ```

2. **验证当前 Python 环境**：
   ```bash
   python -c "import sys; print(sys.executable)"
   ```
   应该显示你激活的环境路径，例如：`g:\install_app\anaconda\envs\yolov8_seg\python.exe`

### 安装 mmrotate

mmrotate 是 mmdetection 的旋转目标检测扩展框架，支持 Oriented R-CNN 等旋转目标检测算法。

#### Windows 安装（推荐）

在 Windows 上，`mmcv` 需要特殊处理，建议使用预编译版本：

```bash
# 1. 先升级构建工具
pip install --upgrade pip setuptools wheel

# 2. 安装 mmcv-lite（轻量版，不需要编译，推荐）
pip install mmcv-lite

# 或者安装预编译的 mmcv-full（需要根据你的 CUDA 和 PyTorch 版本选择）
# 查看你的 PyTorch 和 CUDA 版本：
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 然后根据版本安装，例如 CUDA 11.8 + PyTorch 2.0：
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# 3. 安装其他依赖
pip install mmengine mmdet

# 4. 安装 mmrotate
pip install mmrotate
```

#### Linux 安装

```bash
# 方式1：使用 pip（推荐）
pip install mmrotate mmdet mmengine mmcv

# 方式2：从源码安装（最新版本）
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
```

#### 常见问题

**问题1：`metadata-generation-failed` 错误**
- **原因**：Windows 上 `mmcv` 从源码编译失败
- **解决**：使用 `mmcv-lite` 或预编译的 `mmcv-full`

**问题2：找不到预编译版本**
- **解决**：使用 `mmcv-lite`，它是轻量版，功能足够使用

**问题3：`DLL load failed while importing _ext` 错误**
- **原因**：mmcv-full 的 C++ 扩展与当前环境不兼容（Windows 常见问题）
- **解决方案**：
  1. **推荐**：使用 `mmcv-lite`（不需要 C++ 扩展）
     ```bash
     pip uninstall mmcv-full -y
     pip install mmcv-lite
     ```
  2. 重新安装兼容的 `mmcv-full`（根据 PyTorch 和 CUDA 版本）
  3. 安装 Visual C++ Redistributable：https://aka.ms/vs/17/release/vc_redist.x64.exe

### 验证安装

```bash
python -c "from mmrotate.apis import init_detector; print('✅ mmrotate 安装成功')"
```

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

例如：
```bash
python eval.py --ckpt work_dirs/oriented_rcnn_dota/epoch_300.pth --out results.json
```

## 数据集格式

Oriented R-CNN 使用 mmrotate 框架，需要 DOTA 数据集的原始格式：

```
data/DOTAv2.0/
├── train/
│   ├── images/
│   │   ├── P0001.png
│   │   └── ...
│   └── annfiles/
│       ├── P0001.txt
│       └── ...
├── val/
│   ├── images/
│   └── annfiles/
└── test/
    ├── images/
    └── annfiles/
```

DOTA 标注格式（每行一个目标）：
```
x1 y1 x2 y2 x3 y3 x4 y4 category difficult
```

## 自动配置

运行 `train.py` 时，如果 `oriented_rcnn_config.py` 不存在，脚本会自动创建一个基础配置文件。你可以根据需要修改该配置文件。

## 结果记录

实验结果将记录在 `work_dirs/` 目录下，包括：
- 训练日志
- 模型检查点
- 评估结果（mAP@0.5, mAP@0.5:0.95 等）

## 注意事项

- 确保使用与 YOLO_DDBC 相同的训练和测试设置，以保证对比的公平性
- 建议使用相同的随机种子，确保数据划分一致
- DOTA 数据集需要按照 mmrotate 的格式组织（images 和 annfiles 分开）

## 关于 Oriented R-CNN

Oriented R-CNN 是一个经典的两阶段旋转目标检测器，主要特点：

1. **旋转 RPN**：使用旋转锚框和旋转 RoI 对齐
2. **旋转 RoI Head**：专门设计的旋转框回归头
3. **两阶段设计**：先提取候选区域，再精细分类和回归

该方法在 DOTA 数据集上取得了良好的性能，是旋转目标检测领域的经典基线方法。

## 参考论文

- **Oriented R-CNN**: [Oriented R-CNN for Object Detection](https://arxiv.org/abs/2108.07639)

