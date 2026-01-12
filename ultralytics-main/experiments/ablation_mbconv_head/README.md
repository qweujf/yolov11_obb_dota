# MBConv 检测头消融实验

## 实验目的

验证不同 MBConv 检测头配置（SE 注意力模块的位置）对模型性能的影响。

## 实验设计

### 实验配置说明

本消融实验在 GhostNetV2 骨干网络的基础上，对比四种不同的 MBConv 检测头配置：

1. **MBConv（回归有SE）**: 分类分支无 SE，回归分支有 SE
2. **MBConv（分类有SE）**: 分类分支有 SE，回归分支无 SE
3. **MBConv（分类、回归均有SE）**: 分类和回归分支都有 SE
4. **MBConv（分类无SE，回归有SE）**: 分类分支无 SE，回归分支有 SE（与配置1相同，作为对照）

### 实验 1: MBConv（回归有SE）

- **目录**: `mbconv_reg_se/`
- **配置**: 
  - 分类分支：MBConv（无 SE，扩张系数 k=3）
  - 回归分支：MBConv（有 SE，扩张系数 k=3，仅替换前两层）

### 实验 2: MBConv（分类有SE）

- **目录**: `mbconv_cls_se/`
- **配置**: 
  - 分类分支：MBConv（有 SE，扩张系数 k=3）
  - 回归分支：MBConv（无 SE，扩张系数 k=3，仅替换前两层）

### 实验 3: MBConv（分类、回归均有SE）

- **目录**: `mbconv_both_se/`
- **配置**: 
  - 分类分支：MBConv（有 SE，扩张系数 k=3）
  - 回归分支：MBConv（有 SE，扩张系数 k=3，仅替换前两层）

### 实验 4: MBConv（分类无SE，回归有SE）

- **目录**: `mbconv_cls_no_se_reg_se/`
- **配置**: 
  - 分类分支：MBConv（无 SE，扩张系数 k=3）
  - 回归分支：MBConv（有 SE，扩张系数 k=3，仅替换前两层）
  - **说明**: 这是推荐的配置（与实验1相同）

## 对比指标

- **参数量** (Parameters)
- **计算量** (FLOPs)
- **检测精度** (mAP50, mAP50:95)
- **推理速度** (FPS)

## 运行实验

### 1. 运行 MBConv（回归有SE）实验

```bash
cd mbconv_reg_se
python train.py
```

### 2. 运行 MBConv（分类有SE）实验

```bash
cd mbconv_cls_se
python train.py
```

### 3. 运行 MBConv（分类、回归均有SE）实验

```bash
cd mbconv_both_se
python train.py
```

### 4. 运行 MBConv（分类无SE，回归有SE）实验

```bash
cd mbconv_cls_no_se_reg_se
python train.py
```

## 注意事项

1. **模型配置文件**: 需要确保以下配置文件存在：
   - `configs/model/yolo11-obb-ddbc-ghostnetv2-mbconv-reg-se.yaml`
   - `configs/model/yolo11-obb-ddbc-ghostnetv2-mbconv-cls-se.yaml`
   - `configs/model/yolo11-obb-ddbc-ghostnetv2-mbconv-both-se.yaml`
   - `configs/model/yolo11-obb-ddbc-ghostnetv2-mbconv-cls-no-se-reg-se.yaml`

2. **模块注册**: 确保 MBConv 模块已在 `ultralytics/nn/modules/__init__.py` 中注册

3. **预训练权重**: 建议使用 GhostNetV2 骨干网络的预训练权重进行初始化

## 实验结果

实验结果将保存在 `runs/obb/ablation_mbconv_head/` 目录下，包括：
- 训练日志
- 模型权重
- 验证结果
- 性能指标

