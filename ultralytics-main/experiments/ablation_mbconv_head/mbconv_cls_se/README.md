# MBConv（分类有SE）检测头实验

## 实验说明

本实验使用 GhostNetV2 骨干网络，检测头采用 MBConv 模块，配置如下：

- **分类分支**: MBConv（有 SE 注意力模块，扩张系数 k=3）
- **回归分支**: MBConv（无 SE 注意力模块，扩张系数 k=3，仅替换前两层标准卷积）

## 模型组成

- **Backbone**: GhostNetV2（GhostConv + C3Ghost + DFC Attention）
- **Neck**: DDPG + BiFPN_DWC
- **Head**: MBConv 检测头（分类有SE，回归无SE）
- **Loss**: CSL_ProbIoU

## 运行

```bash
cd ultralytics-main/experiments/ablation_mbconv_head/mbconv_cls_se
python train.py
```

## 配置文件说明

- `config.yaml`: 实验配置，指定使用的模型配置文件
- `train.py`: 训练脚本，自动加载配置并开始训练

## 模型配置文件

模型配置文件位于：`configs/model/yolo11-obb-ddbc-ghostnetv2-mbconv-cls-se.yaml`

**注意**: 如果该配置文件不存在，需要先创建包含 MBConv 检测头（分类有SE）的配置文件。

