# SFE_DRB 模块说明（Small-object Feature Enhancement with Dilated-Residual Bridge）

## 1. 设计目标
- 面向 DOTA 小目标检测，增强浅层高分辨率细节在中高层的表达；
- 以卷积为核心（非注意力），便于稳定训练与推理；
- 可无缝替换 Backbone 中的 MCA 模块，Head 不做改动（C3k2 保持）。

## 2. 模块结构
SFE_DRB = DDP（Dilated-Depthwise Pyramid） + GFF（Gated Fine-grained Fusion） + 轻量残差细化

- DDP：并联的深度可分离卷积，膨胀率 r∈{1,2,3}，捕获不同感受野上下文；随后 1×1 Conv 融合。
- GFF：引入高分辨率分支（来自更浅层特征），做门控加权融合（GAP→1×1→Sigmoid 生成门控系数 g）。
- 轻量细化：LightConv(1×1→DWConv 3×3) 细化，最后 1×1 输出；与输入同形状时采用残差。

结构示意（逻辑）：
```
          ┌───────────── DDP ─────────────┐
输入 x →  │ DWConv(d=1)  DWConv(d=2)  DWConv(d=3) → Concat → 1×1 → y
          └────────────────────────────────┘
高分辨率 hr(来自浅层，对齐到 x 尺寸) → 1×1 对齐 → hr_a
门控 g = Sigmoid(GAP([y, hr_a]))   （逐通道/逐分支门控的轻量实现）
融合 y ← y + g * hr_a
细化 y ← LightConv(1×1→DWConv3×3) → 1×1 → 输出（若维度一致，y += x 残差）
```

## 3. 在 YOLO 模型中的放置与替换
- 替换 Backbone 中原有的 MCA 模块，Head 不变。
- 新模型配置：`configs/model/yolo11-obb-sfe-drb.yaml`
- 放置索引与高分辨率分支（hr）来源：
  - 3：`SFE_DRB([上一路输出, idx 1(P2/4)])`
  - 6：`SFE_DRB([上一路输出, idx 4(P3/8)])`
  - 9：`SFE_DRB([上一路输出, idx 4(P3/8)])`
  - 14：`SFE_DRB([上一路输出, idx 7(P4/16)])`

YAML 片段（示意）：
```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]     # 0
  - [-1, 1, Conv, [128, 3, 2]]    # 1 (P2/4)
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [[-1, 1], 1, SFE_DRB, [256, True]]   # 3  使用 idx 1 作为 hr
  - [-1, 1, Conv, [256, 3, 2]]    # 4 (P3/8)
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [[-1, 4], 1, SFE_DRB, [512, True]]   # 6  使用 idx 4 作为 hr
  ...
```

## 4. 与原模型的差异
- 被替换部分：Backbone 的 MCA 模块；
- 不改动的部分：Head（C3k2 结构不变）、检测头与损失函数；
- 新增高分辨率门控融合，重点强化小目标的细节表达与召回。

## 5. 预期效果
- 小目标 mAP50：提升 2.0 ~ 3.5%（需以实验为准）；
- 整体 mAP50：有望提升 1.0 ~ 2.0%（随训练与数据分布波动）；
- 计算量与标准卷积同量级，明显低于注意力型替换。

## 6. 代码位置
- 模块实现：`src/research/nn/modules/sfe_drb.py`
- 模块导出：`src/research/nn/modules/__init__.py`（SFE_DRB）
- 框架注册：`ultralytics/nn/modules/__init__.py`、`ultralytics/nn/tasks.py`
- 模型配置：`configs/model/yolo11-obb-sfe-drb.yaml`
- 实验目录：`experiments/sfe_drb/`

---

## 7. 最小化验证（Minimal Sanity Check）
目标：不依赖数据集与训练，快速确认“可构建、可前向、维度匹配”。

步骤A：模块级验证
1) 构造 x、hr 张量（如 x=[1,256,80,80], hr=[1,256,160,160]），验证 SFE_DRB 可前向、自动对齐 hr 尺寸；
2) 断言输出形状与 x 相同。

步骤B：整模型级验证
1) 使用 `YOLO('configs/model/yolo11-obb-sfe-drb.yaml')` 构建模型；
2) `model.model.info()` 打印结构；
3) 随机张量前向 `model.model(torch.randn(1,3,640,640))`，仅检查能跑通到 Detect/OBB Head。

你可以直接运行 `experiments/sfe_drb/test_sfe_drb.py` 完成上述两个检查。


