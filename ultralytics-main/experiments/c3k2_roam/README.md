# C3k2_ROAM 实验

## 目录结构

- `config.yaml`：仅记录相对默认配置的差异（模型路径、run 名称等）。
- `train.py`：遵循“默认配置 + 实验覆盖”规则，直接 `python train.py` 即可训练。
- `test_c3k2_roam.py`：最小化验证脚本，快速检查模块与整网可否顺利前向。
- `runs/`：训练输出目录（训练后自动生成）。

## 使用方式

```bash
cd ultralytics-main/experiments/c3k2_roam
python test_c3k2_roam.py   # 可选，快速 sanity check
python train.py            # 启动训练
```

若需修改训练超参或本地权重，编辑 `config.yaml`；通用默认值位于 `configs/train/default.yaml`。
# C3k2_ROAM 实验

## 目录结构

- `config.yaml`：仅包含相对默认配置的差异参数（模型路径、run 名称等）。
- `train.py`：按既定流程加载 `configs/train/default.yaml`，再用本目录配置覆盖，直接 `python train.py` 即可启动训练。
- `test_c3k2_roam.py`：最小化验证脚本，快速检查 `C3k2_ROAM` 模块和整网构建是否正常。
- `runs/`：训练输出目录（训练后自动创建，含 weights、results 等）。

## 使用步骤

```bash
cd ultralytics-main/experiments/c3k2_roam
python test_c3k2_roam.py   # 可选，快速 sanity check
python train.py           # 启动训练
```

如需修改训练超参或权重路径，在 `config.yaml` 中按需覆盖；全局默认参数位于 `configs/train/default.yaml`。


