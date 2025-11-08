# YOLOv11-OBB DOTA 工程结构说明与修改规则

本文档说明当前 YOLOv11-OBB DOTA 工程的结构与后续修改规则，请遵循执行。

---

## 工程结构说明

- `ultralytics-main/`

  - `src/`

    - `ultralytics/`：官方核心框架（**不要动**，除非同步官方更新）

    - `research/`：你自定义的可复用模块（模型、损失、头、数据增强、工具等）

  - `configs/`

    - `train/`、`data/`、`model/` 等目录：全局默认配置

    - `train/default.yaml`：训练默认参数（只需在此处维护通用默认值）

  - `experiments/`

    - 每个实验一个子目录（例如 `baseline/`）

    - 子目录内：

      - `config.yaml`：只写与全局默认不同的参数

      - `train.py` / `eval.py` / `analyze.py` 等脚本：直接运行，无命令行参数

      - `runs/`：训练日志、结果（`weights/`、`results.csv`、`results.png` 等）

  - `scripts/`：通用工具脚本（数据切分、批量评估、导出模型等）

  - `data_processing/`：旧版或第三方工具脚本（按需调用，不直接修改）

  - 其他目录（`docs/`、`tests/` 等）：保持原样

---

## 修改规则（请提醒我遵守）

1. **训练脚本执行方式**  
   所有训练/评估脚本默认从 `configs/train/default.yaml` 加载配置，再用当前实验目录 `config.yaml` 覆盖差异化参数；脚本内不再解析命令行参数，直接 `python train.py` 即可。

2. **实验配置管理**  
   - 只在 `experiments/<name>/config.yaml` 写与默认不同的参数；若全部使用默认值，可以留空。

   - 默认配置更新只改 `configs/train/default.yaml`；避免在多个地方维护重复参数。

3. **数据脚本与处理流程**

   - 数据切分、过滤脚本集中放在 `scripts/`。

   - 若要调整算法逻辑（如 `split_dota_v2_standard.py`），先确认与训练数据一致，再更新脚本并记录修改原因。

4. **模型/库代码**  
   - 自定义或改进功能写到 `src/research/` 下；便于复用与实验隔离。

   - 不直接修改 `src/ultralytics/`（除非同步官方修复或临时热修补，且要详细标注）。

5. **实验输出**  
   - 训练输出统一存到 `experiments/<name>/runs/`。  

   - 输出超过预期的中间文件（额外 `.csv`、脚本等）使用后记得删除或移走，保持目录干净。

6. **文档与提醒**  
   - 若新增脚本或配置规则，请同步更新该说明文件或 README。  

   - 后续提到"按既定规则修改"时，即按照此文档执行。

---

**注意**：后续提到"按规则来"时，将优先遵循此文档。

