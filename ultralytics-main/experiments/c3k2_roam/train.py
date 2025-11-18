import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量禁用自动下载（必须在导入 YOLO 之前�?
os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            return loaded if isinstance(loaded, dict) else {}
    except Exception as e:
        print(f"⚠️ 加载配置失败 {config_path}: {e}")
        return {}


def merge_configs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = a.copy()
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(__file__).resolve().parent
    exp_cfg_path = exp_dir / "config.yaml"

    default_train_cfg = load_yaml_config(repo_root / "configs" / "train" / "default.yaml")
    exp_cfg_raw = load_yaml_config(exp_cfg_path)

    # 提取 weights，并从训练参数剔�?
    weights_path = None
    if "weights" in exp_cfg_raw and exp_cfg_raw["weights"]:
        weights_path = Path(exp_cfg_raw["weights"]).expanduser()
        if not weights_path.is_absolute():
            weights_path = repo_root / weights_path
        del exp_cfg_raw["weights"]

    # 合并 train 子配�?
    exp_train_cfg = exp_cfg_raw.get("train", {})
    cfg = merge_configs(default_train_cfg, exp_train_cfg)

    # 模型/数据路径
    model_path = str((repo_root / exp_cfg_raw.get("model", "configs/model/yolo11-obb-sfe-drb.yaml")).resolve())
    data_path = str((repo_root / exp_cfg_raw.get("data", "configs/data/dota_obb.yaml")).resolve())
    cfg["data"] = data_path

    # 支持 research 路径
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # 构建模型
    model = YOLO(model_path)

    # 加载本地权重（如指定�?
    if weights_path and weights_path.exists():
        print(f"�?加载本地权重: {weights_path}")
        try:
            model.load(str(weights_path))
            cfg["pretrained"] = False
        except Exception as e:
            print(f"⚠️ 本地权重加载失败: {e}，将从头训练")

    # 打印关键训练参数
    print("\n============== 训练配置摘要 ==============")
    for k in ["epochs", "batch", "imgsz", "device", "lr0", "optimizer", "name"]:
        if k in cfg:
            print(f"{k}: {cfg[k]}")
    print("========================================\n")

    results = model.train(**cfg)
    try:
        save_dir: Optional[Path] = Path(model.trainer.save_dir)  # type: ignore[attr-defined]
        print(f"Run saved to {save_dir}")
    except Exception:
        pass


if __name__ == "__main__":
    main()


