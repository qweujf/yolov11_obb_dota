import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# 确保项目根路径在 sys.path 中
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# 禁用自动下载
os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO  # noqa: E402


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


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = base.copy()
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    exp_cfg_path = exp_dir / "config.yaml"

    default_train_cfg = load_yaml_config(repo_root / "configs" / "train" / "default.yaml")
    exp_cfg_raw = load_yaml_config(exp_cfg_path)

    # 权重路径（若提供则加载）
    weights_path = None
    if "weights" in exp_cfg_raw and exp_cfg_raw["weights"]:
        weights_path = Path(exp_cfg_raw["weights"]).expanduser()
        if not weights_path.is_absolute():
            weights_path = repo_root / weights_path
        del exp_cfg_raw["weights"]

    # 合并训练配置
    exp_train_cfg = exp_cfg_raw.get("train", {})
    train_cfg = merge_configs(default_train_cfg, exp_train_cfg)

    # 模型、数据路径
    model_path = str((repo_root / exp_cfg_raw.get("model", "configs/model/yolo11-obb-sfe-drb.yaml")).resolve())
    data_path = str((repo_root / exp_cfg_raw.get("data", "configs/data/dota_obb.yaml")).resolve())
    train_cfg["data"] = data_path

    # research 模块路径
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # 构建模型
    model = YOLO(model_path)

    # 加载本地预训练权重
    if weights_path and weights_path.exists():
        print(f"✅ 加载本地权重: {weights_path}")
        try:
            model.load(str(weights_path))
            train_cfg["pretrained"] = False
        except Exception as e:
            print(f"⚠️ 本地权重加载失败: {e}，将从头训练")

    # 输出关键训练参数
    print("\n============== 训练配置摘要 ==============")
    for k in ["epochs", "batch", "imgsz", "device", "lr0", "optimizer", "name"]:
        if k in train_cfg:
            print(f"{k}: {train_cfg[k]}")
    print("========================================\n")

    results = model.train(**train_cfg)
    try:
        save_dir: Optional[Path] = Path(model.trainer.save_dir)  # type: ignore[attr-defined]
        print(f"Run saved to {save_dir}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

