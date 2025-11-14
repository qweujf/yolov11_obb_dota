import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO  # noqa: E402


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️ 加载配置失败 {path}: {e}")
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

    default_train = load_yaml_config(repo_root / "configs" / "train" / "default.yaml")
    exp_cfg = load_yaml_config(exp_cfg_path)

    weights_path = None
    if exp_cfg.get("weights"):
        weights_path = Path(exp_cfg["weights"]).expanduser()
        if not weights_path.is_absolute():
            weights_path = repo_root / weights_path
        del exp_cfg["weights"]

    train_override = exp_cfg.get("train", {})
    train_cfg = merge_configs(default_train, train_override)

    model_path = str((repo_root / exp_cfg.get("model", "configs/model/yolo11-obb-sfe-drb.yaml")).resolve())
    data_path = str((repo_root / exp_cfg.get("data", "configs/data/dota_obb.yaml")).resolve())
    train_cfg["data"] = data_path

    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    model = YOLO(model_path)

    if weights_path and weights_path.exists():
        print(f"✅ 加载本地权重: {weights_path}")
        try:
            model.load(str(weights_path))
            train_cfg["pretrained"] = False
        except Exception as e:
            print(f"⚠️ 本地权重加载失败: {e}，将从头训练")

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

