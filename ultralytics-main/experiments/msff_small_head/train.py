import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# 禁用自动下载（需在导入 YOLO 之前设置）
os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """安全加载 YAML 配置文件。不存在或解析失败时返回空字典。"""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        print(f"⚠️ 无法解析配置 {config_path}: {exc}")
        return {}


def merge_configs(default_cfg: Dict[str, Any], override_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并配置；override_cfg 的值将覆盖默认配置。"""
    merged = default_cfg.copy()
    for key, value in override_cfg.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = Path(__file__).resolve().parent
    exp_cfg_path = exp_dir / "config.yaml"

    default_train_cfg = repo_root / "configs" / "train" / "default.yaml"
    default_model_cfg = repo_root / "configs" / "model" / "yolov11_obb.yaml"
    default_data_cfg = repo_root / "configs" / "data" / "dota_obb.yaml"

    # 将 src 目录加入 sys.path 以便载入 research.* 自定义模块
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    print(f"📄 加载默认训练配置: {default_train_cfg}")
    base_cfg = load_yaml_config(default_train_cfg)

    exp_cfg = {}
    if exp_cfg_path.exists():
        print(f"📄 加载实验配置: {exp_cfg_path}")
        exp_cfg_raw = load_yaml_config(exp_cfg_path)
        if "train" in exp_cfg_raw:
            exp_cfg = merge_configs(exp_cfg, exp_cfg_raw["train"])
        for key, value in exp_cfg_raw.items():
            if key != "train":
                exp_cfg[key] = value

    # 合并配置
    train_cfg = merge_configs(base_cfg, exp_cfg)

    # 模型配置：优先使用实验中显式指定的 MSFF 模型
    if "model" in exp_cfg and exp_cfg["model"]:
        model_path = (repo_root / exp_cfg["model"]).resolve()
        print(f"✅ 使用 MSFF 模型配置: {model_path}")
    else:
        model_path = default_model_cfg.resolve()
        print(f"⚠️ 未指定模型，回退到默认: {model_path}")
    train_cfg.pop("model", None)

    # 数据配置
    if "data" in exp_cfg and exp_cfg["data"]:
        data_path = (repo_root / exp_cfg["data"]).resolve()
    else:
        data_path = default_data_cfg.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"❌ 数据配置不存在: {data_path}")
    print(f"✅ 数据配置: {data_path}")

    # 清理可能损坏的缓存文件
    data_cfg = load_yaml_config(data_path)
    if data_cfg and "path" in data_cfg:
        data_root = Path(data_cfg["path"])
        if data_root.exists():
            cache_files = list(data_root.rglob("*.cache"))
            if cache_files:
                print("🧹 检查缓存文件...")
                for cache_file in cache_files:
                    try:
                        import numpy as np

                        np.load(str(cache_file), allow_pickle=True).item()
                    except (EOFError, OSError, ValueError):
                        cache_file.unlink()
                        print(f"  ✅ 已删除损坏缓存: {cache_file.name}")

    model = YOLO(str(model_path))
    train_cfg["data"] = str(data_path)

    print("\n" + "=" * 60)
    print("📋 MSFF Small-Head 训练配置摘要")
    print("=" * 60)
    for key in ["epochs", "batch", "imgsz", "device", "lr0", "optimizer", "name"]:
        if key in train_cfg:
            print(f"  {key}: {train_cfg[key]}")
    print("=" * 60 + "\n")

    results = model.train(**train_cfg)

    save_dir: Optional[Path] = None
    try:
        save_dir = Path(model.trainer.save_dir)  # type: ignore[attr-defined]
    except Exception:
        pass
    if save_dir:
        print(f"Run saved to {save_dir}")
    else:
        print("Training finished.")


if __name__ == "__main__":
    main()


