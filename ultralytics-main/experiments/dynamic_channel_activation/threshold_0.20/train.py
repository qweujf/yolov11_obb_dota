import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# 设置环境变量禁用自动下载（必须在导入 YOLO 之前）
os.environ["YOLO_OFFLINE"] = "True"

# 设置动态通道激活阈值
THRESHOLD = 0.20
os.environ["DYNAMIC_ACTIVATION_THRESHOLD"] = str(THRESHOLD)

root_path = str(Path(__file__).parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

from ultralytics import YOLO


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            return loaded if isinstance(loaded, dict) else {}
    except Exception as e:
        print(f"⚠️ 警告：加载配置文件 {config_path} 失败: {e}")
        return {}


def merge_configs(default_cfg: Dict[str, Any], override_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并配置，override_cfg 覆盖 default_cfg"""
    merged = default_cfg.copy()
    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    # 仓库根目录
    repo_root = Path(__file__).resolve().parents[3]
    
    # 实验目录
    exp_dir = Path(__file__).resolve().parent
    exp_config_path = exp_dir / "config.yaml"

    # 默认配置文件路径
    default_train_cfg_path = (repo_root / "configs" / "train" / "default.yaml").resolve()
    default_model_cfg_path = (repo_root / "configs" / "model" / "yolov11_obb.yaml").resolve()
    default_data_cfg_path = (repo_root / "configs" / "data" / "dota_obb.yaml").resolve()

    # 将 src 加入 sys.path 以支持自定义模块
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # 1. 加载默认训练配置
    print(f"📄 加载默认配置: {default_train_cfg_path}")
    default_cfg = load_yaml_config(default_train_cfg_path)
    
    # 2. 加载实验特定配置（覆盖）
    exp_cfg = {}
    if exp_config_path.exists():
        print(f"📄 加载实验配置: {exp_config_path}")
        exp_cfg_raw = load_yaml_config(exp_config_path)
        if "train" in exp_cfg_raw:
            exp_cfg = merge_configs(exp_cfg, exp_cfg_raw["train"])
        for key in exp_cfg_raw:
            if key != "train":
                exp_cfg[key] = exp_cfg_raw[key]
    
    # 3. 合并配置
    cfg = merge_configs(default_cfg, exp_cfg)
    
    # 4. 处理模型路径
    model_path: str = ""
    if "model" in exp_cfg and exp_cfg["model"]:
        model_cfg_path = Path(exp_cfg["model"])
        if model_cfg_path.is_absolute():
            model_path = str(model_cfg_path.expanduser().resolve())
        else:
            # 相对路径，相对于 repo_root 解析
            model_path = str((repo_root / model_cfg_path).resolve())
    else:
        local_weights = repo_root / "yolo11n-obb.pt"
        if local_weights.exists():
            model_path = str(local_weights.resolve())
            print(f"✅ 使用本地预训练权重: {model_path}")
        else:
            model_path = str(default_model_cfg_path)
            print(f"✅ 使用模型配置: {model_path}")
    
    if Path(model_path).suffix == ".pt" and Path(model_path).exists():
        cfg["pretrained"] = False
    
    # 5. 处理数据路径
    if "data" in exp_cfg and exp_cfg["data"]:
        data_cfg_path = Path(exp_cfg["data"])
        if data_cfg_path.is_absolute():
            data_path = str(data_cfg_path.expanduser().resolve())
        else:
            # 相对路径，相对于 repo_root 解析
            data_path = str((repo_root / data_cfg_path).resolve())
    else:
        data_path = str(default_data_cfg_path)
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"❌ 数据配置文件不存在: {data_path}")
    
    print(f"✅ 数据配置: {data_path}")
    
    # 6. 清理缓存
    data_cfg = load_yaml_config(Path(data_path))
    if data_cfg and "path" in data_cfg:
        data_root = Path(data_cfg["path"])
        if data_root.exists():
            cache_files = list(data_root.rglob("*.cache"))
            if cache_files:
                print(f"🧹 清理可能损坏的缓存文件...")
                for cache_file in cache_files:
                    try:
                        import numpy as np
                        np.load(str(cache_file), allow_pickle=True).item()
                    except (EOFError, OSError, ValueError):
                        cache_file.unlink()
                        print(f"  ✅ 已删除损坏的缓存: {cache_file.name}")
    
    # 7. 构建模型
    print(f"🔧 构建模型: {model_path}")
    print(f"🔧 动态通道激活阈值: τ = {THRESHOLD}")
    model = YOLO(model_path, task='obb')
    
    # 8. 设置 data 路径
    cfg["data"] = data_path
    
    # 9. 打印配置摘要
    print("\n" + "="*60)
    print("📋 训练配置摘要")
    print("="*60)
    print(f"  实验: 动态通道激活阈值 τ = {THRESHOLD}")
    key_params = ["epochs", "batch", "imgsz", "device", "lr0", "optimizer", "name"]
    for key in key_params:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")
    print("="*60 + "\n")
    
    # 10. 开始训练
    results = model.train(**cfg)

    save_dir: Optional[Path] = None
    try:
        save_dir = Path(model.trainer.save_dir)  # type: ignore[attr-defined]
    except Exception:
        pass
    if save_dir:
        print(f"\n✅ 训练完成，结果保存在: {save_dir}")
    else:
        print("Training finished.")


if __name__ == "__main__":
    main()

