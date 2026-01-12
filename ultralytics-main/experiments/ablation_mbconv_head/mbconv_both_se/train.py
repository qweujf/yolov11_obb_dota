import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# 设置环境变量禁用自动下载（必须在导入 YOLO 之前）
os.environ["YOLO_OFFLINE"] = "True"
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
        # 如果实验配置中有 train 子字典，提取出来合并到顶层
        if "train" in exp_cfg_raw:
            exp_cfg = merge_configs(exp_cfg, exp_cfg_raw["train"])
        # 其他顶层键也合并（如 model, data）
        for key in exp_cfg_raw:
            if key != "train":
                exp_cfg[key] = exp_cfg_raw[key]
    
    # 3. 合并配置（实验配置覆盖默认配置）
    cfg = merge_configs(default_cfg, exp_cfg)
    
    # 4. 处理模型路径（优先使用实验配置中的模型）
    model_path: str = ""
    if "model" in exp_cfg and exp_cfg["model"]:
        # 实验配置中指定了模型
        model_path = str((repo_root / exp_cfg["model"]).resolve())
        print(f"✅ 使用实验模型配置: {model_path}")
    else:
        # 默认使用本地预训练权重
        local_weights = repo_root / "yolo11n-obb.pt"
        if local_weights.exists():
            model_path = str(local_weights.resolve())
            print(f"✅ 使用本地预训练权重: {model_path}")
        else:
            model_path = str(default_model_cfg_path)
            print(f"✅ 使用默认模型配置: {model_path}")
    
    # 如果使用本地 .pt 文件，明确设置 pretrained=False（避免再次下载）
    if Path(model_path).suffix == ".pt" and Path(model_path).exists():
        cfg["pretrained"] = False
    
    # 5. 处理数据路径
    if "data" in exp_cfg and exp_cfg["data"]:
        data_path = str(Path(exp_cfg["data"]).expanduser().resolve())
    else:
        data_path = str(default_data_cfg_path)
    
    # 确保数据路径存在
    if not Path(data_path).exists():
        raise FileNotFoundError(f"❌ 数据配置文件不存在: {data_path}")
    
    print(f"✅ 数据配置: {data_path}")
    
    # 6. 清理可能损坏的缓存文件
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
    
    # 7. 构建模型（明确指定任务类型为OBB）
    model = YOLO(model_path, task='obb')
    
    # 8. 设置 data 路径（训练时必需，必须在构建模型之后）
    cfg["data"] = data_path
    
    # 9. 打印使用的关键配置（用于确认）
    print("\n" + "="*60)
    print("📋 MBConv 检测头（分类、回归均有SE）训练配置摘要")
    print("="*60)
    print("🔧 模型组件:")
    print("  - Backbone: GhostNetV2（GhostConv + C3Ghost + DFC Attention）")
    print("  - Neck: DDPG + BiFPN_DWC（保持不变）")
    print("  - Head: MBConv 检测头（分类有SE，回归有SE）")
    print("  - Loss: CSL_ProbIoU（保持不变）")
    print("="*60)
    print("📐 检测头配置:")
    print("  - 分类分支: MBConv（有 SE，扩张系数 k=3）")
    print("  - 回归分支: MBConv（有 SE，扩张系数 k=3，仅替换前两层）")
    print("="*60)
    key_params = ["epochs", "batch", "imgsz", "device", "lr0", "optimizer", "name"]
    for key in key_params:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")
    print("="*60 + "\n")
    
    # 10. 开始训练（所有参数从配置读取）
    results = model.train(**cfg)

    # 打印保存目录与关键结果摘要
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

