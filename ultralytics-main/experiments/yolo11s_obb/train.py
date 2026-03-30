import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# 设置环境变量禁用自动下载（必须在导入 YOLO 之前）
# 如果你本地/远端已经缓存了权重文件，则不会触发下载；否则训练可能需要联网或先手动准备权重。
os.environ["YOLO_OFFLINE"] = "True"

# 将仓库根目录加入 sys.path，确保能导入本地 ultralytics-main/ultralytics
root_path = str(Path(__file__).resolve().parents[2])
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
    repo_root = Path(__file__).resolve().parents[2]

    # 实验目录（yolo11s_obb）
    exp_dir = Path(__file__).resolve().parent
    exp_config_path = exp_dir / "config.yaml"

    # 默认配置文件路径
    default_train_cfg_path = (repo_root / "configs" / "train" / "default.yaml").resolve()
    default_model_cfg_path = (repo_root / "configs" / "model" / "yolov11_obb.yaml").resolve()
    default_data_cfg_path = (repo_root / "configs" / "data" / "dota_obb.yaml").resolve()

    # 将 src 加入 sys.path 以支持自定义模块（如 research.*）
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

    # 4. 处理模型路径（优先使用本地权重；否则使用模型名让 YOLO 处理缓存）
    model_path: str = ""
    if "model" in exp_cfg and exp_cfg["model"]:
        # 实验配置里显式指定了 model
        model_path = str(Path(exp_cfg["model"]).expanduser().resolve())
        # 如果是本地 .pt 且存在，明确设置 pretrained=False（避免再次下载）
        if Path(model_path).suffix == ".pt" and Path(model_path).exists():
            cfg["pretrained"] = False
    else:
        # 默认使用 YOLO11s-OBB 预训练权重
        local_weights = repo_root / "yolo11s-obb.pt"
        if local_weights.exists():
            model_path = str(local_weights.resolve())
            cfg["pretrained"] = False
            print(f"✅ 使用本地预训练权重: {model_path}")
        else:
            # 依赖 YOLO 的缓存机制；如果缓存也没有且 YOLO_OFFLINE=True，则可能失败
            model_path = "yolo11s-obb.pt"
            cfg["pretrained"] = True
            print("✅ 使用 YOLO11s-OBB 模型名：yolo11s-obb.pt（若缓存存在则可离线加载）")

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

    # 7. 构建模型
    model = YOLO(model_path)

    # 8. 设置 data 路径（训练时必需）
    cfg["data"] = data_path

    # 9. 打印使用的关键配置（用于确认）
    print("\n" + "=" * 60)
    print("📋 训练配置摘要")
    print("=" * 60)
    key_params = ["epochs", "batch", "imgsz", "device", "lr0", "optimizer", "name"]
    for key in key_params:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")
    print("=" * 60 + "\n")

    # 10. 开始训练
    _ = model.train(**cfg)

    # 打印保存目录
    save_dir: Optional[Path] = None
    try:
        save_dir = Path(model.trainer.save_dir)  # type: ignore[attr-defined]
    except Exception:
        pass
    if save_dir:
        print(f"\n✅ 训练完成！结果保存在: {save_dir}")
    else:
        print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()

