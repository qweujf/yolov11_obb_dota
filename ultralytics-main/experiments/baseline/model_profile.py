"""
Quick baseline model profiler for parameter count and GFLOPs.

运行:
    python experiments/baseline/model_profile.py

通过修改脚本顶部的常量，可切换不同的模型 YAML 或权重文件。
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 可配置项
# ============================================================================
# 支持填写 YAML（结构定义）或 PT（训练好的权重）；若同时存在同名 PT，会优先加载 PT。
MODEL_PATH = Path(
    r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\baseline\runs\obb\train\weights\best.pt"
)

# 计算 GFLOPs 时使用的输入尺寸（正方形）。可根据实际推理尺寸调整。
IMG_SIZE = 1024


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ 模型文件不存在: {MODEL_PATH}")

    print(f"📄 加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    core_model = getattr(model, "model", model)
    if hasattr(core_model, "info"):
        print(f"📐 计算参数量与 GFLOPs (imgsz={IMG_SIZE}) ...")
        core_model.info(verbose=True, imgsz=IMG_SIZE)
    else:
        print("⚠️ 当前模型实例不支持 info(imgsz=...)，仅输出基础信息。")
        model.info(verbose=True)


if __name__ == "__main__":
    main()

