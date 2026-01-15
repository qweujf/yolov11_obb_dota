#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO训练结果可视化脚本（优化版 - 支持GPU加速）
功能：一次性生成所有图像（损失曲线、混淆矩阵、PR曲线等）
使用方法：直接运行 python plot_result_img.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from scipy.ndimage import gaussian_filter1d

import os
import sys

# 设置环境变量
os.environ["YOLO_OFFLINE"] = "True"

root_path = str(Path(__file__).parents[1])
if root_path not in sys.path:
    sys.path.append(root_path)

from ultralytics import YOLO


# 设置matplotlib后端（适用于无GUI环境）
import matplotlib

matplotlib.use('Agg')

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
RUN_DIR = r"D:\code\yolov11_obb_dota\runs\obb\try"  # 例如: "runs/train/exp" 或 "/path/to/runs/train/exp"

# 模型权重路径（用于重新运行验证生成混淆矩阵和PR曲线等）
MODEL_PATH = r"D:\code\yolov11_obb_dota\runs\obb\mca_attention\weights\best.pt"  # 例如: "best.pt" 或 "/path/to/best.pt"

# 数据集配置文件路径
DATA_YAML = r"D:\code\yolov11_obb_dota\ultralytics-main\configs\data\dota_obb.yaml"  # 例如: "data/dota.yaml" 或 "/path/to/data.yaml"

# 任务类型: "detect", "obb", "segment", "pose", "classify"
TASK = "obb"


# GPU设备设置（加速验证）
DEVICE = "0"  # "0" 表示使用GPU 0, "cpu" 表示使用CPU, "0,1" 表示使用多GPU
BATCH_SIZE = 16  # 批次大小，根据GPU显存调整（越大越快，但需要更多显存）
HALF = True  # 是否使用FP16半精度（可以加速，但可能略微降低精度）
IMGSZ = 1024  # 图像尺寸（与训练时保持一致）

# 是否为分割任务
IS_SEGMENT = False

# 是否为姿态估计任务
IS_POSE = False

# 是否为分类任务
IS_CLASSIFY = False


# ===============================================================


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def plot_results(
        file: str = "results.csv",
        save_dir: Optional[Path] = None,
        segment: bool = False,
        pose: bool = False,
        classify: bool = False,
):
    """从results CSV文件绘制训练结果（损失曲线等）"""
    save_dir = Path(save_dir) if save_dir else Path(file).parent
    file_path = Path(file)

    if not file_path.exists():
        print(f"❌ 找不到文件: {file_path}")
        return False

    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [2, 5, 3, 4]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 8, 9, 12, 13]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 9, 10, 13, 14]
    else:  # 检测任务（detect/obb）
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 9, 10, 11, 7, 8]

    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))

    if not files:
        print(f"❌ 在 {save_dir} 中找不到 results*.csv 文件")
        return False

    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]

            for i, j in enumerate(index):
                if j >= len(s):
                    continue
                y = data.values[:, j].astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f"⚠️ 绘制 {f} 时出错: {e}")

    if len(files) > 0:
        ax[1].legend()
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()
        print(f"✅ 已保存损失曲线: {fname}")
        return True
    else:
        plt.close()
        return False


def regenerate_plots_from_model(
        model_path: str,
        data_yaml: str,
        save_dir: Path,
        task: str = "obb",
        device: str = "0",
        batch: int = 16,
        half: bool = True,
        imgsz: int = 1024,
):
    """通过重新运行验证来生成所有图像（混淆矩阵、PR曲线等）"""
    try:
        from ultralytics import YOLO

        # 检查路径
        model_path = Path(model_path)
        if not model_path.is_absolute():
            # 相对于run_dir的路径
            model_path = save_dir / model_path
            if not model_path.exists():
                # 尝试相对于脚本目录
                model_path = Path(__file__).parent / model_path

        data_yaml = Path(data_yaml)
        if not data_yaml.is_absolute():
            # 尝试相对于脚本目录
            data_yaml = Path(__file__).parent / data_yaml
            if not data_yaml.exists():
                # 尝试相对于run_dir
                data_yaml = save_dir.parent.parent / data_yaml

        if not model_path.exists():
            print(f"❌ 找不到模型文件: {model_path}")
            return False

        if not data_yaml.exists():
            print(f"❌ 找不到数据配置文件: {data_yaml}")
            return False

        print(f"🔄 加载模型: {model_path}")
        print(f"   GPU设备: {device}")
        print(f"   批次大小: {batch}")
        print(f"   半精度: {half}")
        print(f"   图像尺寸: {imgsz}")

        model = YOLO(str(model_path), task=task)

        print(f"🔄 运行验证并生成图像...")
        print(f"   数据集配置: {data_yaml}")
        print(f"   保存目录: {save_dir}")
        print()

        # 使用GPU和优化的参数加速验证
        results = model.val(
            data=str(data_yaml),
            plots=True,
            save_dir=str(save_dir),
            device=device,  # 指定GPU设备
            batch=batch,  # 批次大小
            half=half,  # FP16半精度
            imgsz=imgsz,  # 图像尺寸
            verbose=True,  # 显示详细信息
        )

        print()
        print(f"✅ 验证完成，所有图像已保存到: {save_dir}")
        return True
    except ImportError:
        print("❌ 需要安装 ultralytics 库: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ 生成图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent

    # 处理run_dir路径
    run_dir = Path(RUN_DIR)
    if not run_dir.is_absolute():
        # 先尝试相对于脚本目录
        run_dir = script_dir.parent / run_dir
        if not run_dir.exists():
            # 再尝试相对于当前工作目录
            run_dir = Path(RUN_DIR)

    if not run_dir.exists():
        print(f"❌ 目录不存在: {run_dir}")
        print(f"   请检查 RUN_DIR 配置: {RUN_DIR}")
        return

    print("=" * 60)
    print("YOLO训练结果可视化（GPU加速版）")
    print("=" * 60)
    print(f"📁 工作目录: {run_dir}")
    print()

    # 1. 绘制损失曲线 (results.png)
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        print("📊 [1/2] 绘制损失曲线...")
        success = plot_results(str(results_csv), save_dir=run_dir,
                               segment=IS_SEGMENT, pose=IS_POSE, classify=IS_CLASSIFY)
        if not success:
            print("⚠️ 损失曲线绘制失败")
        print()
    else:
        print(f"⚠️ 未找到 results.csv: {results_csv}")
        print()

    # 2. 重新运行验证生成所有图像（混淆矩阵、PR曲线等）
    print("🔄 [2/2] 重新运行验证生成所有图像...")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   数据配置: {DATA_YAML}")
    print(f"   任务类型: {TASK}")
    print(f"   GPU设备: {DEVICE}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   半精度: {HALF}")
    print(f"   图像尺寸: {IMGSZ}")
    print()

    success = regenerate_plots_from_model(
        MODEL_PATH,
        DATA_YAML,
        run_dir,
        TASK,
        device=DEVICE,
        batch=BATCH_SIZE,
        half=HALF,
        imgsz=IMGSZ,
    )

    if success:
        print()
        print("=" * 60)
        print("✅ 所有图像生成完成！")
        print("=" * 60)
        print(f"📁 图像保存位置: {run_dir}")
        print()
        print("生成的图像包括:")
        print("  - results.png (损失曲线)")
        print("  - confusion_matrix.png (混淆矩阵)")
        print("  - confusion_matrix_normalized.png (归一化混淆矩阵)")
        print("  - PR_curve.png (精确率-召回率曲线)")
        print("  - F1_curve.png (F1曲线)")
        print("  - P_curve.png (精确率曲线)")
        print("  - R_curve.png (召回率曲线)")
        print("  - val_batch*_labels.jpg (验证批次标签)")
        print("  - val_batch*_pred.jpg (验证批次预测)")
    else:
        print()
        print("=" * 60)
        print("⚠️ 部分图像生成失败，请检查配置参数")
        print("=" * 60)


if __name__ == "__main__":
    main()

