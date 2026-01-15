"""
双模型旋转目标检测对比脚本（已修复 OBB 几何畸变与路径管理）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import sys

# 设置环境变量
os.environ["YOLO_OFFLINE"] = "True"

root_path = str(Path(__file__).parents[1])
if root_path not in sys.path:
    sys.path.append(root_path)

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors

# =============================================================================
# 配置参数
# =============================================================================
IMAGE_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard\images\show")

# 模型配置保持不变
MODEL1_CONFIG = Path(r"D:\code\yolov11_obb_dota\ultralytics-main\configs\model\yolo11-obb-ddbc-ghostnetv2.yaml")
MODEL1_WEIGHTS = Path(
    r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\ablation_ghostnetv2_backbone\ghostnetv2\runs\obb\ablation_ghostnetv2_backbone\yolo_ddbc_ghostnetv2_backbone\weights\best.pt")
MODEL1_NAME = "baseline"

MODEL2_CONFIG = Path(r"D:\code\yolov11_obb_dota\ultralytics-main\configs\model\yolov11_obb.yaml")
MODEL2_WEIGHTS = Path(r"D:\code\yolov11_obb_dota\runs\obb\ral_loss_baseline\weights\best.pt")
MODEL2_NAME = "enhanced"

DEVICE = "cuda:0"
IMG_SIZE = 1024
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# 可视化参数
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# =============================================================================
# 核心功能修改
# =============================================================================

def annotate_obb_fixed(base_image: np.ndarray, result: Results) -> np.ndarray:
    """
    采用几何拟合方式修复非矩形 OBB 问题
    """
    annotated = base_image.copy()
    if result.obb is None or len(result.obb) == 0:
        return annotated

    # 获取所有角点
    polygons = result.obb.xyxyxyxy.cpu().numpy()
    confidences = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy().astype(int)
    name_map = result.names or {}

    for idx, poly in enumerate(polygons):
        # --- 核心修复：重新计算最小外接矩形以保证平行性 ---
        # poly 形状为 (4, 2)
        rect = cv2.minAreaRect(poly.astype(np.float32))
        box = cv2.boxPoints(rect)  # 获得 4 个精确的矩形顶点
        box = np.int64(box)  # 使用 int64 保证坐标精度

        cls_id = int(classes[idx])
        cls_name = name_map.get(cls_id, f"class_{cls_id}")
        color = tuple(int(c) for c in colors(cls_id, True))

        # 绘制严格矩形的旋转框
        cv2.drawContours(annotated, [box], 0, color, BOX_THICKNESS, cv2.LINE_AA)

        # 绘制文本标签
        label = f"{cls_name}:{confidences[idx]:.2f}"
        text_x, text_y = box[0][0], box[0][1] - 5
        cv2.putText(annotated, label, (int(text_x), int(text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

    return annotated


def main():
    # 1. 自动创建 result 文件夹
    output_dir = IMAGE_DIR / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型逻辑 (省略重复代码，参考原脚本)
    model1 = YOLO(str(MODEL1_WEIGHTS))  # 简化加载，通常.pt包含足够信息
    model2 = YOLO(str(MODEL2_WEIGHTS))

    image_files = [f for f in IMAGE_DIR.iterdir() if f.is_file() and f.suffix in VALID_IMAGE_EXTENSIONS]

    print(f"🚀 开始处理，结果将保存至: {output_dir}")

    for image_path in tqdm(image_files, desc="Processing"):
        image = cv2.imread(str(image_path))

        # 处理模型 1
        res1 = model1.predict(source=str(image_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        ann1 = annotate_obb_fixed(image, res1)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_{MODEL1_NAME}{image_path.suffix}"), ann1)

        # 处理模型 2
        res2 = model2.predict(source=str(image_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        ann2 = annotate_obb_fixed(image, res2)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_{MODEL2_NAME}{image_path.suffix}"), ann2)

    print(f"\n✅ 处理完成！请查看目录: {output_dir}")


if __name__ == "__main__":
    main()