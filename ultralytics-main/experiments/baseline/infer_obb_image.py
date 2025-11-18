"""
Baseline inference helper for drawing YOLO11-OBB rotated boxes on a single image.

所有可调参数都集中在脚本顶部的常量中，无需命令行输入。

运行方式:
    python experiments/baseline/infer_obb_image.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors

# =============================================================================
# 可配置参数（根据实际情况修改）
# =============================================================================
MODEL_WEIGHTS = Path(
    # r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\baseline\runs\obb\train\weights\best.pt"
    r"D:\code\yolov11_obb_dota\runs\obb\sfe_drb\weights\best.pt"
)
# INPUT_IMAGE = Path(r"D:\code\yolov11_obb_dota\zuhe\try_split\images\val\P0000__1__0___0.jpg")
INPUT_IMAGE = Path(r"D:\code\yolov11_obb_dota\zuhe\raw\images\train\P0002.png")
OUTPUT_IMAGE = Path(r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\baseline\infer_vis_sfe_drb.jpg")

DEVICE = "cuda:0"  # 例如 "cuda:0" / "cpu" / "mps"
IMG_SIZE = 2557
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# 可视化控制
DRAW_LABEL_TEXT = True          # 是否在图像上写类别/置信度
LABEL_WITH_CONFIDENCE = True    # 文本中是否包含置信度
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1


# =============================================================================
# 推理与可视化逻辑
# =============================================================================
def _to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    """安全地将张量或数组转换为 numpy.ndarray。"""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到输入图像: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"❌ 读取图像失败: {path}")
    return image


def run_inference(model: YOLO, image_path: Path) -> Results:
    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        task="obb",
        verbose=True,
        save=False,
        show=False,
    )
    if not results:
        raise RuntimeError("❌ 模型未返回结果，请检查输入。")
    return results[0]


def annotate_obb(
    base_image: np.ndarray,
    result: Results,
    draw_text: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    将旋转框绘制到图像上，并返回检测信息列表。
    """
    annotated = base_image.copy()
    detections: List[Dict] = []

    if result.obb is None or len(result.obb) == 0:
        return annotated, detections

    polygons = _to_numpy(result.obb.xyxyxyxy)  # (N, 4, 2)
    confidences = _to_numpy(result.obb.conf)
    classes = _to_numpy(result.obb.cls).astype(int)
    name_map = result.names or {}

    for idx, poly in enumerate(polygons):
        pts = poly.astype(np.int32).reshape(-1, 2)
        cls_id = int(classes[idx])
        cls_name = name_map.get(cls_id, f"class_{cls_id}")
        conf = float(confidences[idx])
        color = tuple(int(c) for c in colors(cls_id, True))

        cv2.polylines(
            annotated,
            [pts.reshape(-1, 1, 2)],
            isClosed=True,
            color=color,
            thickness=BOX_THICKNESS,
            lineType=cv2.LINE_AA,
        )

        if draw_text:
            label = f"{cls_name}:{conf:.2f}" if LABEL_WITH_CONFIDENCE else cls_name
            text_x = int(pts[:, 0].min())
            text_y = int(pts[:, 1].min()) - 5
            text_y = max(text_y, 10)
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        detections.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "polygon": pts.tolist(),
            }
        )

    return annotated, detections


def main() -> None:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"❌ 找不到模型权重: {MODEL_WEIGHTS}")

    image = load_image(INPUT_IMAGE)
    print(f"✅ 输入图像: {INPUT_IMAGE}  shape={image.shape}")

    model = YOLO(str(MODEL_WEIGHTS))
    print(f"✅ 已加载模型: {MODEL_WEIGHTS.name}")

    result = run_inference(model, INPUT_IMAGE)
    annotated, dets = annotate_obb(image, result, draw_text=DRAW_LABEL_TEXT)

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), annotated)

    print(f"✅ 保存推理结果: {OUTPUT_IMAGE}")
    if dets:
        print(f"🔎 共检测到 {len(dets)} 个目标，前5条如下：")
        for item in dets[:5]:
            print(
                f"  - {item['class_name']} (id={item['class_id']}), "
                f"conf={item['confidence']:.2f}, poly={item['polygon']}"
            )
    else:
        print("ℹ️ 未检测到任何目标。")


if __name__ == "__main__":
    main()

