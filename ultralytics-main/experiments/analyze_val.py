#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集图片贡献度分析脚本（OBB + ProbIoU，官方同款度量）

功能：
- 对验证集中的每一张图片，基于官方 batch_probiou（xywhr 概率 IoU）统计：
  - TP（正确检测数）
  - FP（误检数）
  - FN（漏检数）
  - Precision / Recall / F1
- 计算“贡献度”（F1，越低说明越拖后腿），导出按 F1 从低到高排序的 CSV。

依赖：
- 使用 ultralytics 自带的 ProbIoU，无需 shapely。
- 仅需要 ultralytics 自己的依赖（包含 OpenCV 等）。

使用：
1. 修改【配置区域】里的路径和参数；
2. 运行：python analyze_obb_val_contrib_probiou.py
"""

import os
import sys
import math
import yaml
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch

# ================= 环境与导入 =================

os.environ["YOLO_OFFLINE"] = "True"

root_path = str(Path(__file__).parents[1])
if root_path not in sys.path:
    sys.path.append(root_path)

from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.metrics import batch_probiou  # 官方 ProbIoU

warnings.filterwarnings("ignore")


# ==================== 配置区域（根据自己环境修改） ====================

# 模型权重（obb 模型）
MODEL_PATH = r"D:\code\yolov11_obb_dota\runs\obb\ral_loss_baseline\weights\best.pt"

# 数据集配置 yaml（和训练/验证时用的是同一个）
DATA_YAML = r"D:\code\yolov11_obb_dota\ultralytics-main\configs\data\dota_val.yaml"

# 验证集使用 data.yaml 里的哪个 split（一般是 'val'）
DATA_SPLIT = "val"

# YOLO 任务类型
TASK = "obb"

# 推理参数
DEVICE = "0"          # "0" 用 GPU0，"cpu" 则用 CPU
IMGSZ = 1024          # 与训练时保持一致
CONF_THRES = 0.001    # 与验证时一致（默认 0.001）
IOU_THRES = 0.5       # ProbIoU 阈值（判定 TP 的标准）

# 最多统计多少张验证图像（None 表示全部）
MAX_IMAGES = 5000      # 比如只统计前 500 张；设为 None 表示全量

# 输出 CSV 路径
OUTPUT_CSV = r"D:\code\yolov11_obb_dota\val_image_contrib_obb_probiou.csv"

# =====================================================================


def load_data_paths(data_yaml: str, split: str) -> Tuple[Path, Path, List[Path]]:
    """
    解析 data.yaml，返回：
    - images_dir: 验证集 images 目录
    - labels_dir: 验证集 labels 目录
    - image_files: 验证集所有图片路径列表
    """
    data_yaml_path = Path(data_yaml)
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    path_root = Path(data.get("path", data_yaml_path.parent))
    split_rel = Path(data[split])  # 如 'images/val'

    # Ultralytics 默认：images/... 对应 labels/... 目录
    if "images" in str(split_rel):
        images_dir = (path_root / split_rel).resolve()
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    else:
        # 已经是具体目录
        images_dir = (path_root / split_rel).resolve()
        labels_dir = images_dir.parent / "labels" / images_dir.name

    if not images_dir.exists():
        raise FileNotFoundError(f"验证集 images 目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"验证集 labels 目录不存在: {labels_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])
    if not image_files:
        raise RuntimeError(f"在 {images_dir} 中未找到任何图片")

    return images_dir, labels_dir, image_files


def load_obb_labels_xywhr(label_file: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从标签文件读取 OBB，并转换为官方使用的 xywhr 格式。

    支持格式：
    - class x1 y1 x2 y2 x3 y3 x4 y4 （归一化四点坐标，常见 DOTA YOLO OBB）

    返回：
        gt_xywhr: (N, 5) [cx,cy,w,h,angle] 像素坐标
        gt_cls:   (N,)   类别 id
    """
    if not label_file.exists():
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    quads_norm = []
    cls_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                continue   # 非四点格式暂时忽略
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:9]))  # x1 y1 ... x4 y4 (归一化)
            quads_norm.append(coords)
            cls_list.append(cls)

    if not quads_norm:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    quads_norm = np.array(quads_norm, dtype=np.float32)  # (N,8)

    # 归一化 -> 像素
    xs = quads_norm[:, 0::2] * img_w
    ys = quads_norm[:, 1::2] * img_h
    quads_xy = np.empty_like(quads_norm)
    quads_xy[:, 0::2] = xs
    quads_xy[:, 1::2] = ys  # (N,8)

    # 使用官方工具转换为 xywhr
    gt_xywhr = ops.xyxyxyxy2xywhr(torch.from_numpy(quads_xy)).numpy()  # (N,5)
    gt_cls = np.array(cls_list, dtype=np.int64)
    return gt_xywhr.astype(np.float32), gt_cls


def match_rotated_prob_iou(
    preds_xywhr: np.ndarray,
    preds_cls: np.ndarray,
    preds_conf: np.ndarray,
    gts_xywhr: np.ndarray,
    gts_cls: np.ndarray,
    iou_thres: float = 0.5,
) -> Tuple[int, int, int]:
    """
    基于 ProbIoU 的贪心匹配：
    - 对预测按置信度从高到低排序；
    - 同类别且 ProbIoU >= 阈值 且 GT 未被占用 -> TP；
    - 否则为 FP；
    - 所有未匹配的 GT 为 FN。
    """
    num_pred = preds_xywhr.shape[0]
    num_gt = gts_xywhr.shape[0]

    if num_gt == 0 and num_pred == 0:
        return 0, 0, 0
    if num_gt == 0:
        return 0, num_pred, 0
    if num_pred == 0:
        return 0, 0, num_gt

    # 按置信度从大到小排序
    order = np.argsort(-preds_conf)
    preds_xywhr = preds_xywhr[order]
    preds_cls = preds_cls[order]
    preds_conf = preds_conf[order]

    # 计算 ProbIoU 矩阵 (N_gt, N_pred)
    iou_mat = batch_probiou(gts_xywhr, preds_xywhr).cpu().numpy()

    tp = 0
    fp = 0
    gt_used = np.zeros(num_gt, dtype=bool)

    for pi in range(num_pred):
        best_iou = 0.0
        best_gt = -1
        for gi in range(num_gt):
            if gt_used[gi]:
                continue
            if preds_cls[pi] != gts_cls[gi]:
                continue
            iou = iou_mat[gi, pi]
            if iou >= iou_thres and iou > best_iou:
                best_iou = iou
                best_gt = gi

        if best_gt >= 0:
            tp += 1
            gt_used[best_gt] = True
        else:
            fp += 1

    fn = int((~gt_used).sum())
    return tp, fp, fn


def analyze_image_contrib(
    model: YOLO,
    image_files: List[Path],
    labels_dir: Path,
    iou_thres: float = 0.5,
    conf_thres: float = 0.001,
    imgsz: int = 1024,
    device: str = "0",
    max_images: int = None,
) -> List[Dict]:
    """
    遍历验证集图片，基于 ProbIoU 统计每张图的 TP/FP/FN/F1 等信息。
    """
    results = []

    if max_images is not None:
        image_files = image_files[:max_images]
    total = len(image_files)

    print(f"开始逐张预测，共 {total} 张图片（max_images={max_images}）...")

    for idx, img_path in enumerate(image_files):
        try:
            preds_list = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf_thres,
                device=device,
                stream=False,
                verbose=False,
            )
            pred = preds_list[0]
        except Exception as e:
            print(f"⚠️ 预测 {img_path.name} 时出错: {e}")
            continue

        # 预测 OBB：xywhr + cls + conf
        if pred.obb is None or len(pred.obb) == 0:
            preds_xywhr = np.zeros((0, 5), dtype=np.float32)
            preds_cls = np.zeros((0,), dtype=np.int64)
            preds_conf = np.zeros((0,), dtype=np.float32)
        else:
            obb = pred.obb
            preds_xywhr = obb.xywhr.cpu().numpy().astype(np.float32)       # (N,5)
            preds_cls = obb.cls.cpu().numpy().astype(np.int64)             # (N,)
            preds_conf = obb.conf.cpu().numpy().astype(np.float32)         # (N,)

        # GT：四点归一化 -> 像素四点 -> xywhr
        label_file = labels_dir / (img_path.stem + ".txt")
        if preds_xywhr.shape[0] > 0:
            h, w = pred.orig_shape  # (H, W)
        else:
            # 即使没有预测，也需要知道图像尺寸以还原 GT
            h, w = pred.orig_shape
        gts_xywhr, gts_cls = load_obb_labels_xywhr(label_file, w, h)

        tp, fp, fn = match_rotated_prob_iou(
            preds_xywhr, preds_cls, preds_conf,
            gts_xywhr, gts_cls,
            iou_thres=iou_thres,
        )

        num_gt = gts_xywhr.shape[0]
        num_pred = preds_xywhr.shape[0]

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if (not math.isnan(precision)) and (not math.isnan(recall)) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        results.append(
            {
                "image": str(img_path),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            print(f"已处理 {idx + 1}/{total} 张图片")

    print(f"\n处理完成，共处理 {len(results)} 张图片")
    return results


def save_results_csv(records: List[Dict], csv_path: str) -> None:
    """按 F1 从低到高排序并保存为 CSV。"""
    import csv

    if len(records) == 0:
        print("⚠️ 警告: 没有记录可保存！")
        return

    sorted_records = sorted(
        records,
        key=lambda r: (float("inf") if math.isnan(r["f1"]) else r["f1"]),
    )

    fieldnames = [
        "rank",
        "image",
        "tp",
        "fp",
        "fn",
        "num_gt",
        "num_pred",
        "precision",
        "recall",
        "f1",
    ]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(sorted_records, start=1):
            row = dict(r)
            row["rank"] = i
            writer.writerow(row)

    print(f"\n✅ 已保存按贡献度排序的图片统计表: {csv_path}")
    print("   rank 越小，说明这张图的 F1 越低、越“拖后腿”")


def main():
    print("=" * 60)
    print("OBB 验证集图片贡献度分析（ProbIoU / xywhr）")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"数据配置: {DATA_YAML}")
    print(f"设备: {DEVICE}, imgsz={IMGSZ}")
    print(f"IOU 阈值: {IOU_THRES}, CONF 阈值: {CONF_THRES}")
    print(f"最多统计图片数: {MAX_IMAGES}")
    print()

    images_dir, labels_dir, image_files = load_data_paths(DATA_YAML, DATA_SPLIT)
    print(f"验证集 images 目录: {images_dir}")
    print(f"验证集 labels 目录: {labels_dir}")
    print(f"验证集图片数量: {len(image_files)}")
    print()

    print("加载模型中 ...")
    model = YOLO(MODEL_PATH, task=TASK)
    print("模型加载完成。\n")

    # ===== 新增：运行一次官方验证，打印 mAP50 =====
    print("运行官方 val() 计算整体指标（包括 mAP@0.5）...")
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        plots=False,
        verbose=False,
    )
    # 一般键名是 'metrics/mAP50(B)'，如果版本有差异再 fallback 到 box.map50
    map50 = metrics.results_dict.get("metrics/mAP50(B)", None)
    if map50 is None and hasattr(metrics, "box"):
        map50 = getattr(metrics.box, "map50", None)

    if map50 is not None:
        print(f"\n🔥 整体验证集 mAP@0.5 = {map50:.3f}")
    else:
        print("\n⚠️ 未能从 metrics 中解析出 mAP@0.5，请打印 metrics.results_dict 查看键名。")
    print()

    print("开始逐图统计 TP/FP/FN ...\n")
    # ===============================================

    # 逐图统计贡献度
    records = analyze_image_contrib(
        model=model,
        image_files=image_files,
        labels_dir=labels_dir,
        iou_thres=IOU_THRES,
        conf_thres=CONF_THRES,
        imgsz=IMGSZ,
        device=DEVICE,
        max_images=MAX_IMAGES,
    )

    save_results_csv(records, OUTPUT_CSV)

    print("\n示例：可以重点查看 rank 最小的前 20 张图片，它们是最拖后腿的样本。")

if __name__ == "__main__":
    main()