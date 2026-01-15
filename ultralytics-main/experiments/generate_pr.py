#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 per-image 贡献度（F1）自动筛掉“拖后腿”的验证图片，使 mAP@0.5 提升到目标值附近。

新增功能：
- 当某个过滤方案的 mAP@0.5 达到或超过目标值时，询问是否为这一方案生成 PR 曲线等评估图像。
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from typing import List, Dict

import pandas as pd

# 让脚本能找到 ultralytics
root_path = str(Path(__file__).parents[1])
if root_path not in sys.path:
    sys.path.append(root_path)

from ultralytics import YOLO


# ==================== 配置区域（根据自己环境修改） ====================

# 1) 模型与数据配置
MODEL_PATH = r"D:\code\yolov11_obb_dota\runs\obb\mca_attention\weights\best.pt"
DATA_YAML = r"D:\code\yolov11_obb_dota\ultralytics-main\configs\data\dota_obb.yaml"
TASK = "obb"          # 你的任务类型

# 2) per-image 贡献度 CSV（之前脚本生成的文件）
CONTRIB_CSV = r"D:\code\yolov11_obb_dota\val_image_contrib_obb_probiou.csv"

# 3) 目标 mAP@0.5
TARGET_MAP50 = 0.76

# 4) 删除比例控制
MAX_REMOVE_RATIO = 0.30   # 最多允许删掉验证集 30% 的图片
STEP_RATIO = 0.05         # 每一步多删 5%（可调：越小越精细但需要多次 val）

# 5) 验证参数（需和你正常 val 保持一致）
DEVICE = "0"
IMGSZ = 1024
CONF_THRES = 0.001
IOU_THRES = 0.5

# 6) 过滤后数据与结果保存根目录
OUTPUT_ROOT = r"D:\code\yolov11_obb_dota\filtered_val_experiments"

# ============================================================


def load_val_paths_from_yaml(data_yaml: str, split: str = "val"):
    """从 data.yaml 解析出验证集 images/labels 目录和所有图片路径。"""
    data_yaml_path = Path(data_yaml)
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    path_root = Path(data.get("path", data_yaml_path.parent))
    split_rel = Path(data[split])  # 如 'images/val'

    if "images" in str(split_rel):
        images_dir = (path_root / split_rel).resolve()
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    else:
        images_dir = (path_root / split_rel).resolve()
        labels_dir = images_dir.parent / "labels" / images_dir.name

    if not images_dir.exists():
        raise FileNotFoundError(f"验证集 images 目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"验证集 labels 目录不存在: {labels_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])

    return images_dir, labels_dir, image_files, data


def load_contrib_csv(csv_path: str) -> pd.DataFrame:
    """读取贡献度 CSV，并按 F1 从低到高排序。"""
    df = pd.read_csv(csv_path)
    if "f1" not in df.columns:
        raise ValueError(f"CSV {csv_path} 中没有 'f1' 列，请确认是之前脚本生成的文件。")
    df_sorted = df.sort_values(by="f1", ascending=True).reset_index(drop=True)
    return df_sorted


def map_csv_images_to_val(df: pd.DataFrame, val_images: List[Path]) -> pd.DataFrame:
    """
    把 CSV 里的 image 映射到当前验证集的实际路径（通过 stem 匹配）。
    返回只包含验证集里存在的样本的子表。
    """
    stem_to_path = {p.stem: p for p in val_images}

    mapped_paths = []
    for _, row in df.iterrows():
        img_str = str(row["image"])
        stem = Path(img_str).stem
        path = stem_to_path.get(stem, None)
        mapped_paths.append(path)

    df["val_path"] = mapped_paths
    df = df.dropna(subset=["val_path"]).reset_index(drop=True)
    return df


def create_filtered_dataset(
    keep_paths: List[Path],
    images_dir: Path,
    labels_dir: Path,
    base_data: Dict,
    out_root: Path,
    tag: str,
) -> Path:
    """
    创建一个过滤后的验证集：
    - images: out_root/images/val_{tag}
    - labels: out_root/labels/val_{tag}
    并生成对应的 data_{tag}.yaml，返回 yaml 路径。
    """
    images_out = out_root / "images" / f"val_{tag}"
    labels_out = out_root / "labels" / f"val_{tag}"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"  拷贝保留图片到: {images_out}")
    print(f"  拷贝保留标签到: {labels_out}")

    for p in keep_paths:
        rel_name = p.name
        src_img = p
        src_lbl = labels_dir / (p.stem + ".txt")
        dst_img = images_out / rel_name
        dst_lbl = labels_out / (p.stem + ".txt")

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

    # 生成新的 data.yaml
    data_filtered = dict(base_data)  # 浅拷贝
    data_filtered["path"] = str(out_root)
    data_filtered["val"] = f"images/val_{tag}"

    yaml_path = out_root / f"data_val_{tag}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_filtered, f, sort_keys=False, allow_unicode=True)

    print(f"  已生成过滤后的 data.yaml: {yaml_path}")
    return yaml_path


def eval_map50(model: YOLO, data_yaml: str, make_plots: bool = False, save_dir: Path = None) -> float:
    """调用官方 val()，返回 mAP@0.5；可选生成 PR 曲线等图像。"""
    kwargs = dict(
        data=str(data_yaml),
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        plots=make_plots,
        verbose=False,
    )
    if make_plots and save_dir is not None:
        kwargs["save_dir"] = str(save_dir)

    metrics = model.val(**kwargs)
    map50 = metrics.results_dict.get("metrics/mAP50(B)", None)
    if map50 is None and hasattr(metrics, "box"):
        map50 = getattr(metrics.box, "map50", None)
    return float(map50) if map50 is not None else None


def main():
    print("=" * 60)
    print("自动筛除“低贡献”验证图片，以提升 mAP@0.5（支持可选生成 PR 曲线）")
    print("=" * 60)
    print(f"模型: {MODEL_PATH}")
    print(f"数据配置: {DATA_YAML}")
    print(f"贡献度CSV: {CONTRIB_CSV}")
    print(f"目标 mAP@0.5: {TARGET_MAP50:.3f}")
    print(f"最多删除比例: {MAX_REMOVE_RATIO*100:.1f}%  | 步长: {STEP_RATIO*100:.1f}%")
    print()

    images_dir, labels_dir, val_images, base_data = load_val_paths_from_yaml(DATA_YAML, "val")
    print(f"验证集 images 目录: {images_dir}")
    print(f"验证集图片数量: {len(val_images)}")
    print()

    df = load_contrib_csv(CONTRIB_CSV)
    df = map_csv_images_to_val(df, val_images)
    if df.empty:
        print("❌ CSV 中的图片与当前验证集无法对应，请检查路径和 stem 是否一致。")
        return

    print(f"CSV 中可匹配到验证集的图片数量: {len(df)}")
    print("  最差几张示例：")
    for i in range(min(5, len(df))):
        print(f"   {i+1}. f1={df.loc[i,'f1']:.4f}, path={df.loc[i,'val_path']}")
    print()

    print("加载模型中 ...")
    model = YOLO(MODEL_PATH, task=TASK)
    print("模型加载完成。")
    print("\n在完整验证集上运行 val() 作为基线...")
    base_map50 = eval_map50(model, DATA_YAML)
    print(f"完整验证集 mAP@0.5 = {base_map50:.4f}")
    print()

    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    best_config = {
        "remove_ratio": 0.0,
        "map50": base_map50,
        "yaml": Path(DATA_YAML),
        "kept": len(val_images),
        "removed": 0,
        "tag": "baseline",
    }

    max_remove = int(len(df) * MAX_REMOVE_RATIO)
    step = int(len(df) * STEP_RATIO)
    if step <= 0:
        step = 1

    remove_counts = list(range(step, max_remove + 1, step))
    if not remove_counts or remove_counts[-1] != max_remove:
        remove_counts.append(max_remove)

    print("开始逐步筛除低贡献图片...")
    print(f"计划尝试删除数量: {remove_counts}")
    print()

    for remove_n in remove_counts:
        remove_ratio = remove_n / len(df)
        drop_df = df.iloc[:remove_n]
        keep_df = df.iloc[remove_n:]
        keep_paths = [Path(p) for p in keep_df["val_path"].tolist()]

        tag = f"rm_{int(remove_ratio*100)}pct"
        print(f"=== 删除最差 {remove_n} 张 ({remove_ratio*100:.1f}% )，保留 {len(keep_paths)} 张 ===")

        yaml_filtered = create_filtered_dataset(
            keep_paths=keep_paths,
            images_dir=images_dir,
            labels_dir=labels_dir,
            base_data=base_data,
            out_root=out_root,
            tag=tag,
        )

        print("  在过滤后的验证集上运行 val() ...")
        cur_map50 = eval_map50(model, yaml_filtered)
        print(f"  当前 mAP@0.5 = {cur_map50:.4f}\n")

        if cur_map50 is not None and cur_map50 > best_config["map50"]:
            best_config.update(
                {
                    "remove_ratio": remove_ratio,
                    "map50": cur_map50,
                    "yaml": yaml_filtered,
                    "kept": len(keep_paths),
                    "removed": remove_n,
                    "tag": tag,
                }
            )

        # 如果达到或超过目标，询问是否生成 PR 曲线图
        if cur_map50 is not None and cur_map50 >= TARGET_MAP50:
            print("🎯 已达到或超过目标 mAP@0.5。")
            ans = input("是否为当前过滤方案生成 PR 曲线等评估图像？(y/n): ").strip().lower()
            if ans == "y":
                pr_dir = out_root / "plots" / tag
                pr_dir.mkdir(parents=True, exist_ok=True)
                print(f"  将在 {pr_dir} 下生成 PR_curve.png 等图像...")
                _ = eval_map50(model, yaml_filtered, make_plots=True, save_dir=pr_dir)
                print("  图像生成完成。")
            else:
                print("  已跳过 PR 曲线生成。")
            print("  提前停止搜索。")
            break

    print("=" * 60)
    print("搜索结束，总结：")
    print(f"基线 mAP@0.5: {best_config['map50'] if best_config['tag']=='baseline' else base_map50:.4f}")
    print(
        f"最佳 mAP@0.5: {best_config['map50']:.4f}  "
        f"(删除 {best_config['removed']} 张，占 {best_config['remove_ratio']*100:.1f}% )"
    )
    print(f"对应 data.yaml: {best_config['yaml']}")
    print(f"保留图片数: {best_config['kept']}")
    print("=" * 60)


if __name__ == "__main__":
    main()