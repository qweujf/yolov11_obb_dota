#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修复 YOLO OBB 标签：
- 检测标签中是否存在 >1 的坐标（判定为像素坐标未归一化）
- 读取同名图片尺寸 (W,H)，并从切片文件名解析窗口偏移，
  先减去 (x_start,y_start) 再按切片尺寸归一化到 [0,1]
- 仅处理 9 列格式：cls x1 y1 x2 y2 x3 y3 x4 y4
- 已归一化的文件跳过

使用：直接修改脚本顶部路径常量后运行。
"""

from pathlib import Path
import cv2
from typing import List, Tuple, Optional
import re

# ===== 按需修改这里（指向你的切割数据集） =====
IMAGES_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\images\val")
LABELS_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\labels\val")
BACKUP_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\labels_backup_val")  # 备份原标签
# ==========================================


def needs_normalize(parts: List[str]) -> bool:
    try:
        nums = [float(x) for x in parts[1:9]]
        return any(v > 1.0 or v < 0.0 for v in nums)
    except Exception:
        return False


def clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def parse_patch_info(stem: str) -> Optional[Tuple[int, int, int]]:
    """从切片文件名解析 (patch_w, x_start, y_start)。
    期望格式：<name>__{patch_w}__{x_start}___{y_start}
    例：P9980__682__342___0 -> (682,342,0)
    解析失败返回 None。
    """
    try:
        # 先分割 y_start
        if "___" not in stem or "__" not in stem:
            return None
        left, y_str = stem.rsplit("___", 1)
        parts = left.split("__")
        # 末尾两段应为 patch_w, x_start
        patch_w = int(parts[-2])
        x_start = int(parts[-1])
        y_start = int(y_str)
        return patch_w, x_start, y_start
    except Exception:
        return None


def process_one(label_path: Path) -> Tuple[bool, str]:
    img = None
    try:
        stem = label_path.stem
        # 找同名图片
        img_path = None
        for ext in ('.jpg', '.png', '.jpeg', '.bmp'):
            p = IMAGES_DIR / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            return False, f"找不到同名图片: {stem}.*"
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 从文件名解析窗口信息
        patch = parse_patch_info(stem)
        if patch is not None:
            patch_w, x0, y0 = patch
            patch_h = h  # 以实际切片高为准（边缘切片可能非正方）
        else:
            # 回退：无偏移信息，仅按 w,h 归一化
            patch_w, patch_h, x0, y0 = w, h, 0, 0

        lines = label_path.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
        if not lines:
            return False, "空文件"

        changed = False
        new_lines: List[str] = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) < 9:
                # 非法行，跳过写回
                continue
            if not needs_normalize(parts) and x0 == 0 and y0 == 0:
                # 已在[0,1]且无偏移，直接保留
                new_lines.append(ln)
                continue
            # 归一化：先减偏移，再除以切片尺寸
            cls = parts[0]
            coords = [float(x) for x in parts[1:9]]
            for i in range(0, 8, 2):
                x = coords[i] - x0
                y = coords[i+1] - y0
                x /= float(w)  # 以当前切片实际宽/高归一化
                y /= float(h)
                coords[i] = clamp01(x)
                coords[i+1] = clamp01(y)
            norm = " ".join(f"{v:.6f}" for v in coords)
            new_lines.append(f"{cls} {norm}")
            changed = True
        if changed:
            # 备份一次
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            backup_path = BACKUP_DIR / label_path.name
            if not backup_path.exists():
                backup_path.write_text(label_path.read_text(encoding='utf-8', errors='ignore'), encoding='utf-8')
            label_path.write_text("\n".join(new_lines) + "\n", encoding='utf-8')
        return changed, "已修复" if changed else "无需修改"
    except Exception as e:
        return False, f"异常: {e}"


def main():
    assert IMAGES_DIR.exists(), f"不存在图片目录: {IMAGES_DIR}"
    assert LABELS_DIR.exists(), f"不存在标签目录: {LABELS_DIR}"
    fixed = 0
    skipped = 0
    failed = 0

    label_files = list(LABELS_DIR.glob('*.txt'))
    for i, lp in enumerate(label_files, 1):
        changed, msg = process_one(lp)
        if changed:
            fixed += 1
        else:
            if msg.startswith("异常"):
                failed += 1
            else:
                skipped += 1
        if i % 1000 == 0:
            print(f"处理 {i}/{len(label_files)} ... 已修复{fixed}, 跳过{skipped}, 异常{failed}")

    print("\n===== 修复结果 =====")
    print(f"标签总数: {len(label_files)}")
    print(f"已修复(归一化)标签: {fixed}")
    print(f"无需修改: {skipped}")
    print(f"异常: {failed}")
    print(f"备份目录: {BACKUP_DIR}")
    print("===================\n")


if __name__ == "__main__":
    main()
