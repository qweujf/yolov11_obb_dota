#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查并统计：
- 空标签文件（存在txt但内容为空/空白）
- 缺失标签的图片（存在图像但没有同名txt）
可选：删除空标签及同名图片、删除无标签图片。
"""

from pathlib import Path
from typing import Optional

# ===== 固定目录配置（按需修改这里） =====
LABELS_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\labels\train")
IMAGES_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\images\train")
DELETE_EMPTY = False           # 删除空标签及同名图片
DELETE_BACKGROUND = False      # 删除无标签的图片（background）
VALID_IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp")
# ======================================


def is_empty_label(label_path: Path) -> bool:
    try:
        if not label_path.exists() or label_path.stat().st_size == 0:
            return True
        text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        return len(text) == 0
    except Exception:
        return True


def check_and_cleanup(labels_dir: Path, images_dir: Optional[Path], do_delete_empty: bool, do_delete_bg: bool) -> None:
    assert labels_dir.exists(), f"标签目录不存在: {labels_dir}"
    if images_dir is not None:
        assert images_dir.exists(), f"图片目录不存在: {images_dir}"

    # 统计空标签
    label_files = list(labels_dir.glob("*.txt"))
    total_labels = len(label_files)
    empty_labels = 0
    deleted_empty = 0

    for lf in label_files:
        if is_empty_label(lf):
            empty_labels += 1
            if do_delete_empty:
                try:
                    lf.unlink(missing_ok=True)
                    deleted_empty += 1
                    if images_dir is not None:
                        stem = lf.stem
                        for ext in VALID_IMG_EXTS:
                            img = images_dir / f"{stem}{ext}"
                            if img.exists():
                                img.unlink(missing_ok=True)
                except Exception as e:
                    print(f"删除失败: {lf} -> {e}")

    # 统计无标签图片（background）
    bg_images = 0
    deleted_bg = 0
    if images_dir is not None:
        img_files = [p for p in images_dir.iterdir() if p.suffix.lower() in VALID_IMG_EXTS]
        for img in img_files:
            stem = img.stem
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                bg_images += 1
                if do_delete_bg:
                    try:
                        img.unlink(missing_ok=True)
                        deleted_bg += 1
                    except Exception as e:
                        print(f"删除无标签图片失败: {img} -> {e}")

    # 打印结果
    print("\n===== 标签与背景检查结果 =====")
    print(f"标签目录: {labels_dir}")
    if images_dir is not None:
        print(f"图片目录: {images_dir}")
    print(f"总标签文件数: {total_labels}")
    print(f"空标签文件数: {empty_labels}")
    if do_delete_empty:
        print(f"已删除空标签文件: {deleted_empty}（及其同名图片）")
    if images_dir is not None:
        print(f"无标签图片数(background): {bg_images}")
        if do_delete_bg:
            print(f"已删除无标签图片: {deleted_bg}")
    print("================================\n")


if __name__ == "__main__":
    check_and_cleanup(LABELS_DIR, IMAGES_DIR, DELETE_EMPTY, DELETE_BACKGROUND)
