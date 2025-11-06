#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计并过滤数据集中的背景图像（空标签图像）

使用方法：
    python scripts/filter_background_images.py --data_root D:\code\yolov11_obb_dota\zuhe\split_dota_1024 --dry_run
    python scripts/filter_background_images.py --data_root D:\code\yolov11_obb_dota\zuhe\split_dota_1024 --delete
"""

import argparse
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


def is_empty_label(label_path: Path) -> bool:
    """检查标签文件是否为空"""
    try:
        if not label_path.exists():
            return True
        if label_path.stat().st_size == 0:
            return True
        content = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        return len(content) == 0
    except Exception:
        return True


def count_background_images(data_root: Path, split: str = "train") -> Tuple[int, int, int]:
    """
    统计背景图像数量
    
    Returns:
        (total_images, labeled_images, background_images)
    """
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"⚠️ 目录不存在: {images_dir} 或 {labels_dir}")
        return 0, 0, 0
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    total = len(image_files)
    labeled = 0
    background = 0
    
    print(f"📊 统计 {split} 集...")
    for img_path in tqdm(image_files, desc=f"Processing {split}"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if is_empty_label(label_path):
            background += 1
        else:
            labeled += 1
    
    return total, labeled, background


def delete_background_images(data_root: Path, split: str = "train", dry_run: bool = True) -> int:
    """删除背景图像及其对应的空标签文件"""
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"⚠️ 目录不存在: {images_dir} 或 {labels_dir}")
        return 0
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    deleted_count = 0
    
    mode = "🔍 [DRY RUN] 将删除" if dry_run else "🗑️ 删除"
    print(f"\n{mode} {split} 集中的背景图像...")
    
    for img_path in tqdm(image_files, desc=f"{mode} {split}"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if is_empty_label(label_path):
            if not dry_run:
                img_path.unlink()  # 删除图像
                if label_path.exists():
                    label_path.unlink()  # 删除空标签
            deleted_count += 1
    
    return deleted_count


def main():
    parser = argparse.ArgumentParser(description="统计并过滤数据集中的背景图像")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅统计，不实际删除（默认行为）"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="实际删除背景图像（谨慎使用！）"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="要处理的集合：train, val, 或 both（默认）"
    )
    
    args = parser.parse_args()
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"❌ 数据根目录不存在: {data_root}")
        return
    
    print(f"📁 数据根目录: {data_root}\n")
    
    # 统计信息
    splits = ["train", "val"] if args.split == "both" else [args.split]
    stats = {}
    
    for split in splits:
        total, labeled, background = count_background_images(data_root, split)
        stats[split] = {
            "total": total,
            "labeled": labeled,
            "background": background,
            "ratio": background / total * 100 if total > 0 else 0
        }
    
    # 打印统计结果
    print("\n" + "="*60)
    print("📊 统计结果")
    print("="*60)
    for split, s in stats.items():
        print(f"\n{split.upper()} 集:")
        print(f"  总图像数: {s['total']:,}")
        print(f"  带标签: {s['labeled']:,} ({100-s['ratio']:.1f}%)")
        print(f"  背景图: {s['background']:,} ({s['ratio']:.1f}%)")
    
    # 建议
    print("\n" + "="*60)
    print("💡 建议")
    print("="*60)
    
    max_ratio = max(s["ratio"] for s in stats.values())
    if max_ratio > 50:
        print("⚠️  背景图像比例过高（>50%），建议过滤")
        print("   在目标检测任务中，背景图像比例通常应控制在 10-30% 之间")
    elif max_ratio > 30:
        print("⚠️  背景图像比例较高（>30%），可考虑适当过滤")
    else:
        print("✅ 背景图像比例在合理范围内")
    
    # 删除操作
    if args.delete:
        print("\n" + "="*60)
        print("⚠️  确认删除背景图像？")
        print("="*60)
        confirm = input("输入 'YES' 确认删除（此操作不可逆）: ")
        if confirm == "YES":
            for split in splits:
                deleted = delete_background_images(data_root, split, dry_run=False)
                print(f"✅ {split} 集：删除了 {deleted:,} 个背景图像")
        else:
            print("❌ 操作已取消")
    elif args.dry_run or not args.delete:
        print("\n" + "="*60)
        print("💡 提示")
        print("="*60)
        print("当前为 DRY RUN 模式，仅统计不删除")
        print("要实际删除背景图像，请使用 --delete 参数")
        print("\n示例命令：")
        print(f"  python scripts/filter_background_images.py --data_root {data_root} --delete")


if __name__ == "__main__":
    main()

