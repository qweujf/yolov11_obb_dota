#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOTA-v2 数据集标准切分脚本（论文常用配置）

基于常规做法：
- crop_size: 1024 (标准做法)
- gap: 200 (常用重叠区域)
- rates: (1.0,) (单尺度，避免过多背景图)
- allow_background: False (不允许背景图像，论文标准做法)
- iof_thr: 0.7 (默认值，用于判断目标是否在窗口内)

参考文献：
- DOTA: A Large-scale Dataset for Object Detection in Aerial Images
- 大多数 YOLO-OBB 相关论文使用的标准配置
"""

import sys
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 添加项目路径
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from ultralytics.data.split_dota import split_trainval, split_test
from ultralytics.utils import TQDM
import cv2
import numpy as np


def split_dota_v2_standard(
    data_root: str,
    save_dir: str,
    crop_size: int = 1024,
    gap: int = 200,
    rates: tuple = (1.0,),
    allow_background: bool = False,
    iof_thr: float = 0.7,
):
    """
    DOTA-v2 数据集标准切分（论文常用配置）
    
    Args:
        data_root (str): 原始数据根目录（需包含 images/train, images/val, labels/train, labels/val）
        save_dir (str): 保存切分后的数据目录
        crop_size (int): 切分尺寸，默认 1024（论文标准）
        gap (int): 重叠区域，默认 200（论文常用）
        rates (tuple): 缩放比例，默认 (1.0,) 单尺度（论文中常用，避免过多背景图）
        allow_background (bool): 是否允许背景图像，默认 False（论文标准做法）
        iof_thr (float): IoF阈值，默认 0.7（用于判断目标是否在窗口内）
    
    Notes:
        这是论文中最常用的配置，平衡了数据量和背景图比例。
        单尺度 (1.0) 可以减少背景图，如果确实需要多尺度，可以使用 (0.5, 1.0, 1.5)
        但会增加背景图比例，建议切分后过滤。
    """
    print("="*70)
    print("🚀 DOTA-v2 数据集标准切分（论文配置）")
    print("="*70)
    print(f"📁 数据根目录: {data_root}")
    print(f"💾 保存目录: {save_dir}")
    print(f"📏 切分尺寸: {crop_size}")
    print(f"🔄 重叠区域: {gap}")
    print(f"📐 缩放比例: {rates}")
    print(f"🚫 允许背景图像: {allow_background}")
    print(f"🎯 IoF阈值: {iof_thr}")
    print("="*70)
    
    # 检查数据目录
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"❌ 数据目录不存在: {data_root}")
    
    # 检查必要的子目录
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        raise FileNotFoundError(
            f"❌ 缺少必要的目录:\n" + "\n".join(f"  - {d}" for d in missing_dirs)
        )
    
    # 修改 split_dota 模块的 crop_and_save 函数以支持过滤背景图
    import ultralytics.data.split_dota as split_module
    
    # 保存原始函数
    original_crop_and_save = split_module.crop_and_save
    original_get_window_obj = split_module.get_window_obj
    
    def modified_crop_and_save(anno, windows, window_objs, im_dir, lb_dir, allow_background_images=allow_background):
        """修改后的裁剪和保存函数，支持过滤背景图"""
        im = cv2.imread(anno["filepath"])
        name = Path(anno["filepath"]).stem
        
        for i, window in enumerate(windows):
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            ph, pw = patch_im.shape[:2]

            label = window_objs[i]
            
            # 只有存在标签或允许背景图时才保存
            if len(label) > 0 or allow_background_images:
                cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
                
                if len(label) > 0:
                    # 将像素坐标转换为相对于窗口的坐标
                    label[:, 1::2] -= x_start  # x坐标减去窗口起始x
                    label[:, 2::2] -= y_start  # y坐标减去窗口起始y
                    # 归一化到[0,1]范围
                    label[:, 1::2] /= pw
                    label[:, 2::2] /= ph
                    
                    # 重要：IoF >= 0.7 的目标应该被保留，即使坐标越界
                    # 这是因为 IoF >= 0.7 意味着目标的70%+在窗口内，应该保留标注
                    # 对于越界的坐标，我们将其 clip 到 [0, 1] 范围内
                    # 这样虽然会改变目标的形状（截断），但保留了目标的标注信息
                    valid_labels = []
                    for lb in label:
                        coords = lb[1:].reshape(-1, 2)  # 8个坐标值 -> 4个(x,y)点
                        
                        # 检查坐标是否在合理范围内（允许小的浮点误差）
                        eps = 1e-6
                        # 如果坐标严重越界（超出太多），可能是计算错误，丢弃
                        # 但如果是轻微越界（如 1.01），说明是边界目标，应该保留
                        if np.any(coords < -0.1) or np.any(coords > 1.1):
                            # 严重越界，可能是计算错误，丢弃
                            continue
                        
                        # 将坐标 clip 到 [0, 1] 范围内
                        # 这样可以保留部分在窗口外的目标（IoF >= 0.7 的目标）
                        coords = np.clip(coords, 0.0, 1.0)
                        
                        # 计算clip后的多边形面积，如果面积太小（可能是只剩一个点或一条线），丢弃
                        try:
                            pts = coords.astype(np.float32)
                            area = cv2.contourArea(pts)
                            min_area = 1e-6  # 最小归一化面积（约0.0001%的窗口面积）
                            if area < min_area:
                                continue
                        except Exception:
                            # 如果无法计算面积，保留该标签（可能是特殊情况）
                            pass
                        
                        lb[1:] = coords.flatten()
                        valid_labels.append(lb)
                    
                    # 保存有效标签（如果存在）
                    if len(valid_labels) > 0:
                        with open(Path(lb_dir) / f"{new_name}.txt", "w", encoding="utf-8") as f:
                            for lb in valid_labels:
                                formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                                f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")
    
    def modified_get_window_obj(anno, windows, iof_thr=iof_thr):
        """使用指定的 IoF 阈值"""
        return original_get_window_obj(anno, windows, iof_thr)
    
    # 临时替换函数
    split_module.crop_and_save = modified_crop_and_save
    split_module.get_window_obj = modified_get_window_obj
    
    # 修改 split_images_and_labels 以传递 allow_background_images 参数
    original_split_images_and_labels = split_module.split_images_and_labels
    
    def modified_split_images_and_labels(
        data_root: str,
        save_dir: str,
        split: str = "train",
        crop_sizes: tuple = (1024,),
        gaps: tuple = (200,),
    ):
        """修改后的切分函数，支持传递 allow_background_images"""
        from ultralytics.data.split_dota import load_yolo_dota, get_windows
        
        im_dir = Path(save_dir) / "images" / split
        im_dir.mkdir(parents=True, exist_ok=True)
        lb_dir = Path(save_dir) / "labels" / split
        lb_dir.mkdir(parents=True, exist_ok=True)

        annos = load_yolo_dota(data_root, split=split)
        for anno in TQDM(annos, total=len(annos), desc=split):
            windows = get_windows(anno["ori_size"], crop_sizes, gaps)
            window_objs = modified_get_window_obj(anno, windows)
            modified_crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))
    
    split_module.split_images_and_labels = modified_split_images_and_labels
    
    try:
        # 执行切分
        print("\n📦 开始切分训练集和验证集...")
        split_trainval(
            data_root=data_root,
            save_dir=save_dir,
            crop_size=crop_size,
            gap=gap,
            rates=rates
        )
        print("✅ 训练集和验证集切分完成！")
        
        # 可选：切分测试集（如果存在）
        test_dir = data_path / "images" / "test"
        if test_dir.exists():
            print("\n📦 开始切分测试集...")
            split_test(
                data_root=data_root,
                save_dir=save_dir,
                crop_size=crop_size,
                gap=gap,
                rates=rates
            )
            print("✅ 测试集切分完成！")
        
    finally:
        # 恢复原始函数
        split_module.crop_and_save = original_crop_and_save
        split_module.get_window_obj = original_get_window_obj
        split_module.split_images_and_labels = original_split_images_and_labels
    
    # 统计结果
    print("\n" + "="*70)
    print("📊 切分结果统计")
    print("="*70)
    
    for split in ["train", "val"]:
        im_dir = Path(save_dir) / "images" / split
        lb_dir = Path(save_dir) / "labels" / split
        
        if im_dir.exists() and lb_dir.exists():
            image_files = list(im_dir.glob("*.jpg")) + list(im_dir.glob("*.png"))
            total_images = len(image_files)
            
            labeled_count = 0
            background_count = 0
            
            for img_path in image_files:
                label_path = lb_dir / f"{img_path.stem}.txt"
                if label_path.exists() and label_path.stat().st_size > 0:
                    content = label_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(content) > 0:
                        labeled_count += 1
                    else:
                        background_count += 1
                else:
                    background_count += 1
            
            print(f"\n{split.upper()} 集:")
            print(f"  总图像数: {total_images:,}")
            print(f"  带标签: {labeled_count:,} ({labeled_count/total_images*100:.1f}%)" if total_images > 0 else "  带标签: 0")
            print(f"  背景图: {background_count:,} ({background_count/total_images*100:.1f}%)" if total_images > 0 else "  背景图: 0")
    
    print("\n" + "="*70)
    print("🎉 切分完成！")
    print("="*70)
    print(f"💾 切分后的数据保存在: {save_dir}")
    print("💡 提示: 如果背景图比例仍然较高，可以运行 filter_background_images.py 进一步过滤")


if __name__ == "__main__":
    # ===== 配置参数（按需修改） =====
    config = {
        'data_root': r"D:\code\yolov11_obb_dota\zuhe\raw",  # 原始数据根目录
        'save_dir': r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard",  # 保存目录
        'crop_size': 1024,      # 切分尺寸（论文标准）
        'gap': 200,              # 重叠区域（论文常用）
        'rates': (1.0,),         # 单尺度（避免过多背景图）
        'allow_background': False,  # 不允许背景图像（论文标准做法）
        'iof_thr': 0.7,          # IoF阈值（默认值）
    }
    # =================================
    
    # 执行切分
    split_dota_v2_standard(**config)

