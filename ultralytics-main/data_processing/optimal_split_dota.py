#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的DOTA数据集切割脚本
基于最佳实践参数配置
"""

import os
import sys
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from ultralytics.data.split_dota import split_trainval, split_test
from ultralytics.utils import TQDM
import cv2
import numpy as np

def optimal_split_dota(
    data_root: str,
    save_dir: str,
    crop_size: int = 1024,
    gap: int = 200,
    rates: tuple = (0.5, 1.0, 1.5),
    min_area_ratio: float = 0.05,
    allow_background: bool = False,
    iof_thr: float = 0.7,
    im_rate_thr: float = 0.6
):
    """
    优化的DOTA数据集切割函数
    
    Args:
        data_root (str): 原始数据根目录
        save_dir (str): 保存目录
        crop_size (int): 切割尺寸，默认1024
        gap (int): 重叠区域，默认200
        rates (tuple): 缩放比例，默认(1.0,)
        min_area_ratio (float): 最小目标面积比例
        allow_background (bool): 是否允许背景图像
        iof_thr (float): IoF阈值
        im_rate_thr (float): 图像覆盖率阈值
    """
    print("🚀 开始优化的DOTA数据集切割...")
    print(f"📁 数据根目录: {data_root}")
    print(f"💾 保存目录: {save_dir}")
    print(f"📏 切割尺寸: {crop_size}")
    print(f"🔄 重叠区域: {gap}")
    print(f"📐 缩放比例: {rates}")
    print(f"🎯 最小目标面积比例: {min_area_ratio}")
    print(f"🚫 允许背景图像: {allow_background}")
    
    # 检查数据目录
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    
    # 检查必要的子目录
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"缺少必要目录: {dir_path}")
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 修改split_dota模块中的默认参数
    import ultralytics.data.split_dota as split_module
    
    # 保存原始函数
    original_crop_and_save = split_module.crop_and_save
    original_get_window_obj_func = split_module.get_window_obj  # 保存原始函数引用

    def improved_crop_and_save(anno, windows, window_objs, im_dir, lb_dir, allow_background_images=allow_background):
        """改进的裁剪和保存函数"""
        im = cv2.imread(anno["filepath"])
        name = Path(anno["filepath"]).stem
        
        for i, window in enumerate(windows):
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            ph, pw = patch_im.shape[:2]

            label = window_objs[i]
            
            # 检查是否有有效标签
            valid_labels = []
            if len(label):
                for lb in label:
                    # 计算目标在窗口中的面积比例
                    coords = lb[1:].reshape(-1, 2).astype(np.float32)  # 这里的 lb 已经是像素坐标，无需再乘 w,h
                    
                    # 计算多边形面积
                    area = cv2.contourArea(coords.astype(np.float32))
                    window_area = (x_stop - x_start) * (y_stop - y_start)
                    area_ratio = area / window_area
                    
                    # 只保留面积比例大于阈值的目标
                    if area_ratio >= min_area_ratio:
                        valid_labels.append(lb)
            
            # 只有当存在有效标签或允许背景图像时才保存
            if len(valid_labels) > 0 or allow_background_images:
                cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
                
                if len(valid_labels) > 0:
                    # 转换为numpy数组
                    valid_labels = np.array(valid_labels)
                    # 调整坐标
                    valid_labels[:, 1::2] -= x_start
                    valid_labels[:, 2::2] -= y_start
                    valid_labels[:, 1::2] /= pw
                    valid_labels[:, 2::2] /= ph

                    eps = 1e-6
                    xy = valid_labels[:, 1:].reshape(-1, 2)
                    in01 = (xy[:, 0] >= -eps) & (xy[:, 0] <= 1.0 + eps) & (xy[:, 1] >= -eps) & (xy[:, 1] <= 1.0 + eps)
                    keep_mask = in01.reshape(-1, 4).all(axis=1)  # 四个角都在[0,1]±eps
                    valid_labels = valid_labels[keep_mask]

                    # 可选：轻微越界夹紧
                    valid_labels[:, 1::2] = np.clip(valid_labels[:, 1::2], 0.0, 1.0)
                    valid_labels[:, 2::2] = np.clip(valid_labels[:, 2::2], 0.0, 1.0)

                    # 可选：归一化面积过小的目标再丢弃（避免边缘残片）
                    def _area8(v):
                        pts = v.reshape(4, 2).astype(np.float32)
                        return cv2.contourArea(pts)

                    if len(valid_labels):
                        areas = np.array([_area8(x[1:]) for x in valid_labels], dtype=np.float32)
                        min_area_ratio_post = 0.005  # ≈窗口面积的0.5%
                        valid_labels = valid_labels[areas >= min_area_ratio_post]

                    with open(Path(lb_dir) / f"{new_name}.txt", "w", encoding="utf-8") as f:
                        for lb in valid_labels:
                            formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                            f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")
    
    def improved_get_window_obj(anno, windows, iof_thr=iof_thr):
        """改进的窗口目标获取函数"""
        if anno["label"] is not None:
            return original_get_window_obj_func(anno, windows, iof_thr)
        else:
            return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]

    split_module.crop_and_save = original_crop_and_save
    split_module.get_window_obj = original_get_window_obj_func
    
    try:
        # 执行切割
        print("📊 开始切割训练和验证集...")
        split_trainval(
            data_root=data_root,
            save_dir=save_dir,
            crop_size=crop_size,
            gap=gap,
            rates=rates
        )
        
        print("✅ 切割完成！")
        
        # 统计结果
        count_images_and_labels(save_dir)
        
    finally:
        # 恢复原始函数
        split_module.crop_and_save = original_crop_and_save
        split_module.get_window_obj = original_get_window_obj

def count_images_and_labels(save_dir: str):
    """统计切割后的图像和标签数量"""
    save_path = Path(save_dir)
    
    for split in ["train", "val"]:
        img_dir = save_path / "images" / split
        label_dir = save_path / "labels" / split
        
        if img_dir.exists() and label_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg")))
            label_count = len(list(label_dir.glob("*.txt")))
            print(f"📈 {split}集: {img_count}张图像, {label_count}个标签文件")
        else:
            print(f"⚠️ {split}集目录不存在")

def split_test_set(
    data_root: str,
    save_dir: str,
    crop_size: int = 1024,
    gap: int = 200,
    rates: tuple = (1.0,)
):
    """切割测试集"""
    print("🔍 开始切割测试集...")
    
    test_dir = Path(data_root) / "images" / "test"
    if not test_dir.exists():
        print("⚠️ 测试集目录不存在，跳过测试集切割")
        return
    
    split_test(
        data_root=data_root,
        save_dir=save_dir,
        crop_size=crop_size,
        gap=gap,
        rates=rates
    )
    
    print("✅ 测试集切割完成！")

if __name__ == "__main__":
    # 配置参数
    config = {
        'data_root': r"D:\code\yolov11_obb_dota\zuhe\raw",
        'save_dir': r"D:\code\yolov11_obb_dota\zuhe\try",
        'crop_size': 1024,
        'gap': 200,
        'rates': (0.5, 1.0, 1.5),
        'min_area_ratio': 0.05,
        'allow_background': False,
        'iof_thr': 0.7,
        'im_rate_thr': 0.6
    }
    
    # 执行切割
    optimal_split_dota(**config)
    
    # 可选：切割测试集
    # split_test_set(
    #     data_root=config['data_root'],
    #     save_dir=config['save_dir'],
    #     crop_size=config['crop_size'],
    #     gap=config['gap'],
    #     rates=config['rates']
    # )
    
    print("🎉 所有切割任务完成！")

