import os
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from ultralytics.data.split_dota import split_trainval
from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils import TQDM

def filter_empty_labels(data_root: str):
    """
    过滤并删除空标签文件对应的图像和标签
    
    Args:
        data_root (str): 数据根目录
    """
    print("开始过滤空标签文件...")
    
    for split in ["train", "val"]:
        image_dir = Path(data_root) / "images" / split
        label_dir = Path(data_root) / "labels" / split
        
        if not image_dir.exists() or not label_dir.exists():
            continue
            
        label_files = list(label_dir.glob("*.txt"))
        empty_count = 0
        
        for label_file in TQDM(label_files, desc=f"Processing {split} labels"):
            # 检查标签文件是否为空或只包含空白行
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if not content:  # 空文件
                    # 删除对应的图像文件
                    image_file = image_dir / f"{label_file.stem}.jpg"
                    if image_file.exists():
                        image_file.unlink()
                        print(f"删除空白图像: {image_file.name}")
                    
                    # 删除标签文件
                    label_file.unlink()
                    print(f"删除空标签: {label_file.name}")
                    empty_count += 1
                    
            except Exception as e:
                print(f"处理文件 {label_file} 时出错: {e}")
                
        print(f"{split} 集合中删除了 {empty_count} 个空白图像对")

def improved_split_dota(
    data_root: str, 
    save_dir: str, 
    rates: list = [1.0],  # 减少多尺度，避免过多切片
    gap: int = 300,       # 增加gap，减少重叠
    min_area_ratio: float = 0.1,  # 最小目标面积比例
    allow_background: bool = False  # 不允许背景图像
):
    """
    改进的DOTA数据集切割函数
    
    Args:
        data_root (str): 原始数据根目录
        save_dir (str): 保存目录
        rates (list): 缩放比例列表
        gap (int): 滑动窗口间隔
        min_area_ratio (float): 最小目标面积比例阈值
        allow_background (bool): 是否允许背景图像
    """
    # 修改split_dota模块中的默认参数
    import ultralytics.data.split_dota as split_module
    
    # 保存原始函数
    original_crop_and_save = split_module.crop_and_save
    
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
                    coords = lb[1:].reshape(-1, 2)
                    # 反归一化坐标
                    h, w = anno["ori_size"]
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    
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

                    with open(Path(lb_dir) / f"{new_name}.txt", "w", encoding="utf-8") as f:
                        for lb in valid_labels:
                            formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                            f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")
    
    # 临时替换函数
    split_module.crop_and_save = improved_crop_and_save
    
    try:
        # 执行切割
        split_trainval(
            data_root=data_root,
            save_dir=save_dir,
            rates=rates,
            gap=gap
        )
    finally:
        # 恢复原始函数
        split_module.crop_and_save = original_crop_and_save

if __name__ == "__main__":
    # 改进的切割参数
    print("开始改进的DOTA数据集切割...")
    improved_split_dota(
        data_root=r"D:\code\yolov11_obb_dota\zuhe\try",
        save_dir=r"D:\code\yolov11_obb_dota\zuhe\try_split_improved",
        rates=[1.0],  # 只使用原始尺度
        gap=400,      # 增加间隔
        min_area_ratio=0.05,  # 目标至少占窗口5%的面积
        allow_background=False  # 不允许背景图像
    )
    
    # 过滤空标签
    filter_empty_labels(r"D:\code\yolov11_obb_dota\zuhe\try_split_improved")
    
    print("改进的数据处理完成！")







