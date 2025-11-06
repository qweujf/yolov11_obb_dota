"""
清理空白图像的工具脚本
用于清理已经生成的数据集中的空白图像
"""

import os
from pathlib import Path
from ultralytics.utils import TQDM

def clean_empty_images(data_root: str, dry_run: bool = True):
    """
    清理数据集中的空白图像（没有对应标签或标签为空的图像）
    
    Args:
        data_root (str): 数据根目录
        dry_run (bool): 是否只是预览而不实际删除
    """
    print(f"{'预览' if dry_run else '清理'}空白图像...")
    
    for split in ["train", "val"]:
        image_dir = Path(data_root) / "images" / split
        label_dir = Path(data_root) / "labels" / split
        
        if not image_dir.exists():
            print(f"图像目录不存在: {image_dir}")
            continue
        if not label_dir.exists():
            print(f"标签目录不存在: {label_dir}")
            continue
            
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        removed_count = 0
        
        print(f"\n处理 {split} 集合...")
        for image_file in TQDM(image_files, desc=f"Checking {split} images"):
            label_file = label_dir / f"{image_file.stem}.txt"
            
            should_remove = False
            reason = ""
            
            # 检查是否存在对应的标签文件
            if not label_file.exists():
                should_remove = True
                reason = "无对应标签文件"
            else:
                # 检查标签文件是否为空
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            should_remove = True
                            reason = "标签文件为空"
                except Exception as e:
                    should_remove = True
                    reason = f"读取标签文件失败: {e}"
            
            if should_remove:
                if dry_run:
                    print(f"[预览] 将删除: {image_file.name} - {reason}")
                else:
                    try:
                        image_file.unlink()
                        if label_file.exists():
                            label_file.unlink()
                        print(f"已删除: {image_file.name} - {reason}")
                    except Exception as e:
                        print(f"删除失败 {image_file.name}: {e}")
                
                removed_count += 1
        
        print(f"{split} 集合中{'预计删除' if dry_run else '已删除'} {removed_count} 个空白图像")

def count_images_and_labels(data_root: str):
    """统计图像和标签数量"""
    print("\n数据集统计:")
    
    for split in ["train", "val"]:
        image_dir = Path(data_root) / "images" / split
        label_dir = Path(data_root) / "labels" / split
        
        if image_dir.exists():
            image_count = len(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        else:
            image_count = 0
            
        if label_dir.exists():
            label_count = len(list(label_dir.glob("*.txt")))
            # 统计空标签文件
            empty_labels = 0
            for label_file in label_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        if not f.read().strip():
                            empty_labels += 1
                except:
                    empty_labels += 1
        else:
            label_count = 0
            empty_labels = 0
        
        print(f"{split:>5}: 图像 {image_count:>6}, 标签 {label_count:>6}, 空标签 {empty_labels:>6}")

if __name__ == "__main__":
    # 指定数据集路径
    data_path = r"D:\code\yolov11_obb_dota\zuhe\try_split"
    
    # 首先统计当前数据集
    count_images_and_labels(data_path)
    
    # 预览将要删除的文件
    print("\n=== 预览模式 ===")
    clean_empty_images(data_path, dry_run=True)
    
    # 询问是否执行实际清理
    response = input("\n是否执行实际清理？(y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n=== 执行清理 ===")
        clean_empty_images(data_path, dry_run=False)
        
        # 清理后统计
        print("\n=== 清理后统计 ===")
        count_images_and_labels(data_path)
    else:
        print("已取消清理操作")







