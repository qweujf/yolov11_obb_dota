#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清除损坏的数据集缓存文件"""

import sys
from pathlib import Path

def clear_cache(data_root: str):
    """删除指定目录下的所有 .cache 文件"""
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"❌ 目录不存在: {data_path}")
        return
    
    cache_files = list(data_path.rglob("*.cache"))
    if not cache_files:
        print(f"✅ 未找到缓存文件")
        return
    
    print(f"📁 查找目录: {data_path}")
    print(f"🔍 找到 {len(cache_files)} 个缓存文件")
    
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            print(f"  ✅ 已删除: {cache_file}")
        except Exception as e:
            print(f"  ❌ 删除失败 {cache_file}: {e}")
    
    print(f"\n✅ 清理完成！重新运行训练将自动生成新的缓存文件")

if __name__ == "__main__":
    # 默认清除标准切分数据集的缓存
    default_data_root = r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard"
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = default_data_root
    
    clear_cache(data_root)

