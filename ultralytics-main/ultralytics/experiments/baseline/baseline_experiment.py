#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOTA-v2.0数据集YOLOv11-OBB基准实验训练脚本
基于论文标准参数配置
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载（规避崩溃）
os.environ['OMP_NUM_THREADS'] = '1'          # 限制 OpenMP 线程，避免再触发冲突
os.environ['MKL_NUM_THREADS'] = '1'          # 同上，限制 MKL 线程

import torch
from pathlib import Path
import yaml
import os
os.environ['ULTRALYTICS_OFFLINE'] = '1'  # 禁止一切下载
import sys

# 将 ultralytics-main 添加到 Python 路径
ultralytics_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ultralytics-main')
print("ultralytics-main 路径:", ultralytics_path)

# 验证该路径是否存在
if os.path.exists(ultralytics_path):
    print("✅ 路径存在")
else:
    print("❌ 路径不存在，请检查！")
if ultralytics_path not in sys.path:
    sys.path.insert(0, ultralytics_path)

# 现在可以正常导入
from ultralytics import YOLO

def create_baseline_config():
    return {
        # 基础
        # 'model': r'D:\code\yolov11_obb_dota\ultralytics-main\yolo11n-obb.pt',
        'data': 'experiments/baseline/DOTAv2.yaml',
        'name': 'dota_baseline_img1024',
        'device': '0',
        'save': True,
        'exist_ok': True,
        'seed': 42,

        # 训练
        'epochs': 300,
        'batch': 2,
        'imgsz': 1024,
        'workers': 4,
        'val': True,
        'plots': True,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'single_cls': False,
        'rect': False,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        # 注意：不要加 val_period、deterministic、overlap_mask、mask_ratio、label_smoothing

        # 优化器
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # 数据增强（OBB 不做旋转/剪切）
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,

        # 损失（仅保留支持的）
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
def create_enhanced_config():
    cfg = create_baseline_config()
    cfg.update({
        'model': 'ultralytics/cfg/models/11/yolo11-obb-enhanced.yaml',
        'name': 'dota_enhanced_img1024',
        'lr0': 0.008,
        'weight_decay': 0.0008,
    })
    return cfg

def create_ablation_configs():
    cfgs = {}
    cfgs['baseline']   = create_baseline_config()
    cfgs['enhanced']   = create_enhanced_config()

    cfgs['multiscale'] = create_baseline_config()
    cfgs['multiscale'].update({
        'multi_scale': True,
        'name': 'dota_multiscale_img1024',
    })

    cfgs['strong_aug'] = create_baseline_config()
    cfgs['strong_aug'].update({
        'hsv_h': 0.02, 'hsv_s': 0.8, 'hsv_v': 0.5,
        'translate': 0.2, 'scale': 0.8,
        'mixup': 0.1, 'copy_paste': 0.1,
        'name': 'dota_strong_aug_img1024',
    })

    cfgs['small_batch'] = create_baseline_config()
    cfgs['small_batch'].update({
        'batch': 8,
        'lr0': 0.005,
        'name': 'dota_small_batch_img1024',
    })
    return cfgs

def run_baseline_experiment(config_name='baseline'):
    configs = create_ablation_configs()
    cfg = configs[config_name]

    # 1) 用本地权重先创建模型（把路径改成你真实存在的本地 yolo11n-obb.pt）
    local_pt = r'D:\code\yolov11_obb_dota\ultralytics-main\yolo11n-obb.pt'
    assert os.path.exists(local_pt), f'本地权重不存在: {local_pt}'
    model = YOLO(local_pt)

    # 2) 防止后面config里再把model覆盖成yaml导致下载
    cfg.pop('model', None)
    cfg['pretrained'] = False  # 显式禁用再次拉预训练

    # 3) 开始训练
    results = model.train(**cfg)
    return results
def run_all_experiments():
    """运行所有基准实验"""
    experiments = ['baseline', 'enhanced', 'multiscale', 'strong_aug']
    
    for exp_name in experiments:
        try:
            print(f"\n{'='*60}")
            print(f"🧪 开始实验: {exp_name}")
            print(f"{'='*60}")
            
            results = run_baseline_experiment(exp_name)
            
            print(f"✅ {exp_name} 实验完成")
            
        except Exception as e:
            print(f"❌ {exp_name} 实验失败: {e}")
            continue
    
    print("\n🎉 所有基准实验完成！")

def create_training_script():
    """创建训练脚本"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动基准实验训练脚本
"""

from data_processing.baseline_experiment import run_baseline_experiment, run_all_experiments

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 运行指定实验
        exp_name = sys.argv[1]
        run_baseline_experiment(exp_name)
    else:
        # 运行所有实验
        run_all_experiments()
'''
    
    with open('data_processing/run_baseline.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("📝 已创建快速启动脚本: data_processing/run_baseline.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 运行指定实验
        exp_name = sys.argv[1]
        run_baseline_experiment(exp_name)
    else:
        # 运行基础实验
        run_baseline_experiment('baseline')
    
    # 创建快速启动脚本
    create_training_script()
