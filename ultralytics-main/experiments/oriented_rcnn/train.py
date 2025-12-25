"""
Oriented R-CNN 训练脚本
用于在 DOTAv2.0 数据集上进行对比实验
基于 mmrotate 框架实现
"""

import argparse
import os
import sys
from pathlib import Path

# 检查 mmrotate 是否安装
MMRotate_AVAILABLE = False
missing_packages = []
import_errors = {}

# 检查 mmcv
try:
    import mmcv
    mmcv_ok = True
except Exception as e:
    missing_packages.append("mmcv")
    import_errors["mmcv"] = str(e)
    mmcv_ok = False

# 检查 mmengine
mmengine_ok = False
try:
    import mmengine
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmengine.utils import set_random_seed
        mmengine_ok = True
    except Exception as e:
        import_errors["mmengine"] = f"模块导入失败: {str(e)}"
except Exception as e:
    missing_packages.append("mmengine")
    import_errors["mmengine"] = str(e)

# 检查 mmdet
try:
    import mmdet
    mmdet_ok = True
except Exception as e:
    missing_packages.append("mmdet")
    import_errors["mmdet"] = str(e)
    mmdet_ok = False

# 检查 mmrotate
mmrotate_ok = False
try:
    from mmrotate.apis import init_detector, train_detector
    mmrotate_ok = True
except ImportError as e:
    missing_packages.append("mmrotate")
    error_msg = str(e)
    # 检查是否是 DLL 加载失败
    if "DLL load failed" in error_msg or "_ext" in error_msg:
        import_errors["mmrotate"] = f"DLL 加载失败（通常是 mmcv-full 的 C++ 扩展问题）: {error_msg}"
    else:
        import_errors["mmrotate"] = error_msg
except Exception as e:
    missing_packages.append("mmrotate")
    error_msg = str(e)
    if "DLL load failed" in error_msg or "_ext" in error_msg:
        import_errors["mmrotate"] = f"DLL 加载失败（通常是 mmcv-full 的 C++ 扩展问题）: {error_msg}"
    else:
        import_errors["mmrotate"] = error_msg

# 如果所有包都正常，设置可用标志
if mmcv_ok and mmengine_ok and mmdet_ok and mmrotate_ok:
    MMRotate_AVAILABLE = True
else:
    MMRotate_AVAILABLE = False
    print("⚠️  警告：以下包未安装或导入失败：")
    for pkg in missing_packages:
        error_msg = import_errors.get(pkg, "未知错误")
        print(f"   - {pkg}: {error_msg}")
    
    # 特殊处理 DLL 加载失败的情况
    if "mmrotate" in missing_packages and "DLL load failed" in import_errors.get("mmrotate", ""):
        print("\n" + "="*60)
        print("🔧 DLL 加载失败解决方案（Windows 常见问题）")
        print("="*60)
        print("\n问题原因：mmcv-full 的 C++ 扩展与当前环境不兼容")
        print("\n解决方案（按顺序尝试）：")
        print("\n方案1：使用 mmcv-lite（推荐，最简单）")
        print("   pip uninstall mmcv-full -y")
        print("   pip install mmcv-lite")
        print("   # mmcv-lite 不需要 C++ 扩展，兼容性更好")
        print("\n方案2：重新安装兼容的 mmcv-full")
        print("   # 先查看 PyTorch 和 CUDA 版本")
        print("   python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')\"")
        print("   # 然后安装对应版本的预编译 mmcv-full")
        print("   # 例如 CUDA 11.8 + PyTorch 2.0:")
        print("   pip uninstall mmcv-full -y")
        print("   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html")
        print("\n方案3：安装 Visual C++ Redistributable")
        print("   下载并安装：https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n方案4：检查环境变量")
        print("   确保 PATH 中包含必要的 DLL 路径")
        print("="*60)
    
    print("\n请按以下步骤安装：")
    print("   1. 如果已安装 mmcv-full，可以跳过 mmcv")
    print("   2. 安装缺失的包：")
    if "mmcv" in missing_packages:
        print("      pip install mmcv-lite  # 或 mmcv-full（如果已安装可跳过）")
    if "mmengine" in missing_packages:
        print("      pip install mmengine")
    if "mmdet" in missing_packages:
        print("      pip install mmdet")
    if "mmrotate" in missing_packages and "DLL load failed" not in import_errors.get("mmrotate", ""):
        print("      pip install mmrotate")
    print("\n   或者一次性安装所有依赖：")
    print("      pip install mmcv-lite mmengine mmdet mmrotate")
    print("\n   注意：请确保在正确的 conda 环境中安装")
    print(f"   当前 Python: {sys.executable}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Oriented R-CNN on DOTAv2.0')
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    default_config = script_dir / 'config.yaml'
    default_mmrotate_config = script_dir / 'oriented_rcnn_config.py'
    
    parser.add_argument('--config', type=str, default=str(default_config), 
                       help=f'YAML 配置文件路径（默认: {default_config}）')
    parser.add_argument('--mmrotate-config', type=str, default=str(default_mmrotate_config),
                       help=f'mmrotate 配置文件路径（默认: {default_mmrotate_config}）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='0', help='GPU 设备ID')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--work-dir', type=str, default=None, help='工作目录（保存日志和检查点）')
    return parser.parse_args()


def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    import yaml
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_mmrotate_config_if_not_exists(config_path, yaml_config, seed=42):
    """如果 mmrotate 配置文件不存在，创建一个基础配置"""
    if Path(config_path).exists():
        return
    
    print(f"📝 创建 mmrotate 配置文件: {config_path}")
    
    # 从 YAML 配置中读取参数
    train_cfg = yaml_config.get('train', {})
    model_cfg = yaml_config.get('model', {})
    data_cfg = yaml_config.get('data', {})
    
    # 创建基础 Oriented R-CNN 配置
    config_content = f'''# Oriented R-CNN 配置文件
# 基于 mmrotate 框架

_base_ = [
    'mmrotate::_base_/datasets/dotav2.py',
    'mmrotate::_base_/schedules/schedule_1x.py',
    'mmrotate::_base_/default_runtime.py'
]

# 模型配置
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNet',
        depth={50 if model_cfg.get('backbone', 'resnet50') == 'resnet50' else 101},
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='oc',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            target_means=[.0, .0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes={data_cfg.get('nc', 15)},
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_range='oc',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms_rotated', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms_rotated', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

# 数据集配置
data_root = '{data_cfg.get("path", "data/DOTAv2.0")}'
data = dict(
    train=dict(
        type='DOTADataset',
        ann_file=data_root + '/train/annfiles/',
        img_prefix=data_root + '/train/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(type='mmdet.PackDetInputs')
        ]),
    val=dict(
        type='DOTADataset',
        ann_file=data_root + '/val/annfiles/',
        img_prefix=data_root + '/val/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='mmdet.PackDetInputs')
        ]),
    test=dict(
        type='DOTADataset',
        ann_file=data_root + '/test/annfiles/',
        img_prefix=data_root + '/test/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='mmdet.PackDetInputs')
        ]))

# 训练配置
train_dataloader = dict(batch_size={train_cfg.get('batch_size', 4)})
val_dataloader = dict(batch_size={train_cfg.get('batch_size', 4)})
test_dataloader = dict(batch_size={train_cfg.get('batch_size', 4)})

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr={train_cfg.get('lr0', 0.01)},
        momentum={train_cfg.get('momentum', 0.937)},
        weight_decay={train_cfg.get('weight_decay', 0.0005)}))

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end={int(train_cfg.get('warmup_epochs', 3) * 1000)}),
    dict(
        type='MultiStepLR',
        begin=0,
        end={train_cfg.get('epochs', 300)},
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 训练设置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={train_cfg.get('epochs', 300)}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 随机种子
randomness = dict(seed={seed})
'''
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已创建: {config_path}")


def main():
    args = parse_args()
    
    if not MMRotate_AVAILABLE:
        print("\n❌ 错误：必要的包未安装")
        print("请按照上面的提示安装缺失的包")
        return 1
    
    # 加载 YAML 配置
    yaml_config = load_yaml_config(args.config)
    
    # 检查并创建 mmrotate 配置文件
    mmrotate_config_path = Path(args.mmrotate_config)
    create_mmrotate_config_if_not_exists(mmrotate_config_path, yaml_config, args.seed)
    
    if not mmrotate_config_path.exists():
        print(f"❌ 错误：mmrotate 配置文件不存在: {mmrotate_config_path}")
        return 1
    
    # 加载 mmrotate 配置
    cfg = Config.fromfile(str(mmrotate_config_path))
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = f'work_dirs/{yaml_config.get("name", "oriented_rcnn_dota")}'
    
    # 设置设备
    cfg.device = f'cuda:{args.device}'
    
    # 设置随机种子
    if args.seed:
        set_random_seed(args.seed, deterministic=True)
        cfg.randomness = dict(seed=args.seed, deterministic=True)
    
    # 恢复训练
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
    
    print("="*60)
    print("Oriented R-CNN 训练配置")
    print("="*60)
    print(f"实验名称: {yaml_config.get('name', 'oriented_rcnn_dota')}")
    print(f"mmrotate 配置: {mmrotate_config_path}")
    print(f"工作目录: {cfg.work_dir}")
    print(f"设备: {cfg.device}")
    print(f"随机种子: {args.seed}")
    print(f"训练轮数: {cfg.train_cfg.max_epochs}")
    print(f"批次大小: {cfg.train_dataloader.batch_size}")
    print("="*60)
    
    # 创建 Runner 并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    print("\n✅ 训练完成！")
    print(f"📁 结果保存在: {cfg.work_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

