"""
Faster R-CNN 训练脚本
用于在 DOTAv2.0 数据集上进行对比实验
基于 mmdetection 框架实现
"""

import argparse
import os
import sys
from pathlib import Path

# 检查 mmdetection 是否安装
try:
    from mmdet.apis import init_detector, train_detector
    from mmengine.config import Config
    from mmengine.runner import Runner
    MMDetection_AVAILABLE = True
except ImportError:
    MMDetection_AVAILABLE = False
    print("⚠️  警告：mmdetection 未安装，请先安装：")
    print("   pip install mmdet mmengine mmcv")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on DOTAv2.0')
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    default_config = script_dir / 'config.yaml'
    default_mmdet_config = script_dir / 'faster_rcnn_config.py'
    
    parser.add_argument('--config', type=str, default=str(default_config), 
                       help=f'YAML 配置文件路径（默认: {default_config}）')
    parser.add_argument('--mmdet-config', type=str, default=str(default_mmdet_config),
                       help=f'mmdetection 配置文件路径（默认: {default_mmdet_config}）')
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


def create_mmdet_config_if_not_exists(config_path, yaml_config, seed=42):
    """如果 mmdetection 配置文件不存在，创建一个基础配置"""
    if Path(config_path).exists():
        return
    
    print(f"📝 创建 mmdetection 配置文件: {config_path}")
    
    # 从 YAML 配置中读取参数
    train_cfg = yaml_config.get('train', {})
    model_cfg = yaml_config.get('model', {})
    data_cfg = yaml_config.get('data', {})
    
    # 创建基础 Faster R-CNN 配置
    config_content = f'''# Faster R-CNN 配置文件
# 基于 mmdetection 框架

_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# 模型配置
model = dict(
    type='FasterRCNN',
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
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes={data_cfg.get('nc', 15)},
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
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
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
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
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

# 数据集配置
data_root = '{data_cfg.get("path", "data/DOTAv2.0")}'
data = dict(
    train=dict(
        ann_file=data_root + '/annotations/train.json',
        img_prefix=data_root + '/train/',
    ),
    val=dict(
        ann_file=data_root + '/annotations/val.json',
        img_prefix=data_root + '/val/',
    ),
    test=dict(
        ann_file=data_root + '/annotations/test.json',
        img_prefix=data_root + '/test/',
    ))

# 注意：如果使用 DOTA 数据集，需要先将旋转框转换为水平框（COCO 格式）
# 或者使用 mmrotate 框架进行旋转目标检测

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
    
    if not MMDetection_AVAILABLE:
        print("\n❌ 错误：mmdetection 未安装")
        print("请先安装 mmdetection：")
        print("  pip install mmdet mmengine mmcv")
        print("\n或者使用 conda：")
        print("  conda install -c conda-forge mmdet")
        return 1
    
    # 加载 YAML 配置
    yaml_config = load_yaml_config(args.config)
    
    # 检查并创建 mmdetection 配置文件
    mmdet_config_path = Path(args.mmdet_config)
    create_mmdet_config_if_not_exists(mmdet_config_path, yaml_config, args.seed)
    
    if not mmdet_config_path.exists():
        print(f"❌ 错误：mmdetection 配置文件不存在: {mmdet_config_path}")
        return 1
    
    # 加载 mmdetection 配置
    cfg = Config.fromfile(str(mmdet_config_path))
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = f'work_dirs/{yaml_config.get("name", "faster_rcnn_dota")}'
    
    # 设置设备
    cfg.device = f'cuda:{args.device}'
    
    # 设置随机种子
    if args.seed:
        cfg.randomness = dict(seed=args.seed, deterministic=True)
    
    # 恢复训练
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
    
    print("="*60)
    print("Faster R-CNN 训练配置")
    print("="*60)
    print(f"实验名称: {yaml_config.get('name', 'faster_rcnn_dota')}")
    print(f"mmdetection 配置: {mmdet_config_path}")
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
