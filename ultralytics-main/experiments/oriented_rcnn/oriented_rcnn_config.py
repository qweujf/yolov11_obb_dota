# Oriented R-CNN 配置文件
# 基于 mmrotate 框架的完整独立配置
total_epochs = 300
workflow = [('train', 1)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs)
device = 'cuda'
gpu_ids = [0]
seed = 42
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
resume_from = None
load_from = None
auto_resume = False
log_level = 'INFO'
# 模型配置
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
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
        version='le90',
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
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
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
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

# 数据集配置（显式指定 scope）
# 数据集类别
dota_v2_classes = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'swimming-pool', 'container-crane', 'airport', 'helipad'
)
data_root = 'D:/code/yolov11_obb_dota/zuhe/rcnn_dota'
data = dict(
    train=dict(
        type='DOTADataset',
        ann_file=r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\train\annfiles',
        img_prefix=r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\train\images',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_label=True),
            dict(type='mmdet.Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='mmdet.RandomFlip', flip_ratio=0.5),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=dota_v2_classes),
    val=dict(
        type='DOTADataset',
        ann_file='D:/code/yolov11_obb_dota/zuhe/split_dota_1024_standard/labels/val',
        img_prefix='D:/code/yolov11_obb_dota/zuhe/split_dota_1024_standard/images/val',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_label=True),
            dict(type='mmdet.Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=dota_v2_classes),
    test=dict(
        type='DOTADataset',
        ann_file=data_root + '/test/annfiles/',
        img_prefix=data_root + '/test/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_label=True),
            dict(type='mmdet.Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]))

# 训练配置
train_dataloader = dict(
    batch_size=4,
    num_workers=4)
val_dataloader = dict(
    batch_size=4,
    num_workers=4)
test_dataloader = dict(
    batch_size=4,
    num_workers=4)

# 优化器配置
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率调度器
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 250])

# 训练设置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 验证指标
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = dict(type='DOTAMetric', metric='mAP')

# 默认运行时配置
# 注释掉 default_scope，让系统自动查找（会先查找 mmrotate，然后 fallback 到 mmdet）
# default_scope = 'mmrotate'

# 随机种子
randomness = dict(seed=42, deterministic=False)
