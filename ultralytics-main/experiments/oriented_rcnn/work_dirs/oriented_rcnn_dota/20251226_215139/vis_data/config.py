data = dict(
    test=dict(
        ann_file='data/DOTAv2.0/test/annfiles/',
        img_prefix='data/DOTAv2.0/test/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                img_scale=(
                    1024,
                    1024,
                ), keep_ratio=True, type='mmdet.Resize'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    train=dict(
        ann_file='data/DOTAv2.0/train/annfiles/',
        img_prefix='data/DOTAv2.0/train/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                img_scale=(
                    1024,
                    1024,
                ), keep_ratio=True, type='mmdet.Resize'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    val=dict(
        ann_file='data/DOTAv2.0/val/annfiles/',
        img_prefix='data/DOTAv2.0/val/images/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                img_scale=(
                    1024,
                    1024,
                ), keep_ratio=True, type='mmdet.Resize'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'))
data_root = 'data/DOTAv2.0'
device = 'cuda:0'
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                angle_range='le90',
                edge_swap=True,
                norm_factor=None,
                proj_xy=True,
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.1,
                ),
                type='DeltaXYWHTRBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=15,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='RotatedShared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(
                output_size=7, sampling_ratio=0, type='RoIAlignRotated'),
            type='RotatedSingleRoIExtractor'),
        type='OrientedStandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='MidpointOffsetCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='OrientedRPNHead',
        version='le90'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=2000,
            nms=dict(iou_threshold=0.1, type='nms_rotated'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms_rotated'),
            nms_pre=2000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.8, type='nms_rotated'),
            nms_pre=2000)),
    type='OrientedRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.937, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=300,
        gamma=0.1,
        milestones=[
            200,
            250,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=True, seed=42)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(batch_size=4, num_workers=4)
test_evaluator = dict(metric='mAP', type='DOTAMetric')
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(batch_size=4, num_workers=4)
val_cfg = dict(type='ValLoop')
val_dataloader = dict(batch_size=4, num_workers=4)
val_evaluator = dict(metric='mAP', type='DOTAMetric')
work_dir = 'work_dirs/oriented_rcnn_dota'
