import os
import os.path as osp
import numpy as np
import torch
import cv2
from mmcv import Config
from mmdet.apis import set_random_seed
from mmrotate.datasets import build_dataset, DOTADataset
from mmrotate.models import build_detector
from mmrotate.apis import train_detector
from mmrotate.datasets.builder import DATASETS


# ==========================================
# 1. 数据集解析器 (保持您的逻辑，修正潜在的小问题)
# ==========================================
@DATASETS.register_module(force=True)
class YOLOTransDOTADataset(DOTADataset):
    def load_annotations(self, ann_file):
        data_infos = []
        try:
            image_files = os.listdir(self.img_prefix)
            name_to_file = {osp.splitext(f)[0].strip(): f for f in image_files}
            print(f">>> [数据检查] 磁盘图片: {len(image_files)}, 标签文件: {len(os.listdir(ann_file))}")
        except Exception as e:
            return []

        ann_list = [f for f in os.listdir(ann_file) if f.endswith('.txt') and f != 'classes.txt']
        cls_map = {c.lower().strip(): i for i, c in enumerate(self.CLASSES)}

        for idx, ann_name in enumerate(ann_list):
            img_id = ann_name[:-4].strip()
            filename = name_to_file.get(img_id)
            if filename is None: continue

            bboxes, labels = [], []
            with open(osp.join(ann_file, ann_name), 'r', encoding='utf-8-sig') as f:
                for line in f:
                    items = line.strip().split()
                    if len(items) < 9: continue
                    try:
                        pts = np.array([float(x) for x in items[:8]], dtype=np.float32).reshape(4, 2)
                        rect = cv2.minAreaRect(pts)
                        (cx, cy), (w, h), angle = rect
                        theta = angle * np.pi / 180.0
                        obb = np.array([cx, cy, w, h, theta], dtype=np.float32)

                        raw_label = items[8].lower().strip()
                        if raw_label in cls_map:
                            bboxes.append(obb.flatten())
                            labels.append(cls_map[raw_label])
                    except:
                        continue

            if len(bboxes) == 0: continue

            data_infos.append(dict(
                id=img_id, filename=filename, width=1024, height=1024,
                flip=False, flip_direction=None,
                ann=dict(
                    bboxes=np.array(bboxes, dtype=np.float32),
                    labels=np.array(labels).astype(np.int64),
                    bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
                    labels_ignore=np.zeros((0,), dtype=np.int64)
                )
            ))
        print(f">>> [解析结束] 成功匹配有效图像: {len(data_infos)} 张")
        return data_infos


# ==========================================
# 2. 配置与优化设置
# ==========================================
DOTA2_CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                 'basketball-court', 'storage-tank', 'soccer-ball-field',
                 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                 'container-crane', 'airport', 'helipad')


def get_config(train_paths, val_paths, work_dir):
    num_classes = len(DOTA2_CLASSES)
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(type='RRandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='RResize'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ])
    ]

    cfg_dict = dict(
        log_level='INFO',
        # --- FP16 混合精度训练：大幅提速并减少显存占用 ---
        fp16=dict(loss_scale=512.),

        resume_from=None,
        load_from=None,
        auto_resume=False,
        gpu_ids=[0],
        work_dir=work_dir,
        seed=42,
        device='cuda',
        workflow=[('train', 1)],

        model=dict(
            type='OrientedRCNN',
            backbone=dict(
                type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
                frozen_stages=1, norm_eval=True, style='pytorch',
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
            neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
            rpn_head=dict(
                type='OrientedRPNHead', in_channels=256, feat_channels=256, version='le90',
                anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
                                      strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(type='MidpointOffsetCoder', target_means=[0.] * 6,
                                target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1111, loss_weight=1.0)),
            roi_head=dict(
                type='OrientedStandardRoIHead', version='le90',
                bbox_roi_extractor=dict(
                    type='RotatedSingleRoIExtractor',
                    roi_layer=dict(type='RoIAlignRotated', output_size=7, sampling_ratio=0),
                    out_channels=256, featmap_strides=[4, 8, 16, 32]),
                bbox_head=dict(
                    type='RotatedShared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', target_means=[0.] * 5,
                                    target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                    reg_class_agnostic=False,
                    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),

            # --- 优化训练速度的关键：减少 Proposal 数量 ---
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(type='MaxIoUAssigner', iou_calculator=dict(type='BboxOverlaps2D'), pos_iou_thr=0.7,
                                  neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                    sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1,
                                 add_gt_as_proposals=False),
                    allowed_border=0, pos_weight=-1, debug=False),
                rpn_proposal=dict(
                    nms_pre=1000,  # 减少预选框数量
                    max_per_img=500,  # 减少最终送入 RCNN 的框，大幅缓解 24GB 显存溢出
                    nms=dict(type='nms', iou_threshold=0.8),
                    min_bbox_size=0),
                rcnn=dict(
                    assigner=dict(type='MaxIoUAssigner', iou_calculator=dict(type='RBboxOverlaps2D'), pos_iou_thr=0.5,
                                  neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
                    sampler=dict(type='RRandomSampler', num=256, pos_fraction=0.25, neg_pos_ub=-1,
                                 add_gt_as_proposals=True),
                    pos_weight=-1, debug=False)),
            test_cfg=dict(
                rpn=dict(nms_pre=800, max_per_img=300, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
                rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000))
        ),

        data=dict(
            samples_per_gpu=2,  # 开启 FP16 后，尝试恢复为 2，如果还是崩就改回 1
            workers_per_gpu=4,
            train=dict(type='YOLOTransDOTADataset', version='le90', classes=DOTA2_CLASSES, ann_file=train_paths['ann'],
                       img_prefix=train_paths['img'], pipeline=train_pipeline),
            val=dict(type='YOLOTransDOTADataset', version='le90', classes=DOTA2_CLASSES, ann_file=val_paths['ann'],
                     img_prefix=val_paths['img'], pipeline=test_pipeline),
            test=dict(type='YOLOTransDOTADataset', version='le90', classes=DOTA2_CLASSES, ann_file=val_paths['ann'],
                      img_prefix=val_paths['img'], pipeline=test_pipeline)
        ),

        # --- 减少评估频率，平衡速度与反馈 ---
        evaluation=dict(
            interval=3,  # 每 3 个 Epoch 评估一次，避免频繁卡住
            metric='mAP',
            classwise=True,
            gpu_collect=True  # 增加安全性
        ),

        optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
        optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)),
        lr_config=dict(policy='step', warmup='linear', warmup_iters=500, step=[24, 33]),
        runner=dict(type='EpochBasedRunner', max_epochs=36),
        checkpoint_config=dict(interval=1),
        log_config=dict(interval=50, hooks=[dict(type='TextLoggerHook')]),
    )
    return Config(cfg_dict)


def main():
    PATHS = {
        'train': {'img': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\train\images',
                  'ann': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\train\annfiles'},
        'val': {'img': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\val\images',
                'ann': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\val\annfiles'},
        'work': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\work_dirs'
    }

    cfg = get_config(PATHS['train'], PATHS['val'], PATHS['work'])
    # --- 续训设置：指向最新的 Epoch 2 ---
    cfg.resume_from = r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\work_dirs\epoch_3.pth'

    set_random_seed(cfg.seed, deterministic=True)

    model = build_detector(cfg.model)
    model.CLASSES = DOTA2_CLASSES
    train_ds = build_dataset(cfg.data.train)
    print(">>> 优化配置已加载（FP16+Proposal限制），正在加速恢复训练...")
    train_detector(model, [train_ds], cfg, validate=True)


if __name__ == '__main__':
    main()