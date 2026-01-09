import os
import torch
import mmcv
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmrotate.datasets import build_dataset

# 导入你原训练脚本中的 get_config 和类别定义
import sys

sys.path.append(r'D:\code\yolov11_obb_dota\ultralytics-main\experiments\oriented_rcnn')
from train_obb import get_config, DOTA2_CLASSES


def main():
    # 1. 路径与参数配置
    PATHS = {
        'train': {'img': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\train\images',
                  'ann': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\train\annfiles'},
        'val': {'img': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\val\images',
                'ann': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\val\annfiles'},
        'work': r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\work_dirs'
    }
    CHECKPOINT_FILE = r'D:\code\yolov11_obb_dota\zuhe\rcnn_dota\work_dirs\epoch_3.pth'

    # 2. 直接获取配置对象而非读取文件
    print(">>> 正在构建配置对象...")
    cfg = get_config(PATHS['train'], PATHS['val'], PATHS['work'])

    # 3. 初始化模型
    print(">>> 正在初始化模型 (强制 CPU 模式以避开 NMS 错误)...")
    # 注意：我们直接传入 cfg 对象，而不是路径字符串
    model = init_detector(cfg, CHECKPOINT_FILE, device='cpu')

    # 4. 准备数据集
    print(">>> 正在加载验证集...")
    dataset = build_dataset(cfg.data.val)
    results = []

    # 5. 循环推理
    print(f">>> 开始执行推理 (共 {len(dataset)} 张图)...")
    # 为了快速看到结果，你可以先设为 range(100)
    for i in tqdm(range(len(dataset))):
        img_info = dataset.data_infos[i]
        img_path = os.path.join(dataset.img_prefix, img_info['filename'])

        # inference_detector 会处理图片读取和预处理
        result = inference_detector(model, img_path)
        results.append(result)

    # 6. 计算指标
    print("\n" + "=" * 50)
    print("      ORIENTED R-CNN 各类别 AP 评估结果")
    print("=" * 50)

    eval_results = dataset.evaluate(results, metric='mAP')

    for k, v in eval_results.items():
        print(f"{k.ljust(25)} : {v:.4f}")


if __name__ == '__main__':
    main()