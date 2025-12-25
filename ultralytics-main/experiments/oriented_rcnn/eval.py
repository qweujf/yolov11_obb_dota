"""
Oriented R-CNN 评估脚本
用于在 DOTAv2.0 数据集上评估模型性能
基于 mmrotate 框架实现
"""

import argparse
import sys
from pathlib import Path

# 检查 mmrotate 是否安装
MMRotate_AVAILABLE = False
missing_packages = []

try:
    import mmcv
except ImportError:
    missing_packages.append("mmcv")

try:
    import mmengine
    from mmengine.config import Config
    from mmengine.runner import Runner
except ImportError:
    missing_packages.append("mmengine")

try:
    from mmdet.datasets import build_dataloader, build_dataset
except ImportError:
    missing_packages.append("mmdet")

try:
    from mmrotate.apis import init_detector, inference_detector_by_patches
    from mmrotate.evaluation import DOTAMetric
except ImportError:
    missing_packages.append("mmrotate")

if missing_packages:
    MMRotate_AVAILABLE = False
    print("⚠️  警告：以下包未安装：")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\n请按以下步骤安装：")
    print("   1. 如果已安装 mmcv-full，可以跳过 mmcv")
    print("   2. 安装缺失的包：")
    if "mmcv" in missing_packages:
        print("      pip install mmcv-lite  # 或 mmcv-full（如果已安装可跳过）")
    if "mmengine" in missing_packages:
        print("      pip install mmengine")
    if "mmdet" in missing_packages:
        print("      pip install mmdet")
    if "mmrotate" in missing_packages:
        print("      pip install mmrotate")
    print("\n   或者一次性安装所有依赖：")
    print("      pip install mmcv-lite mmengine mmdet mmrotate")
else:
    MMRotate_AVAILABLE = True


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Oriented R-CNN on DOTAv2.0')
    script_dir = Path(__file__).parent
    default_config = script_dir / 'config.yaml'
    default_mmrotate_config = script_dir / 'oriented_rcnn_config.py'
    
    parser.add_argument('--config', type=str, default=str(default_config), 
                       help=f'YAML 配置文件路径（默认: {default_config}）')
    parser.add_argument('--mmrotate-config', type=str, default=str(default_mmrotate_config),
                       help=f'mmrotate 配置文件路径（默认: {default_mmrotate_config}）')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--device', type=str, default='0', help='GPU 设备ID')
    parser.add_argument('--out', type=str, default=None, help='结果输出文件路径')
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


def main():
    args = parse_args()
    
    if not MMRotate_AVAILABLE:
        print("\n❌ 错误：必要的包未安装")
        print("请按照上面的提示安装缺失的包")
        return 1
    
    # 加载 YAML 配置
    yaml_config = load_yaml_config(args.config)
    
    # 检查 mmrotate 配置文件
    mmrotate_config_path = Path(args.mmrotate_config)
    if not mmrotate_config_path.exists():
        print(f"❌ 错误：mmrotate 配置文件不存在: {mmrotate_config_path}")
        print("请先运行 train.py 生成配置文件")
        return 1
    
    # 检查检查点文件
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ 错误：检查点文件不存在: {ckpt_path}")
        return 1
    
    # 加载 mmrotate 配置
    cfg = Config.fromfile(str(mmrotate_config_path))
    cfg.device = f'cuda:{args.device}'
    
    print("="*60)
    print("Oriented R-CNN 评估配置")
    print("="*60)
    print(f"mmrotate 配置: {mmrotate_config_path}")
    print(f"检查点: {ckpt_path}")
    print(f"设备: {cfg.device}")
    print("="*60)
    
    # 初始化模型
    print("\n📦 加载模型...")
    model = init_detector(str(mmrotate_config_path), str(ckpt_path), device=cfg.device)
    
    # 构建测试数据集
    print("📊 构建测试数据集...")
    test_dataset = build_dataset(cfg.test_dataloader.dataset)
    test_dataloader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.test_dataloader.batch_size,
        workers_per_gpu=cfg.test_dataloader.num_workers,
        dist=False,
        shuffle=False
    )
    
    # 执行评估
    print("\n🔍 开始评估...")
    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        test_dataloader=test_dataloader,
        test_cfg=cfg.test_cfg,
        test_evaluator=cfg.test_evaluator
    )
    
    # 运行测试
    metrics = runner.test()
    
    print("\n✅ 评估完成！")
    print(f"📊 评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    if args.out:
        import json
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {args.out}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

