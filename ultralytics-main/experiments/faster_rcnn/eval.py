"""
Faster R-CNN 评估脚本
用于评估训练好的模型在 DOTAv2.0 测试集上的性能
基于 mmdetection 框架实现
"""

import argparse
import sys
from pathlib import Path

# 检查 mmdetection 是否安装
try:
    from mmdet.apis import init_detector
    from mmengine.config import Config
    from mmengine.runner import Runner
    MMDetection_AVAILABLE = True
except ImportError:
    MMDetection_AVAILABLE = False
    print("⚠️  警告：mmdetection 未安装，请先安装：")
    print("   pip install mmdet mmengine mmcv")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN on DOTAv2.0')
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    default_config = script_dir / 'config.yaml'
    default_mmdet_config = script_dir / 'faster_rcnn_config.py'
    
    parser.add_argument('--config', type=str, default=str(default_config),
                       help=f'YAML 配置文件路径（默认: {default_config}）')
    parser.add_argument('--mmdet-config', type=str, default=str(default_mmdet_config),
                       help=f'mmdetection 配置文件路径（默认: {default_mmdet_config}）')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--device', type=str, default='0', help='GPU 设备ID')
    parser.add_argument('--out', type=str, default=None, help='结果输出文件路径（可选）')
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
    
    if not MMDetection_AVAILABLE:
        print("\n❌ 错误：mmdetection 未安装")
        print("请先安装 mmdetection：")
        print("  pip install mmdet mmengine mmcv")
        return 1
    
    # 检查检查点文件
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ 错误：检查点文件不存在: {ckpt_path}")
        return 1
    
    # 加载 YAML 配置
    yaml_config = load_yaml_config(args.config)
    
    # 加载 mmdetection 配置
    mmdet_config_path = Path(args.mmdet_config)
    if not mmdet_config_path.exists():
        print(f"❌ 错误：mmdetection 配置文件不存在: {mmdet_config_path}")
        print("请先运行 train.py 生成配置文件，或手动创建配置文件")
        return 1
    
    cfg = Config.fromfile(str(mmdet_config_path))
    
    # 设置设备
    cfg.device = f'cuda:{args.device}'
    
    print("="*60)
    print("Faster R-CNN 模型评估")
    print("="*60)
    print(f"检查点: {ckpt_path}")
    print(f"mmdetection 配置: {mmdet_config_path}")
    print(f"设备: {cfg.device}")
    print("="*60)
    
    # 初始化模型
    print("\n📦 加载模型...")
    model = init_detector(str(mmdet_config_path), str(ckpt_path), device=cfg.device)
    print("✅ 模型加载完成")
    
    # 创建 Runner 进行评估
    runner = Runner.from_cfg(cfg)
    
    # 运行测试
    print("\n🔍 开始评估...")
    metrics = runner.test()
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("评估完成，详细结果请查看日志")
    print("="*60)
    
    # 保存结果
    if args.out:
        import json
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics if metrics else {}, f, indent=2)
        print(f"\n💾 结果已保存到: {output_path}")
    
    print("\n✅ 评估完成！")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
