"""
计算模型 GFLOPs 的脚本

支持两种方法：
1. thop 库（推荐，速度快）
2. torch.profiler（备选，需要 PyTorch >= 2.0）

使用方法：
    python calculate_gflops.py
"""

import os
import sys
from pathlib import Path
from copy import deepcopy

# 设置环境变量禁用自动下载（必须在导入 YOLO 之前）
os.environ["YOLO_OFFLINE"] = "True"

root_path = str(Path(__file__).parents[2])
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler, de_parallel

# ============================================================================
# 配置区域
# ============================================================================

# 模型路径（可以是 .yaml 或 .pt）
MODEL_PATH = r"configs/model/yolov11_obb.yaml"

# 计算 GFLOPs 时使用的输入尺寸（正方形）
IMG_SIZE = 1024

# 动态通道激活阈值（如果需要测试带动态激活的模型）
# 设置为 None 表示不使用动态激活
DYNAMIC_ACTIVATION_THRESHOLD = None  # 例如: 0.15, 0.20, 0.30 等

# 计算方法：'thop' 或 'torch_profiler'
METHOD = 'thop'  # 'thop' 速度快，'torch_profiler' 需要 PyTorch >= 2.0

# ============================================================================


def calculate_gflops_thop(model, imgsz=1024):
    """使用 thop 库计算 GFLOPs"""
    try:
        import thop
    except ImportError:
        print("❌ thop 库未安装，请运行: pip install thop")
        return None
    
    print(f"📊 使用 thop 库计算 GFLOPs (imgsz={imgsz})...")
    
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        device = p.device
        
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        
        try:
            # 方法 1: 使用 stride-based 输入（更高效）
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
            im = torch.empty((1, p.shape[1], stride, stride), device=device)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
            gflops = flops * imgsz[0] / stride * imgsz[1] / stride
            print(f"   ✅ 使用 stride-based 方法计算")
            return gflops
        except Exception as e:
            # 方法 2: 使用实际图像尺寸
            print(f"   ⚠️ stride-based 方法失败，尝试使用实际图像尺寸...")
            im = torch.empty((1, p.shape[1], *imgsz), device=device)
            gflops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
            print(f"   ✅ 使用实际图像尺寸方法计算")
            return gflops
    except Exception as e:
        print(f"   ❌ thop 计算失败: {e}")
        return None


def calculate_gflops_torch_profiler(model, imgsz=1024):
    """使用 torch.profiler 计算 GFLOPs（需要 PyTorch >= 2.0）"""
    if not hasattr(torch, 'profiler') or not hasattr(torch.profiler, 'profile'):
        print("❌ torch.profiler 不可用，需要 PyTorch >= 2.0")
        return None
    
    print(f"📊 使用 torch.profiler 计算 GFLOPs (imgsz={imgsz})...")
    
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        device = p.device
        
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        
        try:
            # 方法 1: 使用 stride-based 输入
            stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2
            im = torch.empty((1, p.shape[1], stride, stride), device=device)
            with torch.profiler.profile(with_flops=True) as prof:
                model(im)
            flops = sum(x.flops for x in prof.key_averages()) / 1e9
            gflops = flops * imgsz[0] / stride * imgsz[1] / stride
            print(f"   ✅ 使用 stride-based 方法计算")
            return gflops
        except Exception as e:
            # 方法 2: 使用实际图像尺寸
            print(f"   ⚠️ stride-based 方法失败，尝试使用实际图像尺寸...")
            im = torch.empty((1, p.shape[1], *imgsz), device=device)
            with torch.profiler.profile(with_flops=True) as prof:
                model(im)
            gflops = sum(x.flops for x in prof.key_averages()) / 1e9
            print(f"   ✅ 使用实际图像尺寸方法计算")
            return gflops
    except Exception as e:
        print(f"   ❌ torch.profiler 计算失败: {e}")
        return None


def main():
    print("=" * 60)
    print("模型 GFLOPs 计算工具")
    print("=" * 60)
    
    # 设置动态激活阈值（如果指定）
    if DYNAMIC_ACTIVATION_THRESHOLD is not None:
        os.environ["DYNAMIC_ACTIVATION_THRESHOLD"] = str(DYNAMIC_ACTIVATION_THRESHOLD)
        print(f"🔧 设置动态通道激活阈值: τ = {DYNAMIC_ACTIVATION_THRESHOLD}")
    else:
        # 清除环境变量（如果之前设置过）
        os.environ.pop("DYNAMIC_ACTIVATION_THRESHOLD", None)
        print("🔧 不使用动态通道激活")
    
    print(f"📄 加载模型: {MODEL_PATH}")
    
    # 加载模型
    model_path = Path(MODEL_PATH)
    if not model_path.is_absolute():
        # 相对路径，基于脚本所在目录
        repo_root = Path(__file__).parents[2]
        model_path = repo_root / MODEL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
    
    model = YOLO(str(model_path), task='obb')
    core_model = getattr(model, "model", model)
    
    # 计算参数量
    n_params = sum(p.numel() for p in core_model.parameters())
    n_params_m = n_params / 1e6
    
    print(f"\n📊 模型信息:")
    print(f"   参数量: {n_params:,} ({n_params_m:.2f}M)")
    
    # 计算 GFLOPs
    print(f"\n📐 计算 GFLOPs (imgsz={IMG_SIZE})...")
    
    if METHOD == 'thop':
        gflops = calculate_gflops_thop(core_model, imgsz=IMG_SIZE)
    elif METHOD == 'torch_profiler':
        gflops = calculate_gflops_torch_profiler(core_model, imgsz=IMG_SIZE)
    else:
        print(f"❌ 未知的计算方法: {METHOD}")
        return
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📊 计算结果")
    print("=" * 60)
    print(f"   参数量: {n_params:,} ({n_params_m:.2f}M)")
    if gflops is not None:
        print(f"   GFLOPs: {gflops:.2f} G (imgsz={IMG_SIZE})")
    else:
        print(f"   GFLOPs: 计算失败")
    
    if DYNAMIC_ACTIVATION_THRESHOLD is not None:
        print(f"   动态激活阈值: τ = {DYNAMIC_ACTIVATION_THRESHOLD}")
        print(f"   ⚠️  注意: 动态激活在推理时生效，GFLOPs 会降低")
        print(f"   ⚠️  但 thop 可能无法准确统计动态跳过的计算")
    
    print("=" * 60)
    
    # 同时显示模型摘要
    print("\n📋 模型摘要:")
    if hasattr(core_model, "info"):
        core_model.info(verbose=True, imgsz=IMG_SIZE)
    else:
        print("   ⚠️ 模型不支持 info() 方法")


if __name__ == "__main__":
    main()

