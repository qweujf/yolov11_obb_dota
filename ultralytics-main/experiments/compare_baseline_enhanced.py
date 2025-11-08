#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 Baseline 和 MCAttention 模型的实验结果
包含所有评价指标：mAP50, mAP50-95, 参数量, FLOPs, 推理速度
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

# 设置环境变量禁用自动下载
os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_training_results(exp_name: str) -> Optional[Dict]:
    """加载训练结果（mAP等指标）"""
    repo_root = Path(__file__).resolve().parent
    exp_dir = repo_root / "experiments" / exp_name
    results_file = exp_dir / "runs" / "train" / "results.csv"
    
    if not results_file.exists():
        return None
    
    try:
        df = pd.read_csv(results_file)
        if len(df) == 0:
            return None
        
        # 获取最佳结果（通常是最后一行，或mAP最高的）
        best_idx = df['metrics/mAP50-95(B)'].idxmax() if 'metrics/mAP50-95(B)' in df.columns else len(df) - 1
        best_row = df.iloc[best_idx]
        
        return {
            "mAP50": best_row.get("metrics/mAP50(B)", None),
            "mAP50-95": best_row.get("metrics/mAP50-95(B)", None),
            "epochs": len(df),
            "best_epoch": int(best_row.get("epoch", best_idx + 1)),
        }
    except Exception as e:
        print(f"⚠️ 加载 {exp_name} 训练结果失败: {e}")
        return None


def analyze_model_complexity(exp_name: str, imgsz: int = 1024) -> Optional[Dict]:
    """分析模型复杂度（参数量、FLOPs）"""
    repo_root = Path(__file__).resolve().parent
    
    # 确定模型路径
    if exp_name == "baseline":
        model_path = repo_root / "configs" / "model" / "yolov11_obb.yaml"
    elif exp_name == "mca_attention":
        model_path = repo_root / "configs" / "model" / "yolo11-obb-mca-only.yaml"
    else:
        return None
    
    # 尝试加载训练好的权重
    exp_dir = repo_root / "experiments" / exp_name
    weights_file = exp_dir / "runs" / "train" / "weights" / "best.pt"
    
    if weights_file.exists():
        model_path = str(weights_file)
        print(f"✅ 使用训练好的权重: {model_path}")
    else:
        model_path = str(model_path)
        print(f"✅ 使用模型配置: {model_path}")
    
    try:
        # 将 src 加入 sys.path 以支持自定义模块
        src_dir = repo_root / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
        
        model = YOLO(model_path)
        model.model.eval()
        
        # 计算参数量
        num_params = get_num_params(model.model)
        num_params_m = num_params / 1e6  # 转换为百万
        
        # 计算 FLOPs
        flops = get_flops(model.model, imgsz=imgsz)
        
        return {
            "parameters": num_params,
            "parameters_M": num_params_m,
            "GFLOPs": flops,
        }
    except Exception as e:
        print(f"⚠️ 分析 {exp_name} 模型复杂度失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def measure_inference_speed(exp_name: str, imgsz: int = 1024, num_runs: int = 100, device: str = "cuda") -> Optional[Dict]:
    """测量推理速度"""
    repo_root = Path(__file__).resolve().parent
    
    # 确定模型路径
    exp_dir = repo_root / "experiments" / exp_name
    weights_file = exp_dir / "runs" / "train" / "weights" / "best.pt"
    
    if not weights_file.exists():
        print(f"⚠️ {exp_name} 的训练权重不存在，跳过推理速度测试")
        return None
    
    try:
        # 将 src 加入 sys.path 以支持自定义模块
        src_dir = repo_root / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
        
        model = YOLO(str(weights_file))
        model.model.eval()
        
        # 创建测试输入
        if device == "cuda" and torch.cuda.is_available():
            device_obj = torch.device("cuda")
        else:
            device_obj = torch.device("cpu")
            device = "cpu"
        
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device_obj)
        model.model.to(device_obj)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.model(dummy_input)
        
        # 测量推理时间
        if device == "cuda":
            torch.cuda.synchronize()
        
        import time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                if device == "cuda":
                    torch.cuda.synchronize()
                _ = model.model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # 转换为毫秒
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        min_time = min(times)
        max_time = max(times)
        
        return {
            "inference_time_ms": avg_time,
            "inference_time_std": std_time,
            "inference_time_min": min_time,
            "inference_time_max": max_time,
            "fps": 1000 / avg_time,
            "device": device,
        }
    except Exception as e:
        print(f"⚠️ 测量 {exp_name} 推理速度失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results():
    """对比所有结果"""
    print("="*70)
    print("📊 Baseline vs MCAttention 模型对比分析")
    print("="*70)
    
    results = {}
    
    for exp_name in ["baseline", "mca_attention"]:
        print(f"\n{'='*70}")
        print(f"🔍 分析 {exp_name.upper()} 模型")
        print(f"{'='*70}")
        
        exp_results = {}
        
        # 1. 加载训练结果
        print("\n1️⃣ 加载训练结果...")
        training_results = load_training_results(exp_name)
        if training_results:
            exp_results.update(training_results)
            print(f"   ✅ mAP50: {training_results['mAP50']:.4f}" if training_results.get('mAP50') else "   ⚠️ mAP50: N/A")
            print(f"   ✅ mAP50-95: {training_results['mAP50-95']:.4f}" if training_results.get('mAP50-95') else "   ⚠️ mAP50-95: N/A")
        else:
            print("   ⚠️ 训练结果未找到")
        
        # 2. 分析模型复杂度
        print("\n2️⃣ 分析模型复杂度...")
        complexity = analyze_model_complexity(exp_name)
        if complexity:
            exp_results.update(complexity)
            print(f"   ✅ 参数量: {complexity['parameters_M']:.2f}M ({complexity['parameters']:,})")
            print(f"   ✅ GFLOPs: {complexity['GFLOPs']:.2f}")
        else:
            print("   ⚠️ 模型复杂度分析失败")
        
        # 3. 测量推理速度
        print("\n3️⃣ 测量推理速度...")
        speed = measure_inference_speed(exp_name)
        if speed:
            exp_results.update(speed)
            print(f"   ✅ 平均推理时间: {speed['inference_time_ms']:.2f} ± {speed['inference_time_std']:.2f} ms")
            print(f"   ✅ FPS: {speed['fps']:.2f}")
        else:
            print("   ⚠️ 推理速度测试失败")
        
        results[exp_name] = exp_results
    
    # 打印对比表格
    print("\n" + "="*70)
    print("📋 详细对比表")
    print("="*70)
    
    comparison_data = []
    for exp_name, exp_results in results.items():
        comparison_data.append({
            "模型": "MCAttention" if exp_name == "mca_attention" else exp_name.upper(),
            "mAP50": f"{exp_results.get('mAP50', 0):.4f}" if exp_results.get('mAP50') else "N/A",
            "mAP50-95": f"{exp_results.get('mAP50-95', 0):.4f}" if exp_results.get('mAP50-95') else "N/A",
            "参数量(M)": f"{exp_results.get('parameters_M', 0):.2f}" if exp_results.get('parameters_M') else "N/A",
            "GFLOPs": f"{exp_results.get('GFLOPs', 0):.2f}" if exp_results.get('GFLOPs') else "N/A",
            "推理时间(ms)": f"{exp_results.get('inference_time_ms', 0):.2f}" if exp_results.get('inference_time_ms') else "N/A",
            "FPS": f"{exp_results.get('fps', 0):.2f}" if exp_results.get('fps') else "N/A",
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 计算改进幅度
    if "baseline" in results and "mca_attention" in results:
        baseline = results["baseline"]
        enhanced = results["mca_attention"]
        
        print("\n" + "="*70)
        print("📈 改进幅度分析")
        print("="*70)
        
        if baseline.get('mAP50') and enhanced.get('mAP50'):
            map50_improve = (enhanced['mAP50'] - baseline['mAP50']) / baseline['mAP50'] * 100
            print(f"mAP50 提升: {map50_improve:+.2f}% ({enhanced['mAP50']:.4f} vs {baseline['mAP50']:.4f})")
        
        if baseline.get('mAP50-95') and enhanced.get('mAP50-95'):
            map50_95_improve = (enhanced['mAP50-95'] - baseline['mAP50-95']) / baseline['mAP50-95'] * 100
            print(f"mAP50-95 提升: {map50_95_improve:+.2f}% ({enhanced['mAP50-95']:.4f} vs {baseline['mAP50-95']:.4f})")
        
        if baseline.get('parameters_M') and enhanced.get('parameters_M'):
            param_increase = (enhanced['parameters_M'] - baseline['parameters_M']) / baseline['parameters_M'] * 100
            print(f"参数量增加: {param_increase:+.2f}% ({enhanced['parameters_M']:.2f}M vs {baseline['parameters_M']:.2f}M)")
        
        if baseline.get('GFLOPs') and enhanced.get('GFLOPs'):
            flops_increase = (enhanced['GFLOPs'] - baseline['GFLOPs']) / baseline['GFLOPs'] * 100
            print(f"GFLOPs 增加: {flops_increase:+.2f}% ({enhanced['GFLOPs']:.2f} vs {baseline['GFLOPs']:.2f})")
        
        if baseline.get('inference_time_ms') and enhanced.get('inference_time_ms'):
            speed_change = (enhanced['inference_time_ms'] - baseline['inference_time_ms']) / baseline['inference_time_ms'] * 100
            print(f"推理时间变化: {speed_change:+.2f}% ({enhanced['inference_time_ms']:.2f}ms vs {baseline['inference_time_ms']:.2f}ms)")
    
    # 保存结果
    output_file = Path(__file__).parent / "experiments" / "comparison_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 对比结果已保存到: {output_file}")
    
    # 绘制对比图
    try:
        plot_comparison(results)
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def plot_comparison(results: Dict):
    """绘制对比图"""
    baseline = results.get("baseline", {})
    enhanced = results.get("mca_attention", {})
    
    if not baseline or not enhanced:
        print("⚠️ 数据不足，无法绘图")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Baseline vs MCAttention 模型对比', fontsize=16, fontweight='bold')
    
    # 1. mAP50 对比
    if baseline.get('mAP50') and enhanced.get('mAP50'):
        ax = axes[0, 0]
        ax.bar(['Baseline', 'MCAttention'], [baseline['mAP50'], enhanced['mAP50']], 
               color=['skyblue', 'lightcoral'])
        ax.set_title('mAP50 对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP50', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([baseline['mAP50'], enhanced['mAP50']]):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # 2. mAP50-95 对比
    if baseline.get('mAP50-95') and enhanced.get('mAP50-95'):
        ax = axes[0, 1]
        ax.bar(['Baseline', 'MCAttention'], [baseline['mAP50-95'], enhanced['mAP50-95']], 
               color=['skyblue', 'lightcoral'])
        ax.set_title('mAP50-95 对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP50-95', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([baseline['mAP50-95'], enhanced['mAP50-95']]):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # 3. 参数量对比
    if baseline.get('parameters_M') and enhanced.get('parameters_M'):
        ax = axes[0, 2]
        ax.bar(['Baseline', 'MCAttention'], [baseline['parameters_M'], enhanced['parameters_M']], 
               color=['skyblue', 'lightcoral'])
        ax.set_title('参数量对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('参数量 (M)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([baseline['parameters_M'], enhanced['parameters_M']]):
            ax.text(i, v, f'{v:.2f}M', ha='center', va='bottom')
    
    # 4. GFLOPs 对比
    if baseline.get('GFLOPs') and enhanced.get('GFLOPs'):
        ax = axes[1, 0]
        ax.bar(['Baseline', 'MCAttention'], [baseline['GFLOPs'], enhanced['GFLOPs']], 
               color=['skyblue', 'lightcoral'])
        ax.set_title('GFLOPs 对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('GFLOPs', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([baseline['GFLOPs'], enhanced['GFLOPs']]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # 5. 推理时间对比
    if baseline.get('inference_time_ms') and enhanced.get('inference_time_ms'):
        ax = axes[1, 1]
        ax.bar(['Baseline', 'MCAttention'], [baseline['inference_time_ms'], enhanced['inference_time_ms']], 
               color=['skyblue', 'lightcoral'])
        ax.set_title('推理时间对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('推理时间 (ms)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([baseline['inference_time_ms'], enhanced['inference_time_ms']]):
            ax.text(i, v, f'{v:.2f}ms', ha='center', va='bottom')
    
    # 6. FPS 对比
    if baseline.get('fps') and enhanced.get('fps'):
        ax = axes[1, 2]
        ax.bar(['Baseline', 'MCAttention'], [baseline['fps'], enhanced['fps']], 
               color=['skyblue', 'lightcoral'])
        ax.set_title('FPS 对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([baseline['fps'], enhanced['fps']]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = Path(__file__).parent / "experiments" / "comparison_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存到: {output_file}")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    compare_results()

