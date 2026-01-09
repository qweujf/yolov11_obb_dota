#!/usr/bin/env python3
"""
验证 PyTorch 安装和 CUDA 可用性
"""

import sys

print("=" * 60)
print("PyTorch 安装验证")
print("=" * 60)

# 检查 PyTorch 版本
try:
    import torch
    print(f"✓ PyTorch 版本: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch 导入失败: {e}")
    sys.exit(1)

# 检查 torchvision
try:
    import torchvision
    print(f"✓ torchvision 版本: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ torchvision 导入失败: {e}")

# 检查 torchaudio
try:
    import torchaudio
    print(f"✓ torchaudio 版本: {torchaudio.__version__}")
except ImportError as e:
    print(f"✗ torchaudio 导入失败: {e}")

# 检查 CUDA 可用性
print("\n" + "=" * 60)
print("CUDA 信息")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✓ CUDA 可用")
    print(f"  - CUDA 版本: {torch.version.cuda}")
    print(f"  - cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"  - 可用 GPU 数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    - 总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    - 计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # 测试 GPU 计算
    print("\n测试 GPU 计算...")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU 计算测试成功")
    except Exception as e:
        print(f"✗ GPU 计算测试失败: {e}")
else:
    print("✗ CUDA 不可用")
    print("  注意: 您安装的是 CUDA 11.8 版本的 PyTorch，但系统可能没有可用的 GPU 或 CUDA 驱动")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)




