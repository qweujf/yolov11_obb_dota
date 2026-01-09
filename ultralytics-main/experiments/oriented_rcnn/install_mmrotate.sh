#!/bin/bash
# Oriented R-CNN 环境安装脚本 (Linux/Mac)
# 用于在远程 Linux 电脑上快速配置环境

echo "========================================"
echo "Oriented R-CNN 环境安装脚本"
echo "========================================"
echo ""

# 检查 Python
echo "1. 检查 Python 环境..."
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装或不在 PATH 中"
    exit 1
fi
python --version
python -c "import sys; print(f'   Python 路径: {sys.executable}')"
echo ""

# 检查 PyTorch
echo "2. 检查 PyTorch..."
if ! python -c "import torch" &> /dev/null; then
    echo "❌ PyTorch 未安装"
    echo "   请先安装 PyTorch"
    exit 1
fi
python -c "import torch; print(f'✅ PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
echo ""

# 升级构建工具
echo "3. 升级构建工具..."
pip install --upgrade pip setuptools wheel
echo ""

# 安装依赖（Linux 上可以直接安装完整版）
echo "4. 安装 mmcv..."
pip install mmcv
if [ $? -ne 0 ]; then
    echo "⚠️  mmcv 安装失败，尝试安装 mmcv-lite..."
    pip install mmcv-lite
    if [ $? -ne 0 ]; then
        echo "❌ mmcv 安装失败"
        exit 1
    fi
fi
echo "✅ mmcv 安装成功"
echo ""

# 安装 mmengine
echo "5. 安装 mmengine..."
pip install mmengine
if [ $? -ne 0 ]; then
    echo "❌ mmengine 安装失败"
    exit 1
fi
echo "✅ mmengine 安装成功"
echo ""

# 安装 mmdet
echo "6. 安装 mmdet..."
pip install mmdet
if [ $? -ne 0 ]; then
    echo "❌ mmdet 安装失败"
    exit 1
fi
echo "✅ mmdet 安装成功"
echo ""

# 安装 mmrotate
echo "7. 安装 mmrotate..."
pip install mmrotate
if [ $? -ne 0 ]; then
    echo "❌ mmrotate 安装失败"
    exit 1
fi
echo "✅ mmrotate 安装成功"
echo ""

# 验证安装
echo "8. 验证安装..."
echo ""

all_ok=true

# 检查 mmcv
if python -c "import mmcv" &> /dev/null; then
    version=$(python -c "import mmcv; print(getattr(mmcv, '__version__', 'unknown'))" 2>/dev/null)
    echo "✅ mmcv: $version"
else
    echo "❌ mmcv: 导入失败"
    all_ok=false
fi

# 检查 mmengine
if python -c "import mmengine" &> /dev/null; then
    version=$(python -c "import mmengine; print(getattr(mmengine, '__version__', 'unknown'))" 2>/dev/null)
    echo "✅ mmengine: $version"
else
    echo "❌ mmengine: 导入失败"
    all_ok=false
fi

# 检查 mmdet
if python -c "import mmdet" &> /dev/null; then
    version=$(python -c "import mmdet; print(getattr(mmdet, '__version__', 'unknown'))" 2>/dev/null)
    echo "✅ mmdet: $version"
else
    echo "❌ mmdet: 导入失败"
    all_ok=false
fi

# 检查 mmrotate
if python -c "from mmrotate.apis import init_detector" &> /dev/null; then
    version=$(python -c "import mmrotate; print(getattr(mmrotate, '__version__', 'unknown'))" 2>/dev/null)
    echo "✅ mmrotate: $version"
else
    echo "❌ mmrotate: 导入失败"
    all_ok=false
fi

echo ""

if [ "$all_ok" = true ]; then
    echo "========================================"
    echo "✅ 所有依赖安装成功！"
    echo "========================================"
    echo ""
    echo "现在可以运行训练脚本："
    echo "  python train.py"
else
    echo "========================================"
    echo "❌ 部分依赖安装失败"
    echo "========================================"
    echo ""
    echo "请查看错误信息并参考 环境配置指南.md"
    exit 1
fi




