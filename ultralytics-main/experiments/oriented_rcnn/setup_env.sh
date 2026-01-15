#!/bin/bash
# Oriented R-CNN 环境配置脚本 (Linux/Mac)
# 用于创建新的 conda 环境并安装所有依赖

ENV_NAME="${1:-oriented_rcnn_mmrotate}"
PYTHON_VERSION="${2:-3.10}"
USE_EXISTING="${3:-false}"

echo "========================================"
echo "Oriented R-CNN 环境配置脚本"
echo "========================================"
echo ""

# 检查 conda 是否安装
echo "1. 检查 Conda..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安装或不在 PATH 中"
    echo "   请先安装 Anaconda 或 Miniconda"
    exit 1
fi
echo "✅ $(conda --version)"
echo ""

# 创建或使用现有环境
if [ "$USE_EXISTING" != "true" ]; then
    echo "2. 创建 Conda 环境: $ENV_NAME (Python $PYTHON_VERSION)..."
    
    # 检查环境是否已存在
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "⚠️  环境 $ENV_NAME 已存在"
        read -p "是否删除并重新创建? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "   删除现有环境..."
            conda env remove -n $ENV_NAME -y
        else
            echo "   使用现有环境..."
        fi
    fi
    
    if ! conda env list | grep -q "^$ENV_NAME "; then
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        if [ $? -ne 0 ]; then
            echo "❌ 环境创建失败"
            exit 1
        fi
        echo "✅ 环境创建成功"
    fi
else
    echo "2. 使用现有环境: $ENV_NAME..."
fi

# 激活环境
echo ""
echo "3. 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    echo "   请手动运行: conda activate $ENV_NAME"
    exit 1
fi
echo "✅ 环境已激活"

# 显示当前 Python 路径
PYTHON_PATH=$(python -c "import sys; print(sys.executable)" 2>&1)
echo "   Python 路径: $PYTHON_PATH"
echo ""

# 检查 Python 版本
echo "4. 检查 Python 版本..."
PYTHON_VERSION_OUT=$(python --version 2>&1)
echo "✅ $PYTHON_VERSION_OUT"
echo ""

# 检查 PyTorch
echo "5. 检查 PyTorch..."
TORCH_CHECK=$(python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>&1)
if [ $? -ne 0 ]; then
    echo "⚠️  PyTorch 未安装，正在安装..."
    pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
    if [ $? -ne 0 ]; then
        echo "❌ PyTorch 安装失败"
        exit 1
    fi
    echo "✅ PyTorch 安装成功"
else
    echo "✅ $TORCH_CHECK"
fi
echo ""

# 升级构建工具
echo "6. 升级构建工具..."
pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "⚠️  构建工具升级失败，继续安装..."
fi
echo ""

# 安装 mmcv
echo "7. 安装 mmcv..."
pip install mmcv
if [ $? -ne 0 ]; then
    echo "❌ mmcv 安装失败"
    exit 1
fi
echo "✅ mmcv 安装成功"
echo ""

# 安装 mmengine
echo "8. 安装 mmengine..."
pip install mmengine
if [ $? -ne 0 ]; then
    echo "❌ mmengine 安装失败"
    exit 1
fi
echo "✅ mmengine 安装成功"
echo ""

# 安装 mmdet
echo "9. 安装 mmdet..."
pip install mmdet
if [ $? -ne 0 ]; then
    echo "❌ mmdet 安装失败"
    exit 1
fi
echo "✅ mmdet 安装成功"
echo ""

# 安装 mmrotate
echo "10. 安装 mmrotate..."
pip install mmrotate
if [ $? -ne 0 ]; then
    echo "❌ mmrotate 安装失败"
    exit 1
fi
echo "✅ mmrotate 安装成功"
echo ""

# 安装其他依赖
echo "11. 安装其他依赖..."
pip install numpy>=1.23.0 opencv-python>=4.6.0 pyyaml>=5.3.1 tqdm>=4.64.0 matplotlib>=3.3.0 pillow>=7.1.2 scipy>=1.4.1 pandas>=1.1.4
if [ $? -ne 0 ]; then
    echo "⚠️  部分依赖安装失败，继续验证..."
fi
echo ""

# 验证安装
echo "12. 验证安装..."
echo ""

ALL_OK=true

# 检查 mmcv
MMCV_CHECK=$(python -c "import mmcv; print('OK')" 2>&1)
if [ $? -eq 0 ]; then
    VERSION=$(python -c "import mmcv; print(getattr(mmcv, '__version__', 'unknown'))" 2>&1)
    echo "✅ mmcv: $VERSION"
else
    echo "❌ mmcv: 导入失败"
    ALL_OK=false
fi

# 检查 mmengine
MMENGINE_CHECK=$(python -c "import mmengine; print('OK')" 2>&1)
if [ $? -eq 0 ]; then
    VERSION=$(python -c "import mmengine; print(getattr(mmengine, '__version__', 'unknown'))" 2>&1)
    echo "✅ mmengine: $VERSION"
else
    echo "❌ mmengine: 导入失败"
    ALL_OK=false
fi

# 检查 mmdet
MMDET_CHECK=$(python -c "import mmdet; print('OK')" 2>&1)
if [ $? -eq 0 ]; then
    VERSION=$(python -c "import mmdet; print(getattr(mmdet, '__version__', 'unknown'))" 2>&1)
    echo "✅ mmdet: $VERSION"
else
    echo "❌ mmdet: 导入失败"
    ALL_OK=false
fi

# 检查 mmrotate
MMROTATE_CHECK=$(python -c "from mmrotate.apis import init_detector; print('OK')" 2>&1)
if [ $? -eq 0 ]; then
    VERSION=$(python -c "import mmrotate; print(getattr(mmrotate, '__version__', 'unknown'))" 2>&1)
    echo "✅ mmrotate: $VERSION"
else
    echo "❌ mmrotate: 导入失败"
    ALL_OK=false
fi

echo ""

if [ "$ALL_OK" = true ]; then
    echo "========================================"
    echo "✅ 所有依赖安装成功！"
    echo "========================================"
    echo ""
    echo "环境名称: $ENV_NAME"
    echo ""
    echo "使用方法："
    echo "  1. 激活环境: conda activate $ENV_NAME"
    echo "  2. 进入目录: cd ultralytics-main/experiments/oriented_rcnn"
    echo "  3. 运行训练: python train.py"
    echo ""
    echo "验证安装："
    echo "  python check_install.py"
else
    echo "========================================"
    echo "❌ 部分依赖安装失败"
    echo "========================================"
    echo ""
    echo "请查看错误信息并参考 环境配置完整指南.md"
    exit 1
fi








