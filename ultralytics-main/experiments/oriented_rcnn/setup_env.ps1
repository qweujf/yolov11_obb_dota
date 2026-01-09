# Oriented R-CNN 环境配置脚本 (PowerShell)
# 用于创建新的 conda 环境并安装所有依赖

param(
    [string]$EnvName = "oriented_rcnn_mmrotate",
    [string]$PythonVersion = "3.10",
    [switch]$UseExistingEnv = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Oriented R-CNN 环境配置脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 conda 是否安装
Write-Host "1. 检查 Conda..." -ForegroundColor Yellow
$condaCheck = conda --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Conda 未安装或不在 PATH 中" -ForegroundColor Red
    Write-Host "   请先安装 Anaconda 或 Miniconda" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ $condaCheck" -ForegroundColor Green
Write-Host ""

# 创建或使用现有环境
if (-not $UseExistingEnv) {
    Write-Host "2. 创建 Conda 环境: $EnvName (Python $PythonVersion)..." -ForegroundColor Yellow
    
    # 检查环境是否已存在
    $envExists = conda env list | Select-String -Pattern "^$EnvName\s"
    if ($envExists) {
        Write-Host "⚠️  环境 $EnvName 已存在" -ForegroundColor Yellow
        $response = Read-Host "是否删除并重新创建? (y/n)"
        if ($response -eq "y" -or $response -eq "Y") {
            Write-Host "   删除现有环境..." -ForegroundColor Yellow
            conda env remove -n $EnvName -y
        } else {
            Write-Host "   使用现有环境..." -ForegroundColor Yellow
        }
    }
    
    if (-not (conda env list | Select-String -Pattern "^$EnvName\s")) {
        conda create -n $EnvName python=$PythonVersion -y
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 环境创建失败" -ForegroundColor Red
            exit 1
        }
        Write-Host "✅ 环境创建成功" -ForegroundColor Green
    }
} else {
    Write-Host "2. 使用现有环境: $EnvName..." -ForegroundColor Yellow
}

# 激活环境
Write-Host ""
Write-Host "3. 激活环境..." -ForegroundColor Yellow
conda activate $EnvName
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 环境激活失败" -ForegroundColor Red
    Write-Host "   请手动运行: conda activate $EnvName" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ 环境已激活" -ForegroundColor Green

# 显示当前 Python 路径
$pythonPath = python -c "import sys; print(sys.executable)" 2>&1
Write-Host "   Python 路径: $pythonPath" -ForegroundColor Gray
Write-Host ""

# 检查 Python 版本
Write-Host "4. 检查 Python 版本..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "✅ $pythonVersion" -ForegroundColor Green
Write-Host ""

# 检查 PyTorch
Write-Host "5. 检查 PyTorch..." -ForegroundColor Yellow
$torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  PyTorch 未安装，正在安装..." -ForegroundColor Yellow
    pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ PyTorch 安装失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ PyTorch 安装成功" -ForegroundColor Green
} else {
    Write-Host "✅ $torchCheck" -ForegroundColor Green
}
Write-Host ""

# 升级构建工具
Write-Host "6. 升级构建工具..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  构建工具升级失败，继续安装..." -ForegroundColor Yellow
}
Write-Host ""

# 安装 mmcv-lite
Write-Host "7. 安装 mmcv-lite..." -ForegroundColor Yellow
pip install mmcv-lite
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmcv-lite 安装失败" -ForegroundColor Red
    Write-Host "   尝试安装预编译的 mmcv-full..." -ForegroundColor Yellow
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ mmcv 安装失败，请手动安装" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ mmcv 安装成功" -ForegroundColor Green
Write-Host ""

# 安装 mmengine
Write-Host "8. 安装 mmengine..." -ForegroundColor Yellow
pip install mmengine
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmengine 安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ mmengine 安装成功" -ForegroundColor Green
Write-Host ""

# 安装 mmdet
Write-Host "9. 安装 mmdet..." -ForegroundColor Yellow
pip install mmdet
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmdet 安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ mmdet 安装成功" -ForegroundColor Green
Write-Host ""

# 安装 mmrotate
Write-Host "10. 安装 mmrotate..." -ForegroundColor Yellow
pip install mmrotate
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmrotate 安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ mmrotate 安装成功" -ForegroundColor Green
Write-Host ""

# 安装其他依赖
Write-Host "11. 安装其他依赖..." -ForegroundColor Yellow
pip install numpy>=1.23.0 opencv-python>=4.6.0 pyyaml>=5.3.1 tqdm>=4.64.0 matplotlib>=3.3.0 pillow>=7.1.2 scipy>=1.4.1 pandas>=1.1.4
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  部分依赖安装失败，继续验证..." -ForegroundColor Yellow
}
Write-Host ""

# 验证安装
Write-Host "12. 验证安装..." -ForegroundColor Yellow
Write-Host ""

$allOk = $true

# 检查 mmcv
$mmcvCheck = python -c "import mmcv; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    $version = python -c "import mmcv; print(getattr(mmcv, '__version__', 'unknown'))" 2>&1
    Write-Host "✅ mmcv: $version" -ForegroundColor Green
} else {
    Write-Host "❌ mmcv: 导入失败" -ForegroundColor Red
    $allOk = $false
}

# 检查 mmengine
$mmengineCheck = python -c "import mmengine; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    $version = python -c "import mmengine; print(getattr(mmengine, '__version__', 'unknown'))" 2>&1
    Write-Host "✅ mmengine: $version" -ForegroundColor Green
} else {
    Write-Host "❌ mmengine: 导入失败" -ForegroundColor Red
    $allOk = $false
}

# 检查 mmdet
$mmdetCheck = python -c "import mmdet; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    $version = python -c "import mmdet; print(getattr(mmdet, '__version__', 'unknown'))" 2>&1
    Write-Host "✅ mmdet: $version" -ForegroundColor Green
} else {
    Write-Host "❌ mmdet: 导入失败" -ForegroundColor Red
    $allOk = $false
}

# 检查 mmrotate
$mmrotateCheck = python -c "from mmrotate.apis import init_detector; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    $version = python -c "import mmrotate; print(getattr(mmrotate, '__version__', 'unknown'))" 2>&1
    Write-Host "✅ mmrotate: $version" -ForegroundColor Green
} else {
    Write-Host "❌ mmrotate: 导入失败" -ForegroundColor Red
    $allOk = $false
}

Write-Host ""

if ($allOk) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ 所有依赖安装成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "环境名称: $EnvName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使用方法：" -ForegroundColor Cyan
    Write-Host "  1. 激活环境: conda activate $EnvName" -ForegroundColor White
    Write-Host "  2. 进入目录: cd ultralytics-main/experiments/oriented_rcnn" -ForegroundColor White
    Write-Host "  3. 运行训练: python train.py" -ForegroundColor White
    Write-Host ""
    Write-Host "验证安装：" -ForegroundColor Cyan
    Write-Host "  python check_install.py" -ForegroundColor White
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ 部分依赖安装失败" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "请查看错误信息并参考 环境配置完整指南.md" -ForegroundColor Yellow
    exit 1
}





