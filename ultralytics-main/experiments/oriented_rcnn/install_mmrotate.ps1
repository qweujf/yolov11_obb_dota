# Oriented R-CNN 环境安装脚本 (PowerShell)
# 用于在远程 Windows 电脑上快速配置环境

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Oriented R-CNN 环境安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "1. 检查 Python 环境..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python 未安装或不在 PATH 中" -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pythonVersion" -ForegroundColor Green

# 显示当前 Python 路径
$pythonPath = python -c "import sys; print(sys.executable)" 2>&1
Write-Host "   Python 路径: $pythonPath" -ForegroundColor Gray
Write-Host ""

# 检查 PyTorch
Write-Host "2. 检查 PyTorch..." -ForegroundColor Yellow
$torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ PyTorch 未安装" -ForegroundColor Red
    Write-Host "   请先安装 PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ $torchCheck" -ForegroundColor Green
Write-Host ""

# 升级构建工具
Write-Host "3. 升级构建工具..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  构建工具升级失败，继续安装..." -ForegroundColor Yellow
}
Write-Host ""

# 安装 mmcv-lite
Write-Host "4. 安装 mmcv-lite..." -ForegroundColor Yellow
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
Write-Host "5. 安装 mmengine..." -ForegroundColor Yellow
pip install mmengine
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmengine 安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ mmengine 安装成功" -ForegroundColor Green
Write-Host ""

# 安装 mmdet
Write-Host "6. 安装 mmdet..." -ForegroundColor Yellow
pip install mmdet
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmdet 安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ mmdet 安装成功" -ForegroundColor Green
Write-Host ""

# 安装 mmrotate
Write-Host "7. 安装 mmrotate..." -ForegroundColor Yellow
pip install mmrotate
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ mmrotate 安装失败" -ForegroundColor Red
    exit 1
}
Write-Host "✅ mmrotate 安装成功" -ForegroundColor Green
Write-Host ""

# 验证安装
Write-Host "8. 验证安装..." -ForegroundColor Yellow
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
    Write-Host "现在可以运行训练脚本：" -ForegroundColor Cyan
    Write-Host "  python train.py" -ForegroundColor White
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ 部分依赖安装失败" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "请查看错误信息并参考 环境配置指南.md" -ForegroundColor Yellow
    exit 1
}




