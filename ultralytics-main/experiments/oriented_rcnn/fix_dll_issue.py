"""
修复 mmrotate DLL 加载失败的脚本
尝试多种解决方案
"""
import subprocess
import sys
import os

def run_cmd(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)
    return result.returncode == 0

def check_versions():
    """检查当前安装的版本"""
    print("\n检查当前安装的版本...")
    packages = ['mmcv-lite', 'mmcv-full', 'mmengine', 'mmdet', 'mmrotate']
    for pkg in packages:
        result = subprocess.run(f"pip show {pkg}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Version:'):
                    print(f"  {pkg}: {line.split(':')[1].strip()}")
                    break

def check_pytorch():
    """检查 PyTorch 和 CUDA 版本"""
    print("\n检查 PyTorch 和 CUDA 版本...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  CUDA 设备数量: {torch.cuda.device_count()}")
        else:
            print("  CUDA: 不可用")
    except ImportError:
        print("  PyTorch 未安装")

def solution1_reinstall_compatible():
    """方案1：重新安装兼容版本"""
    print("\n" + "="*60)
    print("方案1：重新安装兼容版本组合")
    print("="*60)
    
    print("\n步骤1：卸载现有包")
    run_cmd("pip uninstall mmcv-lite mmcv-full mmrotate -y", "卸载 mmcv 和 mmrotate")
    
    print("\n步骤2：安装 mmcv-lite 2.0.1")
    run_cmd("pip install mmcv-lite==2.0.1", "安装 mmcv-lite")
    
    print("\n步骤3：重新安装 mmrotate")
    run_cmd("pip install mmrotate==0.3.4", "安装 mmrotate")
    
    print("\n步骤4：测试导入")
    test_import()

def solution2_use_mmcv_full():
    """方案2：使用 mmcv-full"""
    print("\n" + "="*60)
    print("方案2：使用 mmcv-full（需要 CUDA）")
    print("="*60)
    
    # 检查 CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  警告：CUDA 不可用，mmcv-full 可能无法正常工作")
            return False
        cuda_version = torch.version.cuda
        torch_version = torch.__version__
        print(f"检测到: PyTorch {torch_version}, CUDA {cuda_version}")
        
        # 确定 mmcv-full 版本
        if "2.0" in torch_version:
            if "11.8" in cuda_version or "118" in cuda_version:
                mmcv_url = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"
            elif "11.7" in cuda_version or "117" in cuda_version:
                mmcv_url = "https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html"
            else:
                print("⚠️  无法确定 CUDA 版本，使用默认 URL")
                mmcv_url = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"
        else:
            print("⚠️  无法确定 PyTorch 版本，使用默认 URL")
            mmcv_url = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"
        
        print(f"\n使用 URL: {mmcv_url}")
        
        print("\n步骤1：卸载 mmcv-lite")
        run_cmd("pip uninstall mmcv-lite -y", "卸载 mmcv-lite")
        
        print("\n步骤2：安装 mmcv-full")
        run_cmd(f"pip install mmcv-full -f {mmcv_url}", "安装 mmcv-full")
        
        print("\n步骤3：测试导入")
        return test_import()
    except ImportError:
        print("❌ PyTorch 未安装，无法使用此方案")
        return False

def solution3_downgrade():
    """方案3：降级到稳定版本"""
    print("\n" + "="*60)
    print("方案3：降级到稳定版本组合")
    print("="*60)
    
    print("\n步骤1：卸载所有相关包")
    run_cmd("pip uninstall mmcv-lite mmcv-full mmrotate mmdet mmengine -y", "卸载所有包")
    
    print("\n步骤2：安装稳定版本")
    run_cmd("pip install mmcv-lite==2.0.0", "安装 mmcv-lite 2.0.0")
    run_cmd("pip install mmengine==0.10.0", "安装 mmengine 0.10.0")
    run_cmd("pip install mmdet==2.28.0", "安装 mmdet 2.28.0")
    run_cmd("pip install mmrotate==0.3.3", "安装 mmrotate 0.3.3")
    
    print("\n步骤3：测试导入")
    return test_import()

def test_import():
    """测试导入"""
    print("\n测试导入 mmrotate...")
    try:
        from mmrotate.apis import init_detector
        print("✅ mmrotate 导入成功！")
        return True
    except Exception as e:
        print(f"❌ mmrotate 导入失败: {e}")
        return False

def main():
    print("="*60)
    print("mmrotate DLL 加载失败修复工具")
    print("="*60)
    
    # 检查当前状态
    check_versions()
    check_pytorch()
    
    print("\n" + "="*60)
    print("请选择修复方案：")
    print("="*60)
    print("1. 重新安装兼容版本（mmcv-lite 2.0.1 + mmrotate 0.3.4）")
    print("2. 使用 mmcv-full（需要 CUDA）")
    print("3. 降级到稳定版本组合")
    print("4. 仅测试导入（不修改）")
    print("0. 退出")
    
    choice = input("\n请选择 (0-4): ").strip()
    
    if choice == "1":
        solution1_reinstall_compatible()
    elif choice == "2":
        solution2_use_mmcv_full()
    elif choice == "3":
        solution3_downgrade()
    elif choice == "4":
        test_import()
    elif choice == "0":
        print("退出")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()

