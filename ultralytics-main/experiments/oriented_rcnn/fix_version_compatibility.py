"""
修复 mmrotate 版本兼容性问题
解决 mmcv 和 mmdet 版本不兼容的问题
"""
import subprocess
import sys

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
    packages = ['mmcv', 'mmcv-lite', 'mmcv-full', 'mmengine', 'mmdet', 'mmrotate']
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
            # 确定 CUDA 版本号用于 URL
            cuda_version = torch.version.cuda
            if "11.8" in cuda_version or "118" in cuda_version:
                return "cu118"
            elif "11.7" in cuda_version or "117" in cuda_version:
                return "cu117"
            elif "11.6" in cuda_version or "116" in cuda_version:
                return "cu116"
            else:
                return "cu118"  # 默认
        else:
            print("  CUDA: 不可用")
            return None
    except ImportError:
        print("  PyTorch 未安装")
        return None

def solution1_install_compatible_mmcv_full():
    """方案1：安装兼容的 mmcv-full 1.8.0"""
    print("\n" + "="*60)
    print("方案1：安装兼容的 mmcv-full 1.8.0")
    print("="*60)
    
    cuda_version = check_pytorch()
    if not cuda_version:
        print("⚠️  警告：CUDA 不可用，可能无法使用 mmcv-full")
        return False
    
    # 确定 PyTorch 版本
    try:
        import torch
        torch_version = torch.__version__
        # 检查实际版本
        if "2.4" in torch_version:
            # PyTorch 2.4，尝试使用 2.0 的 mmcv（向后兼容）
            torch_ver = "2.0.0"
            print("⚠️  注意：PyTorch 2.4 可能不兼容，尝试使用 PyTorch 2.0 的 mmcv")
        elif "2.0" in torch_version:
            torch_ver = "2.0.0"
        elif "1.13" in torch_version:
            torch_ver = "1.13.0"
        else:
            torch_ver = "2.0.0"  # 默认
    except:
        torch_ver = "2.0.0"
    
    print(f"\n使用 CUDA: {cuda_version}, PyTorch: {torch_ver}")
    
    print("\n步骤1：卸载不兼容的版本")
    run_cmd("pip uninstall mmcv-lite mmcv-full mmdet mmrotate -y", "卸载现有包")
    
    print("\n步骤2：尝试安装 mmcv-full 1.8.0")
    url = f"https://download.openmmlab.com/mmcv/dist/{cuda_version}/torch{torch_ver}/index.html"
    success = run_cmd(f"pip install mmcv-full==1.8.0 -f {url}", "安装 mmcv-full 1.8.0")
    
    if not success:
        print("\n⚠️  mmcv-full 1.8.0 不可用，尝试 1.7.2")
        success = run_cmd(f"pip install mmcv-full==1.7.2 -f {url}", "安装 mmcv-full 1.7.2")
    
    if success:
        print("\n步骤3：安装 mmdet 和 mmrotate")
        run_cmd("pip install mmdet==2.28.2", "安装 mmdet")
        run_cmd("pip install mmrotate==0.3.4", "安装 mmrotate")
        
        print("\n步骤4：测试导入")
        result = test_import()
        if not result:
            print("\n⚠️  DLL 加载失败，可能的原因：")
            print("   1. mmcv-full 版本与 PyTorch/CUDA 版本不完全匹配")
            print("   2. 缺少 Visual C++ Redistributable")
            print("   3. 环境变量 PATH 中缺少必要的 DLL 路径")
            print("\n建议：")
            print("   1. 安装 Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("   2. 安装后重启终端/IDE")
            print("   3. 如果还是不行，可能需要降级 PyTorch 到 2.0.0")
        return result
    else:
        print("\n❌ 无法安装 mmcv-full，尝试方案2")
        return False

def solution2_upgrade_mmdet():
    """方案2：升级 mmdet 到支持 mmcv 2.0 的版本"""
    print("\n" + "="*60)
    print("方案2：升级 mmdet 到支持 mmcv 2.0 的版本")
    print("="*60)
    
    print("\n步骤1：升级 mmdet")
    run_cmd("pip install mmdet --upgrade", "升级 mmdet")
    
    print("\n步骤2：重新安装 mmrotate")
    run_cmd("pip install mmrotate==0.3.4 --no-deps", "安装 mmrotate（跳过依赖）")
    
    print("\n步骤3：测试导入")
    return test_import()

def solution3_use_mmcv_lite_1_8():
    """方案3：使用 mmcv-lite 1.8.0（如果存在）"""
    print("\n" + "="*60)
    print("方案3：使用 mmcv-lite 1.8.0")
    print("="*60)
    
    print("\n步骤1：卸载现有版本")
    run_cmd("pip uninstall mmcv-lite mmcv-full mmdet mmrotate -y", "卸载现有包")
    
    print("\n步骤2：尝试安装 mmcv-lite 1.8.0")
    # 注意：mmcv-lite 1.8.0 可能不存在，因为 mmcv-lite 是从 2.0 开始的
    # 但我们可以尝试
    success = run_cmd("pip install mmcv-lite==1.8.0", "安装 mmcv-lite 1.8.0")
    
    if not success:
        print("\n⚠️  mmcv-lite 1.8.0 不存在，此方案不可行")
        return False
    
    print("\n步骤3：安装 mmdet 和 mmrotate")
    run_cmd("pip install mmdet==2.28.2", "安装 mmdet")
    run_cmd("pip install mmrotate==0.3.4", "安装 mmrotate")
    
    print("\n步骤4：测试导入")
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
    print("mmrotate 版本兼容性修复工具")
    print("="*60)
    
    # 检查当前状态
    check_versions()
    check_pytorch()
    
    print("\n" + "="*60)
    print("检测到版本不兼容问题：")
    print("  - mmdet 2.28.2 需要 mmcv <= 1.8.0")
    print("  - 但当前安装的是 mmcv 2.0.1")
    print("="*60)
    
    print("\n请选择修复方案：")
    print("1. 安装兼容的 mmcv-full 1.8.0（推荐，需要 CUDA）")
    print("2. 升级 mmdet 到支持 mmcv 2.0 的版本（可能不可用）")
    print("3. 仅测试导入（不修改）")
    print("0. 退出")
    
    choice = input("\n请选择 (0-3): ").strip()
    
    if choice == "1":
        if solution1_install_compatible_mmcv_full():
            print("\n✅ 修复成功！")
        else:
            print("\n❌ 修复失败，请尝试其他方案")
    elif choice == "2":
        if solution2_upgrade_mmdet():
            print("\n✅ 修复成功！")
        else:
            print("\n❌ 修复失败，请尝试方案1")
    elif choice == "3":
        test_import()
    elif choice == "0":
        print("退出")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()

