"""测试 mmrotate 是否可以正常导入"""
import sys

print("="*60)
print("测试 mmrotate 导入")
print("="*60)
print(f"Python 路径: {sys.executable}")
print()

# 测试 mmcv
print("1. 测试 mmcv...")
try:
    import mmcv
    print(f"   ✅ mmcv 导入成功，版本: {mmcv.__version__}")
except Exception as e:
    print(f"   ❌ mmcv 导入失败: {e}")
    sys.exit(1)

# 测试 mmengine
print("\n2. 测试 mmengine...")
try:
    import mmengine
    print(f"   ✅ mmengine 导入成功，版本: {mmengine.__version__}")
except Exception as e:
    print(f"   ❌ mmengine 导入失败: {e}")
    sys.exit(1)

# 测试 mmdet
print("\n3. 测试 mmdet...")
try:
    import mmdet
    print(f"   ✅ mmdet 导入成功，版本: {mmdet.__version__}")
except Exception as e:
    print(f"   ❌ mmdet 导入失败: {e}")
    sys.exit(1)

# 测试 mmrotate（关键测试）
print("\n4. 测试 mmrotate...")
try:
    from mmrotate.apis import init_detector
    print("   ✅ mmrotate 导入成功！")
    print("\n" + "="*60)
    print("🎉 所有包都已成功安装并可以正常导入！")
    print("="*60)
except Exception as e:
    print(f"   ❌ mmrotate 导入失败: {e}")
    print("\n" + "="*60)
    print("⚠️  mmrotate 导入失败")
    print("="*60)
    print("\n可能的原因：")
    print("1. DLL 加载失败（最常见）")
    print("   - 解决方案：安装 Visual C++ Redistributable")
    print("   - 下载：https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   - 安装后重启终端/IDE")
    print("\n2. 版本不兼容")
    print("   - mmcv-full 1.7.2 是为 PyTorch 2.0.0 编译的")
    print("   - 你的 PyTorch 是 2.4.1，可能需要降级")
    print("\n3. 环境变量问题")
    print("   - 检查 PATH 中是否包含必要的 DLL 路径")
    sys.exit(1)






