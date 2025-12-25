"""快速检查 mmrotate 相关包的安装情况"""
import sys

print("=" * 60)
print("检查 mmrotate 相关包的安装情况")
print("=" * 60)
print(f"Python 路径: {sys.executable}")
print(f"Python 版本: {sys.version}")
print()

packages_status = {}

# 检查 mmcv
try:
    import mmcv
    version = getattr(mmcv, '__version__', 'unknown')
    packages_status['mmcv'] = (True, version)
except ImportError as e:
    packages_status['mmcv'] = (False, str(e))

# 检查 mmengine
try:
    import mmengine
    version = getattr(mmengine, '__version__', 'unknown')
    packages_status['mmengine'] = (True, version)
except ImportError as e:
    packages_status['mmengine'] = (False, str(e))

# 检查 mmdet
try:
    import mmdet
    version = getattr(mmdet, '__version__', 'unknown')
    packages_status['mmdet'] = (True, version)
except ImportError as e:
    packages_status['mmdet'] = (False, str(e))

# 检查 mmrotate
try:
    import mmrotate
    version = getattr(mmrotate, '__version__', 'unknown')
    packages_status['mmrotate'] = (True, version)
except ImportError as e:
    packages_status['mmrotate'] = (False, str(e))

# 显示结果
print("包安装状态：")
print("-" * 60)
all_ok = True
for pkg, (installed, info) in packages_status.items():
    if installed:
        print(f"✅ {pkg:12s} - 已安装 (版本: {info})")
    else:
        print(f"❌ {pkg:12s} - 未安装 ({info})")
        all_ok = False
print("-" * 60)

if all_ok:
    print("\n✅ 所有包都已安装！")
    print("\n测试导入关键模块...")
    try:
        from mmrotate.apis import init_detector
        from mmengine.config import Config
        from mmengine.runner import Runner
        print("✅ 关键模块导入成功！")
    except Exception as e:
        print(f"⚠️  关键模块导入失败: {e}")
else:
    print("\n❌ 有包未安装，请按以下步骤安装：")
    print()
    missing = [pkg for pkg, (installed, _) in packages_status.items() if not installed]
    
    if 'mmcv' in missing:
        print("1. mmcv:")
        print("   pip install mmcv-lite  # 推荐（轻量版，不需要编译）")
        print("   或")
        print("   pip install mmcv-full  # 完整版（需要预编译版本）")
        print()
    
    if 'mmengine' in missing:
        print("2. mmengine:")
        print("   pip install mmengine")
        print()
    
    if 'mmdet' in missing:
        print("3. mmdet:")
        print("   pip install mmdet")
        print()
    
    if 'mmrotate' in missing:
        print("4. mmrotate:")
        print("   pip install mmrotate")
        print()
    
    print("或者一次性安装所有依赖：")
    print("   pip install mmcv-lite mmengine mmdet mmrotate")

