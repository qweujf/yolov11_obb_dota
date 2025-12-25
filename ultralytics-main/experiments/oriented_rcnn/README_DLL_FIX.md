# 解决 mmrotate DLL 加载失败问题

## 问题描述

安装 `mmcv-full 1.7.2` 后，导入 `mmrotate` 时出现：
```
DLL load failed while importing _ext: 找不到指定的程序。
```

## 可能原因

1. **版本不匹配**：`mmcv-full 1.7.2` 是为 CUDA 11.8 + PyTorch 2.0.0 编译的，但你的环境是 PyTorch 2.4.1 + CUDA 12.1
2. **缺少运行时库**：缺少 Visual C++ Redistributable
3. **环境变量问题**：PATH 中缺少必要的 DLL 路径

## 解决方案

### 方案1：安装 Visual C++ Redistributable（最简单，先尝试）

1. 下载并安装：
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. 安装后**重启终端/IDE**
3. 再次测试导入

### 方案2：尝试安装与 CUDA 12.1 兼容的版本

```bash
# 卸载现有版本
pip uninstall mmcv-full mmdet mmrotate -y

# 尝试安装 CUDA 12.1 的版本（如果存在）
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0.0/index.html

# 如果上面不行，尝试 cu118（向后兼容）
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# 重新安装 mmdet 和 mmrotate
pip install mmdet==2.28.2
pip install mmrotate==0.3.4
```

### 方案3：降级 PyTorch（不推荐，但可能有效）

如果上述方案都不行，可以尝试降级 PyTorch：

```bash
# 卸载 PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装 PyTorch 2.0.0 + CUDA 11.8（与 mmcv-full 1.7.2 匹配）
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 然后重新安装 mmcv-full
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install mmdet==2.28.2
pip install mmrotate==0.3.4
```

### 方案4：使用 Docker 环境（最稳定）

如果以上方案都不行，建议使用 Docker 环境，可以避免 Windows 上的兼容性问题。

## 验证

安装完成后，测试导入：

```bash
python -c "from mmrotate.apis import init_detector; print('✅ 成功')"
```

## 注意事项

- **优先尝试方案1**（安装 Visual C++ Redistributable），这是最常见的解决方案
- 如果方案1不行，再尝试方案2
- 方案3（降级 PyTorch）会影响其他项目，谨慎使用
- 方案4（Docker）是最稳定的长期解决方案

