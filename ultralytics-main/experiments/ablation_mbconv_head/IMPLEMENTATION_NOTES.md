# MBConv 检测头实现说明

## 重要提示

当前创建的配置文件使用了 `OBB_MBConv` 类，该类需要您自行实现。配置文件中的参数格式为：

```yaml
- [[19, 22, 25], 1, OBB_MBConv, [nc, 1, cls_se, reg_se, expand_ratio, reg_replace_layers]]
```

参数说明：
- `nc`: 类别数量
- `1`: ne (extra parameters，角度预测)
- `cls_se`: 分类分支是否使用SE注意力（True/False）
- `reg_se`: 回归分支是否使用SE注意力（True/False）
- `expand_ratio`: MBConv扩张系数（默认3）
- `reg_replace_layers`: 回归分支替换的层数（默认2，仅替换前两层）

## 需要实现的内容

### 1. MBConv 模块

需要在 `ultralytics/nn/modules/conv.py` 中实现 MBConv 模块：

```python
class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) module.
    
    Args:
        c1: 输入通道数
        c2: 输出通道数
        expand_ratio: 扩张系数 k
        se: 是否使用SE注意力模块
    """
    def __init__(self, c1, c2, expand_ratio=3, se=False):
        # 实现MBConv结构
        # 1x1 Conv (扩展) -> 3x3 DWConv -> SE (可选) -> 1x1 Conv (压缩) -> 残差连接
        pass
```

### 2. OBB_MBConv 检测头类

需要在 `ultralytics/nn/modules/head.py` 中实现 `OBB_MBConv` 类：

```python
class OBB_MBConv(OBB):
    """
    OBB detection head with MBConv modules.
    
    Args:
        nc: 类别数量
        ne: extra parameters (角度预测)
        cls_se: 分类分支是否使用SE
        reg_se: 回归分支是否使用SE
        expand_ratio: MBConv扩张系数
        reg_replace_layers: 回归分支替换的层数
    """
    def __init__(self, nc=80, ne=1, cls_se=False, reg_se=True, expand_ratio=3, reg_replace_layers=2, ch=()):
        # 实现基于MBConv的检测头
        # cv2 (回归分支): 根据reg_se和reg_replace_layers决定是否使用MBConv+SE
        # cv3 (分类分支): 根据cls_se决定是否使用MBConv+SE
        # cv4 (角度分支): 保持标准Conv结构
        pass
```

### 3. 模块注册

在 `ultralytics/nn/modules/__init__.py` 中注册新模块：

```python
from .conv import MBConv
from .head import OBB_MBConv

__all__ = [
    # ... 其他模块
    "MBConv",
    "OBB_MBConv",
]
```

### 4. 在 tasks.py 中支持

在 `ultralytics/nn/tasks.py` 的 `parse_model` 函数中，确保 `OBB_MBConv` 能够正确解析参数。

## 配置对照表

| 配置文件 | cls_se | reg_se | 说明 |
|---------|--------|--------|------|
| mbconv-reg-se.yaml | False | True | 分类无SE，回归有SE |
| mbconv-cls-se.yaml | True | False | 分类有SE，回归无SE |
| mbconv-both-se.yaml | True | True | 分类有SE，回归有SE |
| mbconv-cls-no-se-reg-se.yaml | False | True | 分类无SE，回归有SE（推荐） |

## 替代方案

如果暂时无法实现 `OBB_MBConv` 类，可以考虑：

1. **方案A**: 创建4个不同的OBB类（如 `OBB_MBConv_RegSE`, `OBB_MBConv_ClsSE` 等），每个类对应一种配置
2. **方案B**: 修改现有的 `OBB` 类，添加参数来支持MBConv配置
3. **方案C**: 使用环境变量或配置文件来动态配置检测头结构

## 参考实现

可以参考 `ultralytics/nn/modules/head.py` 中的 `Detect` 和 `OBB` 类的实现，特别是：
- `cv2` (回归分支) 的构建逻辑
- `cv3` (分类分支) 的构建逻辑
- `cv4` (角度分支) 的构建逻辑

