1111111111111# 代码同步指南

11111111111## 📋 快速同步步骤

### 🏫 在学校电脑修改代码后 → 同步到公司电脑

```bash
# 1. 在学校电脑上：提交并推送代码
git add .
git commit -m "你的修改说明"
git commit -m "mca3"
git push

# 2. 在公司电脑上：拉取最新代码
cd D:\code\yolov11_obb_dota
git pull
```

### 💼 在公司电脑修改代码后 → 同步到学校电脑

```bash
# 1. 在公司电脑上：提交并推送代码
git add .
git commit -m "你的修改说明"
git push

# 2. 在学校电脑上：拉取最新代码
cd D:\code\yolov11_obb_dota
git pull
```

---

## 🔄 完整工作流程

### 开始工作前（每次都要做）

```bash
# 先拉取最新代码，确保本地是最新的
git pull
```

### 修改代码后

```bash
# 1. 查看修改了哪些文件
git status

# 2. 添加所有修改的文件
git add .

# 3. 提交修改（写清楚你做了什么）
git commit -m "修改说明：例如：修复了数据处理bug"

# 4. 推送到 GitHub
git push
```

---

## 📝 提交说明（Commit Message）怎么写？

### 单个修改的写法

**格式：** `类型: 简短描述`

**常用类型：**
- `fix:` - 修复bug
- `feat:` - 新功能
- `update:` - 更新/改进
- `docs:` - 文档相关
- `refactor:` - 代码重构
- `style:` - 代码格式调整

**示例：**
```bash
git commit -m "fix: 修复数据集加载时的内存溢出问题"
git commit -m "feat: 添加新的数据增强方法"
git commit -m "update: 优化模型训练速度"
git commit -m "docs: 更新README说明文档"
```

### 一次修改多个文件/多个功能怎么办？

#### 方法1：分多次提交（推荐）⭐

**优点：** 每个提交只做一件事，历史清晰，容易回退

```bash
# 假设你修改了3个文件，分别做不同的事

# 1. 只提交第一个修改
git add ultralytics/data/dataset.py
git commit -m "fix: 修复数据集加载bug"

# 2. 提交第二个修改
git add scripts/split_dota_v2_standard.py
git commit -m "feat: 添加新的数据分割功能"

# 3. 提交第三个修改
git add ultralytics/models/yolo/obb/train.py
git commit -m "update: 优化OBB训练参数"

# 4. 一次性推送所有提交
git push
```

#### 方法2：一次提交，写多个要点

**适用场景：** 多个小修改，都是同一件事的不同部分

```bash
# 使用多行提交信息
git commit -m "fix: 修复数据处理相关问题

- 修复数据集加载时的内存溢出
- 修复标签文件读取错误
- 优化数据预处理速度"
```

**或者用简短的列表：**
```bash
git commit -m "update: 优化训练流程

修复数据加载bug，添加新的数据增强，优化训练参数"
```

#### 方法3：查看修改后决定

```bash
# 1. 先看看改了什么
git status
git diff

# 2. 如果修改都是相关的，一起提交
git add .
git commit -m "fix: 修复数据处理相关问题（包含3个文件的修改）"

# 3. 如果修改不相关，分别提交（参考方法1）
```

### 提交信息最佳实践

✅ **好的提交信息：**
```bash
git commit -m "fix: 修复OBB标签解析错误"
git commit -m "feat: 添加数据可视化功能"
git commit -m "update: 优化模型推理速度，提升30%"
```

❌ **不好的提交信息：**
```bash
git commit -m "修改"           # 太模糊
git commit -m "fix bug"        # 没说清楚什么bug
git commit -m "更新"           # 不知道更新了什么
git commit -m "asdf"           # 完全没意义
```

### 实际例子

**场景1：修复了一个bug**
```bash
git add ultralytics/data/dataset.py
git commit -m "fix: 修复空标签文件导致的程序崩溃"
git push
```

**场景2：添加了新功能**
```bash
git add scripts/new_feature.py
git commit -m "feat: 添加数据质量检查脚本"
git push
```

**场景3：修改了多个相关文件**
```bash
git add .
git commit -m "update: 优化数据处理流程

- 改进数据加载速度
- 修复内存泄漏问题
- 添加错误处理机制"
git push
```

**场景4：修改了多个不相关的文件**
```bash
# 分别提交
git add file1.py
git commit -m "fix: 修复bug1"

git add file2.py
git commit -m "feat: 添加功能2"

git add file3.py
git commit -m "docs: 更新文档"

# 一次性推送
git push
```

---

## ⚠️ 注意事项

1. **每次开始工作前先 `git pull`**，避免冲突
2. **提交信息要清晰**，方便以后查看历史
3. **数据集文件不会被同步**（已在 .gitignore 中忽略）
   - `dota1.0/`
   - `dota2.0/`
   - `dotav1.5/`
   - `zuhe/`
   - `*.jpg`, `*.png`, `*.jpeg`
   - `*.bin`, `*.hdf5`

---

## 🆘 遇到冲突怎么办？

如果 `git pull` 时提示冲突：

```bash
# 1. 查看冲突文件
git status

# 2. 手动解决冲突后
git add .
git commit -m "解决冲突"
git push
```

---

## 📝 常用命令速查

| 操作 | 命令 |
|------|------|
| 查看状态 | `git status` |
| 查看修改内容 | `git diff` |
| 拉取最新代码 | `git pull` |
| 提交代码 | `git add . && git commit -m "说明"` |
| 推送代码 | `git push` |
| 查看提交历史 | `git log --oneline` |

---

**记住：先 pull，再修改，最后 push！** 🚀

