# 代码同步指南

## 📋 快速同步步骤

### 🏫 在学校电脑修改代码后 → 同步到公司电脑

```bash
# 1. 在学校电脑上：提交并推送代码
git add .
git commit -m "你的修改说明"
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

