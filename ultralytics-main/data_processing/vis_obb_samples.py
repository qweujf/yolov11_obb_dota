# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 可视化切片数据集中的 OBB 标注（将多边形画到图像上）。
# 使用：直接运行本脚本，路径在顶部常量中配置。
# """
#
# from pathlib import Path
# import random
# import cv2
# import numpy as np
#
# # ===== 按需修改：数据与输出目录 =====
# IMG_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\images\train")
# LBL_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\labels\train")
# OUT_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\vis_train")
# SAMPLE_NUM = 50  # 随机抽样数量
# # ===================================
#
# VALID_IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp")
#
#
# def draw_obb(im: np.ndarray, xy_norm: np.ndarray, color=(0, 255, 0), thickness=2):
#     """xy_norm: (4,2) 归一化坐标，范围[0,1]"""
#     h, w = im.shape[:2]
#     pts = xy_norm.copy()
#     pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * w
#     pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * h
#     pts = pts.astype(np.int32).reshape(-1, 1, 2)
#     cv2.polylines(im, [pts], True, color, thickness, lineType=cv2.LINE_AA)
#
#
# def visualize_one(img_path: Path, lbl_path: Path, out_path: Path):
#     im = cv2.imread(str(img_path))
#     if im is None:
#         return False, f"读图失败: {img_path}"
#     if not lbl_path.exists():
#         return False, f"缺少同名标签: {lbl_path.name}"
#
#     lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
#     for ln in lines:
#         ps = ln.split()
#         if len(ps) < 9:
#             continue
#         cls = int(float(ps[0]))  # 可用于选色
#         coords = np.array([float(x) for x in ps[1:9]], dtype=np.float32).reshape(4, 2)
#         color = (0, 255, 0)
#         if cls % 3 == 1:
#             color = (255, 0, 0)
#         elif cls % 3 == 2:
#             color = (0, 128, 255)
#         draw_obb(im, coords, color=color)
#
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), im)
#     return True, f"保存: {out_path.name}"
#
#
# def main():
#     assert IMG_DIR.exists(), f"图像目录不存在: {IMG_DIR}"
#     assert LBL_DIR.exists(), f"标签目录不存在: {LBL_DIR}"
#     imgs = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in VALID_IMG_EXTS]
#     if not imgs:
#         print("⚠️ 未找到图像文件")
#         return
#
#     random.shuffle(imgs)
#     sel = imgs[: min(SAMPLE_NUM, len(imgs))]
#     ok, fail = 0, 0
#     for p in sel:
#         out_path = OUT_DIR / (p.stem + ".jpg")
#         lbl_path = LBL_DIR / (p.stem + ".txt")
#         success, msg = visualize_one(p, lbl_path, out_path)
#         if success:
#             ok += 1
#         else:
#             fail += 1
#             print("⚠️", msg)
#     print(f"完成，可视化成功 {ok} 张，失败 {fail} 张 → {OUT_DIR}")
#
#
# if __name__ == "__main__":
#     main()


# data_processing/audit_split_pairs.py
from pathlib import Path

IMG_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\images\train")
LBL_DIR = Path(r"D:\code\yolov11_obb_dota\zuhe\try\labels\train")
FIX = False  # True 则删除无匹配的图片/标签

ex = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

imgs = {p.stem: p for p in IMG_DIR.iterdir() if p.suffix in ex}
lbls = {p.stem: p for p in LBL_DIR.glob("*.txt")}

only_img = sorted(set(imgs) - set(lbls))
only_lbl = sorted(set(lbls) - set(imgs))

print(f"总图像: {len(imgs)}, 总标签: {len(lbls)}")
print(f"仅图片无标签: {len(only_img)}")
print(f"仅标签无图片: {len(only_lbl)}")

if only_img:
    print("样例仅图片:", only_img[:10])
if only_lbl:
    print("样例仅标签:", only_lbl[:10])

if FIX:
    for s in only_img:
        imgs[s].unlink(missing_ok=True)
    for s in only_lbl:
        lbls[s].unlink(missing_ok=True)
    print("已删除不匹配文件。")