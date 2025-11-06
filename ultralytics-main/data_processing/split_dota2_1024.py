import os
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root=r"D:\code\yolov11_obb_dota\zuhe\try",
    save_dir=r"D:\code\yolov11_obb_dota\zuhe\try_split",
    rates=[0.5, 1.0, 1.5],  # multiscale 多尺度缩放比例，用于对原始图像进行缩放，以适应不同分辨率的目标检测需求
    gap=500, #滑动窗口的步长（单位：像素），控制切片之间的间隔
)


# split test set, without labels.
# split_test(
#     data_root=r"D:\code\yolov11_obb_dota\zuhe\raw",
#     save_dir=r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024",
#     rates=[0.5, 1.0, 1.5],  # multiscale
#     gap=500,
# )