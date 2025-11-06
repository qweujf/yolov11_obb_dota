import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ultralytics_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ultralytics_path)

from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO("yolov8s-obb.pt", task="obb")
    # 使用增强版YOLOv11-OBB模型
    # model = YOLO(r'D:\code\yolov11_obb_dota\ultralytics-main\ultralytics\cfg\models\11\yolo11-obb-enhanced.yaml')  # 使用增强版模型架构
    model = YOLO(r'D:\code\yolov11_obb_dota\ultralytics-main\ultralytics\cfg\models\11\yolo11-obb.yaml')
    # model = YOLO(r'D:\code\yolov11_obb_dota\ultralytics-main\ultralytics\cfg\models\11\yolo11-obb.yaml').load(r"D:\code\yolov11_obb_dota\ultralytics-main\yolo11n-obb.pt")  # 使用预训练权重训练
    # 开始训练
    results = model.train(
        data=r"D:\code\yolov11_obb_dota\ultralytics-main\data_processing\DOTAv2.yaml",     # 数据配置文件路径
        device='0',
        epochs=100,                        # 总训练轮数
        imgsz=682,                         # 输入图像尺寸
        batch=8,                          # 批处理大小
        name="obb_train",                  # 训练结果保存目录名称
        save=True,                         # 是否保存最佳模型权重
        exist_ok=True,                      # 允许覆盖已存在的实验结果目录
        verbose=True
    )