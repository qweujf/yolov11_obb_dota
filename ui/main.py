import sys
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QTableWidgetItem
from PyQt6.QtGui import QPixmap, QImage


# 导入您生成的 UI 类
from main_window import Ui_MainWindow
import sys
import os
from pathlib import Path

# 获取当前 main.py 所在的目录 (ui 文件夹)
current_dir = Path(__file__).resolve().parent

# 定位到 ultralytics-main 文件夹的绝对路径
# 结构为：项目根目录/ultralytics-main/ultralytics
pkg_root = current_dir.parent / "ultralytics-main"

# 将该路径加入系统搜索路径
if pkg_root.exists() and str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))  # 使用 insert(0, ...) 确保优先搜索

# 现在尝试导入
try:
    from ultralytics import YOLO
    print("Ultralytics 源码加载成功")
except ImportError as e:
    print(f"导入失败，请检查路径。当前尝试路径: {pkg_root}")
    print(f"详细错误: {e}")


class DetectionThread(QtCore.QThread):
    """后台推理线程：防止界面卡死"""
    # 定义信号：传输处理后的图片和检测到的数据列表
    finished_sig = QtCore.pyqtSignal(np.ndarray, list)

    def __init__(self, model, image_path, conf, nms):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.conf = conf
        self.nms = nms

    def run(self):
        # 执行推理
        results = self.model.predict(
            source=self.image_path,
            conf=self.conf,
            iou=self.nms,
            save=False
        )

        # 获取第一张图的结果 (OBB)
        result = results[0]
        # 绘制检测框后的 numpy 数组 (BGR 格式)
        annotated_frame = result.plot()

        # 解析 OBB 结果数据用于填充表格
        # OBB 数据通常包含：class_id, conf, xywhr (中心x, 中心y, 宽, 高, 弧度)
        detection_data = []
        if result.obb is not None:
            boxes = result.obb.xywhr.cpu().numpy()
            clss = result.obb.cls.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()

            for i in range(len(boxes)):
                cls_name = result.names[int(clss[i])]
                # 提取数据：序号, 类别, 置信度, x, y, w, h
                row = [
                    str(i + 1),
                    cls_name,
                    f"{confs[i]:.2f}",
                    f"{boxes[i][0]:.1f}",
                    f"{boxes[i][1]:.1f}",
                    f"{boxes[i][2]:.1f}",
                    f"{boxes[i][3]:.1f}"
                ]
                detection_data.append(row)

        # 发送信号回到主线程
        self.finished_sig.emit(annotated_frame, detection_data)


class DragDropLabel(QLabel):
    """支持拖拽加载图像的自定义标签类"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                event.accept()
                return
        event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.window().load_image_from_path(file_path)


class MyYoloWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.init_drag_label()

        # 核心变量初始化
        self.model = None
        self.current_image_path = None

        self.init_slots()

    def init_drag_label(self):
        orig_geo = self.label_3.geometry()
        orig_parent = self.label_3.parent()
        orig_style = self.label_3.styleSheet()
        self.label_3.deleteLater()
        self.label_3 = DragDropLabel(orig_parent)
        self.label_3.setGeometry(orig_geo)
        self.label_3.setStyleSheet(orig_style)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setText("图像显示区域\n(支持拖拽)")

    def init_slots(self):
        self.pushButton.clicked.connect(self.open_image_dialog)
        self.pushButton_5.clicked.connect(self.load_model)
        self.pushButton_6.clicked.connect(self.start_detection)  # 开始检测按钮
        self.horizontalSlider.valueChanged.connect(self.update_conf_label)
        self.horizontalSlider_2.valueChanged.connect(self.update_nms_label)

    def load_model(self):
        """加载模型逻辑"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "YOLO Weights (*.pt)")
        if path:
            try:
                self.model = YOLO(path)
                print(f"模型加载成功: {path}")
                self.pushButton_5.setText("模型已就绪")
            except Exception as e:
                print(f"模型加载失败: {e}")

    def start_detection(self):
        """点击‘开始检测’按钮后的逻辑"""
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载模型！")
            return
        if self.current_image_path is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载图像！")
            return

        # 1. 禁用按钮并更改文字
        self.pushButton_6.setEnabled(False)
        self.pushButton_6.setText("检测中...")

        # 2. 获取滑动条当前的阈值
        conf_val = self.horizontalSlider.value() / 100.0
        nms_val = self.horizontalSlider_2.value() / 100.0

        # 3. 启动后台推理线程
        self.thread = DetectionThread(self.model, self.current_image_path, conf_val, nms_val)
        self.thread.finished_sig.connect(self.on_detection_finished)
        self.thread.start()

    def on_detection_finished(self, frame, data):
        """当推理结束时的回调函数"""
        # 1. 更新图像显示区域 (将 OpenCV BGR 转为 QPixmap)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label_3.setPixmap(pixmap.scaled(self.label_3.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        # 2. 更新结果展示区域 (表格)
        self.tableWidget.setRowCount(0)  # 先清空表格
        for row_idx, row_data in enumerate(data):
            self.tableWidget.insertRow(row_idx)
            for col_idx, item_text in enumerate(row_data):
                self.tableWidget.setItem(row_idx, col_idx, QTableWidgetItem(item_text))

        # 3. 更新提示标签
        self.label_2.setText(f" 检测结果：共检测到 {len(set([d[1] for d in data]))} 个类别， {len(data)} 个目标")

        # 4. 恢复按钮状态
        self.pushButton_6.setEnabled(True)
        self.pushButton_6.setText("开始检测")
        print("推理任务已完成。")

    # 其余加载图片和滑动条逻辑保持不变...
    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.jpg *.png *.bmp)")
        if path: self.load_image_from_path(path)

    def load_image_from_path(self, path):
        self.current_image_path = path
        pixmap = QPixmap(path)
        self.label_3.setPixmap(pixmap.scaled(self.label_3.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def update_conf_label(self, v):
        self.label_4.setText(f"置信度阈值：{v / 100.0:.2f}")

    def update_nms_label(self, v):
        self.label_5.setText(f"NMS阈值：{v / 100.0:.2f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyYoloWindow()
    window.show()
    sys.exit(app.exec())