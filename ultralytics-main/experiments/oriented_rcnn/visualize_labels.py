"""
将 YOLO OBB 格式的标签绘制在图像上

标签格式：class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标 0-1)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# DOTA 类别名称（与 dota_obb.yaml 中的类别对应）
DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'swimming-pool', 'container-crane', 'airport', 'helipad'
]

# 类别颜色（BGR格式，用于OpenCV）
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128),
    (255, 165, 0), (255, 192, 203), (0, 255, 127), (255, 20, 147)
]


def parse_yolo_obb_label(label_file: Path, img_width: int, img_height: int):
    """
    解析 YOLO OBB 格式的标签文件
    
    Args:
        label_file: 标签文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        list: [(class_id, class_name, points), ...]
        points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    annotations = []
    
    if not label_file.exists():
        return annotations
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            class_id = int(parts[0])
            if class_id < 0 or class_id >= len(DOTA_CLASSES):
                continue
            
            class_name = DOTA_CLASSES[class_id]
            
            # 归一化坐标 (0-1)
            coords = [float(x) for x in parts[1:9]]
            
            # 转换为绝对坐标
            points = []
            for i in range(0, 8, 2):
                x = int(coords[i] * img_width)
                y = int(coords[i + 1] * img_height)
                points.append((x, y))
            
            annotations.append((class_id, class_name, points))
    
    return annotations


def sort_points_clockwise(points):
    """
    将4个点按顺时针顺序排序，确保形成正确的矩形
    
    Args:
        points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Returns:
        排序后的点列表（按顺时针顺序）
    """
    if len(points) != 4:
        return points
    
    # 计算中心点
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4
    
    # 计算每个点相对于中心点的角度
    def get_angle(point):
        dx = point[0] - center_x
        dy = point[1] - center_y
        # 使用 atan2 计算角度，范围 [-π, π]
        angle = np.arctan2(dy, dx)
        # 转换为 [0, 2π] 范围，便于排序
        if angle < 0:
            angle += 2 * np.pi
        return angle
    
    # 按角度排序（顺时针）
    sorted_points = sorted(points, key=get_angle)
    
    return sorted_points


def draw_obb_on_image(image, annotations, line_thickness=2, font_scale=0.5, show_text=True):
    """
    在图像上绘制旋转边界框
    
    Args:
        image: 输入图像（numpy array）
        annotations: 标注列表 [(class_id, class_name, points), ...]
        points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] - 4个顶点坐标
        line_thickness: 线条粗细
        font_scale: 字体大小（如果 <= 0 则不显示文字）
        show_text: 是否显示类别名称
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    for class_id, class_name, points in annotations:
        if len(points) != 4:
            continue
        
        # 获取类别颜色
        color = COLORS[class_id % len(COLORS)]
        
        # 确保点的顺序正确（按顺时针排序）
        sorted_points = sort_points_clockwise(points)
        
        # 将点转换为numpy数组
        pts = np.array(sorted_points, dtype=np.int32)
        
        # 绘制旋转边界框（4个点的闭合多边形）
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_thickness)
        
        # 绘制4个顶点（可选，用于调试）
        for pt in sorted_points:
            cv2.circle(img, pt, 3, color, -1)
        
        # 在第一个点附近绘制类别名称（只有当 show_text=True 且 font_scale > 0 时才绘制）
        if show_text and font_scale > 0 and len(sorted_points) > 0:
            text_pos = (sorted_points[0][0], sorted_points[0][1] - 5)
            cv2.putText(img, class_name, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 
                       thickness=1, lineType=cv2.LINE_AA)
    
    return img


def visualize_labels(
    labels_dir: str,
    images_dir: str,
    output_dir: str,
    image_names: list = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_text: bool = True
):
    """
    将标签绘制在图像上并保存
    
    Args:
        labels_dir: 标签文件目录
        images_dir: 图像文件目录
        output_dir: 输出目录
        image_names: 要处理的图像名称列表（不带扩展名），如果为None则处理所有图像
        line_thickness: 线条粗细
        font_scale: 字体大小
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 如果指定了图像名称列表，只处理这些图像
    if image_names is not None:
        print(f"📁 处理指定的 {len(image_names)} 个图像")
        image_name_set = set(image_names)
    else:
        # 获取所有标签文件
        label_files = list(labels_dir.glob('*.txt'))
        if not label_files:
            print(f"⚠️  未找到标签文件: {labels_dir}")
            return
        image_name_set = {f.stem for f in label_files}
        print(f"📁 找到 {len(image_name_set)} 个标签文件，处理所有图像")
    
    print(f"   标签目录: {labels_dir}")
    print(f"   图像目录: {images_dir}")
    print(f"   输出目录: {output_dir}")
    print()
    
    processed_count = 0
    skipped_count = 0
    
    for img_name in tqdm(sorted(image_name_set), desc="处理进度"):
        # 查找标签文件
        label_file = labels_dir / f"{img_name}.txt"
        if not label_file.exists():
            skipped_count += 1
            print(f"⚠️  跳过 {img_name}：未找到标签文件")
            continue
        
        # 查找对应的图像文件
        img_file = None
        for ext in image_extensions:
            candidate = images_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            # 尝试大写扩展名
            for ext in image_extensions:
                candidate = images_dir / f"{img_name}{ext.upper()}"
                if candidate.exists():
                    img_file = candidate
                    break
        
        if img_file is None:
            skipped_count += 1
            print(f"⚠️  跳过 {img_name}：未找到对应图像文件")
            continue
        
        # 读取图像
        img = cv2.imread(str(img_file))
        if img is None:
            skipped_count += 1
            print(f"⚠️  跳过 {img_name}：无法读取图像文件")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 解析标签
        annotations = parse_yolo_obb_label(label_file, img_width, img_height)
        
        # 绘制标签
        img_with_labels = draw_obb_on_image(img, annotations, line_thickness, font_scale, show_text)
        
        # 保存图像
        output_file = output_dir / f"{img_name}.png"
        cv2.imwrite(str(output_file), img_with_labels)
        
        processed_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ 处理完成！")
    print(f"   成功处理: {processed_count} 个文件")
    if skipped_count > 0:
        print(f"   跳过: {skipped_count} 个文件")
    print(f"   输出目录: {output_dir}")
    print("=" * 60)


def main():
    """主函数"""
    # ========== 配置区域：在这里修改路径 ==========
    LABELS_DIR = r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard\labels\train"
    IMAGES_DIR = r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard\images\train"
    OUTPUT_DIR = r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard\images\show"
    
    # 要处理的图像名称列表（不带扩展名）
    # 如果为空列表 []，则处理所有图像
    # 如果指定了图像名称，则只处理这些图像
    IMAGE_NAMES = [
        # 在这里添加要处理的图像名称，例如：
        # "P0001",
        # "P0002",
    ]
    
    # 绘制参数
    LINE_THICKNESS = 2  # 线条粗细
    FONT_SCALE = 0      # 字体大小（设置为 0 或不显示文字时设为 False）
    SHOW_TEXT = False   # 是否显示类别名称（False 则不显示，True 则显示）
    # ============================================
    
    print("=" * 60)
    print("YOLO OBB 标签可视化工具")
    print("=" * 60)
    print(f"标签目录: {LABELS_DIR}")
    print(f"图像目录: {IMAGES_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    if IMAGE_NAMES:
        print(f"处理图像: {len(IMAGE_NAMES)} 个指定图像")
    else:
        print(f"处理图像: 所有图像")
    print(f"线条粗细: {LINE_THICKNESS}")
    print(f"显示文字: {SHOW_TEXT}")
    if SHOW_TEXT:
        print(f"字体大小: {FONT_SCALE}")
    print("=" * 60)
    print()
    
    visualize_labels(
        labels_dir=LABELS_DIR,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        image_names=IMAGE_NAMES if IMAGE_NAMES else None,
        line_thickness=LINE_THICKNESS,
        font_scale=FONT_SCALE,
        show_text=SHOW_TEXT
    )


if __name__ == '__main__':
    main()

