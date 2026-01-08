"""
将 YOLO OBB 格式数据集转换为 DOTA 格式
用于 MMRotate/Oriented R-CNN 训练

YOLO OBB 格式：
- 标注文件：labels/train/xxx.txt
- 每行格式：class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标 0-1)

DOTA 格式：
- 标注文件：train/annfiles/xxx.txt
- 每行格式：x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult (绝对坐标)
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

# DOTA 类别名称（与 dota_obb.yaml 中的类别对应）
# 注意：DOTA 格式使用连字符，YOLO 配置中使用空格
# 映射关系：YOLO 配置中的 "storage tank" -> DOTA 格式的 "storage-tank"
DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'swimming-pool', 'container-crane', 'airport', 'helipad'
]


def parse_yolo_obb_line(line: str, img_width: int, img_height: int) -> Tuple[List[float], int]:
    """
    解析 YOLO OBB 格式的一行标注

    Args:
        line: YOLO OBB 格式的标注行
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        (bbox_coords, class_id): 边界框坐标（绝对坐标）和类别ID
    """
    parts = line.strip().split()
    if len(parts) < 9:
        return None, None

    class_id = int(parts[0])
    # 归一化坐标 (0-1)
    coords = [float(x) for x in parts[1:9]]

    # 转换为绝对坐标
    abs_coords = []
    for i in range(0, 8, 2):
        x = coords[i] * img_width
        y = coords[i + 1] * img_height
        abs_coords.extend([x, y])

    return abs_coords, class_id


def convert_yolo_to_dota_format(
        yolo_data_root: str,
        dota_output_root: str,
        split: str = 'train',
        convert_to_png: bool = False
):
    """
    将 YOLO OBB 格式转换为 DOTA 格式

    Args:
        yolo_data_root: YOLO 数据集根目录（包含 images/ 和 labels/）
        dota_output_root: DOTA 格式输出根目录
        split: 数据集划分（train/val/test）
        convert_to_png: 是否将图片从 JPG 转换为 PNG
    """
    yolo_root = Path(yolo_data_root)
    dota_root = Path(dota_output_root)

    # YOLO 格式路径
    yolo_images_dir = yolo_root / 'images' / split
    yolo_labels_dir = yolo_root / 'labels' / split

    # DOTA 格式路径
    dota_images_dir = dota_root / split / 'images'
    dota_annfiles_dir = dota_root / split / 'annfiles'

    # 创建输出目录
    dota_images_dir.mkdir(parents=True, exist_ok=True)
    dota_annfiles_dir.mkdir(parents=True, exist_ok=True)

    if not yolo_images_dir.exists():
        print(f"❌ 错误：YOLO 图像目录不存在: {yolo_images_dir}")
        return False

    if not yolo_labels_dir.exists():
        print(f"❌ 错误：YOLO 标注目录不存在: {yolo_labels_dir}")
        return False

    # 获取所有图像文件
    image_files = list(yolo_images_dir.glob('*.jpg')) + list(yolo_images_dir.glob('*.png')) + \
                  list(yolo_images_dir.glob('*.JPG')) + list(yolo_images_dir.glob('*.PNG'))

    if not image_files:
        print(f"⚠️  警告：在 {yolo_images_dir} 中未找到图像文件")
        return False

    print(f"📁 处理 {split} 数据集...")
    print(f"   图像目录: {yolo_images_dir}")
    print(f"   标注目录: {yolo_labels_dir}")
    print(f"   找到 {len(image_files)} 个图像文件")
    if convert_to_png:
        print(f"   🔄 将图片转换为 PNG 格式")

    converted_count = 0
    skipped_count = 0

    for img_file in image_files:
        # 对应的标注文件
        label_file = yolo_labels_dir / f"{img_file.stem}.txt"

        # 确定输出图像文件名
        if convert_to_png:
            # 转换为 PNG 格式
            dst_img = dota_images_dir / f"{img_file.stem}.png"
        else:
            # 保持原格式
            dst_img = dota_images_dir / img_file.name

        # 转换或复制图像文件
        if convert_to_png and img_file.suffix.lower() in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            # 需要转换格式
            try:
                from PIL import Image
                with Image.open(img_file) as img:
                    # 如果是 RGBA，转换为 RGB（PNG 支持 RGBA，但为了兼容性转换为 RGB）
                    if img.mode == 'RGBA':
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为 mask
                        rgb_img.save(dst_img, 'PNG')
                    else:
                        img.save(dst_img, 'PNG')
            except Exception as e:
                print(f"⚠️  警告：转换图片失败 {img_file}: {e}")
                # 如果转换失败，尝试直接复制
                shutil.copy2(img_file, dst_img)
        else:
            # 直接复制
            shutil.copy2(img_file, dst_img)

        # 读取图像尺寸（用于坐标转换）
        try:
            from PIL import Image
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️  警告：无法读取图像尺寸 {img_file}: {e}")
            # 使用默认尺寸（假设是 1024x1024，因为 split_dota_1024_standard）
            img_width, img_height = 1024, 1024

        # 创建 DOTA 格式的标注文件（使用输出图像的文件名，不带扩展名）
        output_img_stem = dst_img.stem  # 如果是 PNG，使用 .png 的 stem
        dota_ann_file = dota_annfiles_dir / f"{output_img_stem}.txt"

        if not label_file.exists():
            # 如果没有标注文件，创建空文件
            dota_ann_file.write_text("")
            skipped_count += 1
            continue

        # 读取 YOLO 格式标注
        dota_lines = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                coords, class_id = parse_yolo_obb_line(line, img_width, img_height)

                if coords is None or class_id is None:
                    continue

                # 检查类别ID是否有效
                if class_id < 0 or class_id >= len(DOTA_CLASSES):
                    print(f"⚠️  警告：无效的类别ID {class_id}，跳过")
                    continue

                class_name = DOTA_CLASSES[class_id]

                # DOTA 格式：x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
                # difficult 通常设为 0（非困难样本）
                dota_line = f"{coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f} {coords[3]:.2f} " \
                            f"{coords[4]:.2f} {coords[5]:.2f} {coords[6]:.2f} {coords[7]:.2f} " \
                            f"{class_name} 0\n"
                dota_lines.append(dota_line)

        # 写入 DOTA 格式标注文件
        with open(dota_ann_file, 'w', encoding='utf-8') as f:
            f.writelines(dota_lines)

        converted_count += 1

        if converted_count % 100 == 0:
            print(f"   已转换 {converted_count} 个文件...")

    print(f"✅ {split} 数据集转换完成！")
    print(f"   成功转换: {converted_count} 个文件")
    print(f"   跳过（无标注）: {skipped_count} 个文件")
    print(f"   输出目录: {dota_root / split}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='将 YOLO OBB 格式数据集转换为 DOTA 格式')
    parser.add_argument(
        '--yolo-root',
        type=str,
        default='D:/code/yolov11_obb_dota/zuhe/split_dota_1024_standard',
        help='YOLO 数据集根目录（包含 images/ 和 labels/）'
    )
    parser.add_argument(
        '--dota-root',
        type=str,
        default='D:/code/yolov11_obb_dota/zuhe/rcnn_dota',
        help='DOTA 格式输出根目录'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        help='要转换的数据集划分（train/val/test）'
    )
    parser.add_argument(
        '--convert-to-png',
        action='store_true',
        help='将图片从 JPG 转换为 PNG 格式'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO OBB → DOTA 格式转换工具")
    print("=" * 60)
    print(f"YOLO 数据集根目录: {args.yolo_root}")
    print(f"DOTA 输出根目录: {args.dota_root}")
    print(f"数据集划分: {args.splits}")
    print("=" * 60)
    print()

    # 检查 YOLO 数据集是否存在
    yolo_root = Path(args.yolo_root)
    if not yolo_root.exists():
        print(f"❌ 错误：YOLO 数据集根目录不存在: {yolo_root}")
        return 1

    # 转换每个数据集划分
    success = True
    for split in args.splits:
        print()
        if not convert_yolo_to_dota_format(args.yolo_root, args.dota_root, split, args.convert_to_png):
            success = False

    print()
    print("=" * 60)
    if success:
        print("✅ 所有数据集转换完成！")
        print(f"📁 DOTA 格式数据集位置: {args.dota_root}")
        print()
        print("现在可以在 config.yaml 中使用以下路径：")
        print(f"  data:")
        print(f"    path: {args.dota_root}")
    else:
        print("❌ 转换过程中出现错误，请检查上面的输出")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    import sys

    sys.exit(main())

