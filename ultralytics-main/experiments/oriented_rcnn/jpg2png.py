"""
将 JPG/JPEG 图片批量转换为 PNG 格式
支持递归处理子目录
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_jpg_to_png(input_path: str, output_path: str = None, recursive: bool = True, delete_original: bool = False):
    """
    将 JPG/JPEG 图片转换为 PNG 格式

    Args:
        input_path: 输入路径（文件或目录）
        output_path: 输出路径（如果为 None，则在原位置转换）
        recursive: 是否递归处理子目录
        delete_original: 是否删除原始 JPG 文件
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ 错误：路径不存在: {input_path}")
        return False

    # 收集所有需要转换的图片文件
    image_files = []

    if input_path.is_file():
        # 单个文件
        if input_path.suffix.lower() in ['.jpg', '.jpeg']:
            image_files.append(input_path)
    else:
        # 目录
        if recursive:
            # 递归查找所有 JPG/JPEG 文件
            image_files = list(input_path.rglob('*.jpg')) + list(input_path.rglob('*.jpeg')) + \
                          list(input_path.rglob('*.JPG')) + list(input_path.rglob('*.JPEG'))
        else:
            # 只查找当前目录
            image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + \
                          list(input_path.glob('*.JPG')) + list(input_path.glob('*.JPEG'))

    if not image_files:
        print(f"⚠️  未找到 JPG/JPEG 图片文件")
        return False

    print(f"📁 找到 {len(image_files)} 个 JPG/JPEG 文件")
    print(f"   输入路径: {input_path}")
    if output_path:
        print(f"   输出路径: {output_path}")
    if delete_original:
        print(f"   ⚠️  警告：转换后将删除原始 JPG 文件")
    print()

    converted_count = 0
    failed_count = 0

    # 处理每个文件
    for jpg_file in tqdm(image_files, desc="转换进度"):
        try:
            # 确定输出文件路径
            if output_path:
                # 如果指定了输出路径，保持相对目录结构
                if input_path.is_file():
                    # 输入是文件
                    png_file = Path(output_path) / f"{jpg_file.stem}.png"
                else:
                    # 输入是目录，保持相对路径
                    rel_path = jpg_file.relative_to(input_path)
                    png_file = Path(output_path) / rel_path.parent / f"{jpg_file.stem}.png"
                    png_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # 在原位置转换
                png_file = jpg_file.parent / f"{jpg_file.stem}.png"

            # 检查 PNG 文件是否已存在
            if png_file.exists():
                print(f"⚠️  跳过（PNG 已存在）: {jpg_file.name}")
                continue

            # 打开并转换图片
            with Image.open(jpg_file) as img:
                # 如果是 RGBA，转换为 RGB（PNG 支持 RGBA，但为了兼容性可以转换为 RGB）
                # 这里保持原模式，PNG 支持 RGB 和 RGBA
                img.save(png_file, 'PNG', optimize=True)

            converted_count += 1

            # 删除原始文件（如果指定）
            if delete_original:
                jpg_file.unlink()

        except Exception as e:
            print(f"❌ 转换失败 {jpg_file}: {e}")
            failed_count += 1

    print()
    print("=" * 60)
    print(f"✅ 转换完成！")
    print(f"   成功转换: {converted_count} 个文件")
    if failed_count > 0:
        print(f"   失败: {failed_count} 个文件")
    print("=" * 60)

    return True


def main():
    # ========== 配置区域：在这里修改路径 ==========
    # 输入路径（可以是文件或目录）
    # JPG 文件所在目录
    INPUT_PATH = r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard\images\val"

    # 输出路径（如果为 None，则在原位置转换）
    # PNG 文件输出目录
    OUTPUT_PATH = r"D:\code\yolov11_obb_dota\zuhe\rcnn_dota\val\images"

    # 是否递归处理子目录
    # 如果只处理 train 目录下的文件（不处理子目录），设为 False
    RECURSIVE = False

    # 是否删除原始 JPG 文件（谨慎使用！）
    DELETE_ORIGINAL = False
    # ============================================

    # 如果提供了命令行参数，则使用命令行参数（向后兼容）
    import sys
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description='将 JPG/JPEG 图片批量转换为 PNG 格式',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  # 转换单个文件
  python convert_jpg_to_png.py input.jpg

  # 转换目录中的所有 JPG 文件（在原位置）
  python convert_jpg_to_png.py input_dir/

  # 转换并保存到指定目录
  python convert_jpg_to_png.py input_dir/ --output output_dir/

  # 递归转换子目录
  python convert_jpg_to_png.py input_dir/ --recursive

  # 转换后删除原始 JPG 文件
  python convert_jpg_to_png.py input_dir/ --delete-original
            """
        )

        parser.add_argument(
            'input',
            type=str,
            nargs='?',
            default=INPUT_PATH,
            help='输入文件或目录路径'
        )
        parser.add_argument(
            '--output', '-o',
            type=str,
            default=OUTPUT_PATH,
            help='输出目录路径（如果为 None，则在原位置转换）'
        )
        parser.add_argument(
            '--recursive', '-r',
            action='store_true',
            default=RECURSIVE,
            help='递归处理子目录（默认：True）'
        )
        parser.add_argument(
            '--no-recursive',
            action='store_false',
            dest='recursive',
            help='不递归处理子目录'
        )
        parser.add_argument(
            '--delete-original', '-d',
            action='store_true',
            default=DELETE_ORIGINAL,
            help='转换后删除原始 JPG 文件'
        )

        args = parser.parse_args()

        input_path = args.input
        output_path = args.output
        recursive = args.recursive
        delete_original = args.delete_original
    else:
        # 直接使用配置区域的设置
        input_path = INPUT_PATH
        output_path = OUTPUT_PATH
        recursive = RECURSIVE
        delete_original = DELETE_ORIGINAL

    print("=" * 60)
    print("JPG → PNG 图片格式转换工具")
    print("=" * 60)
    print(f"输入路径: {input_path}")
    if output_path:
        print(f"输出路径: {output_path}")
    else:
        print(f"输出路径: 原位置（覆盖）")
    print(f"递归处理: {recursive}")
    print(f"删除原始文件: {delete_original}")
    print("=" * 60)
    print()

    convert_jpg_to_png(
        input_path=input_path,
        output_path=output_path,
        recursive=recursive,
        delete_original=delete_original
    )

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())

