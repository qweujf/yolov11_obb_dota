"""
可视化 C3k2 与 C3k2_DCA 中间特征的热力图，用于论文中的感受野对比图。

用法示例（在仓库根目录下）：

    cd ultralytics-main/experiments/c3k2_dcn_ca
    python visualize_heatmap.py ^
        --baseline-weights  D:/code/yolov11_obb_dota/runs/obb/baseline/weights/best.pt ^
        --dca-weights       D:/code/yolov11_obb_dota/runs/obb/c3k2_dcn_ca/weights/best.pt ^
        --images            D:/data/dota_obb/vis_samples ^
        --output-dir        D:/code/yolov11_obb_dota/vis/c3k2_dca

脚本不会依赖数据集配置，只需要若干待可视化的图像即可。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C3k2 vs C3k2_DCA 特征热力图可视化")
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default=r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\baseline\runs\obb\train\weights\best.pt",
        help="baseline 模型权重路径（使用标准 C3k2 的 YOLOv11-OBB），默认使用论文实验的基线权重",
    )
    parser.add_argument(
        "--dca-weights",
        type=str,
        default=(
            r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\c3k2_dcn_ca"
            r"\runs\obb\c3k2_dcn_ca\c3k2_dcn_ca_exp\weights\best.pt"
        ),
        help="改进模型权重路径（使用 C3k2_DCA 的 YOLOv11-OBB），默认使用论文实验的改进模型权重",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=r"D:\code\yolov11_obb_dota\zuhe\split_dota_1024_standard\images\train",
        help="单张图像路径或包含多张图像的目录，默认使用 DOTA 训练集切块中的样例图像",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            r"D:\code\yolov11_obb_dota\ultralytics-main\experiments\c3k2_dcn_ca"
            r"\runs\obb\c3k2_dcn_ca\c3k2_dcn_ca_exp\heatmap_vis"
        ),
        help="可视化结果保存目录，默认保存在 C3k2_DCA 实验目录下的 heatmap_vis 中",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="送入模型的输入尺寸（默认 1024，与训练保持一致）",
    )
    parser.add_argument(
        "--baseline-layer",
        type=str,
        default="C3k2",
        help="baseline 中用于可视化的模块类名（默认 C3k2）",
    )
    parser.add_argument(
        "--dca-layer",
        type=str,
        default="C3k2_DCN_CA",
        help="C3k2_DCA 模型中用于可视化的模块类名（默认 C3k2_DCN_CA）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    return parser.parse_args()


def add_repo_root_to_path() -> Path:
    """将仓库根目录加入 sys.path，便于导入 ultralytics."""
    import sys

    this_file = Path(__file__).resolve()
    # experiments/c3k2_dcn_ca/visualize_heatmap.py -> 仓库根目录在上上级
    repo_root = this_file.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_yolo_model(weights: Path, device: str):
    """加载 YOLO 模型（Ultralytics 接口），仅用于前向和中间特征提取。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.to(device)
    model.eval()
    return model


def find_last_module_by_class(model: torch.nn.Module, class_name: str) -> Tuple[torch.nn.Module, str]:
    """
    在 model.model.named_modules() 中寻找最后一个指定类名的模块。
    返回 (module, name)。若未找到，将抛出 ValueError。
    """
    target = None
    target_name = ""
    for name, m in model.model.named_modules():  # type: ignore[attr-defined]
        if m.__class__.__name__ == class_name:
            target = m
            target_name = name
    if target is None:
        raise ValueError(f"未在模型中找到类名为 '{class_name}' 的模块，请检查参数。")
    return target, target_name


def collect_activation(
    model,
    target_layer: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """
    前向一次，获取 target_layer 的激活特征，并生成上采样到输入尺寸的 activation heatmap。

    这里采用简单的“通道均值 + ReLU”方式，而非严格的 Grad-CAM，
    但足以用于论文中的感受野/响应区域可视化。
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(_m, _inp, out):
        activations["feat"] = out.detach()

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model.model(image_tensor)  # type: ignore[attr-defined]
    handle.remove()

    feat = activations["feat"]  # [1, C, H', W']
    feat = feat.mean(dim=1, keepdim=True)  # 通道均值
    feat = F.relu(feat)
    feat = feat / (feat.max() + 1e-6)

    cam = F.interpolate(
        feat,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return cam


def load_image(path: Path, imgsz: int, device: str) -> Tuple[torch.Tensor, np.ndarray]:
    """读取单张图像，返回模型输入张量和用于可视化的原始 RGB 图。"""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    img = rgb_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor, rgb_resized


def overlay_heatmap(
    img_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """将热力图叠加到灰度原图上，生成可视化结果。"""
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * gray_rgb).astype(np.uint8)
    return overlay


def gather_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def visualize_for_image(
    img_path: Path,
    baseline_model,
    dca_model,
    baseline_layer: torch.nn.Module,
    dca_layer: torch.nn.Module,
    imgsz: int,
    device: str,
    save_dir: Path,
) -> None:
    """对单张图像生成三行的可视化：原图 / baseline / C3k2_DCA。"""
    tensor, rgb = load_image(img_path, imgsz, device)

    cam_base = collect_activation(baseline_model, baseline_layer, tensor)
    cam_dca = collect_activation(dca_model, dca_layer, tensor)

    overlay_base = overlay_heatmap(rgb, cam_base)
    overlay_dca = overlay_heatmap(rgb, cam_dca)

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(4, 9))

    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(overlay_base)
    axes[1].set_title("Baseline (C3k2)")
    axes[1].axis("off")

    axes[2].imshow(overlay_dca)
    axes[2].set_title("C3k2_DCA")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = save_dir / f"{img_path.stem}_heatmap.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[保存] {out_path}")


def main() -> None:
    args = parse_args()
    repo_root = add_repo_root_to_path()

    baseline_weights = Path(args.baseline_weights)
    dca_weights = Path(args.dca_weights)
    img_path = Path(args.images)
    out_dir = Path(args.output_dir)

    if not baseline_weights.exists():
        raise FileNotFoundError(f"baseline 权重不存在: {baseline_weights}")
    if not dca_weights.exists():
        raise FileNotFoundError(f"C3k2_DCA 权重不存在: {dca_weights}")

    print(f"[信息] 仓库根目录: {repo_root}")
    print(f"[信息] 加载 baseline 模型: {baseline_weights}")
    baseline_model = load_yolo_model(baseline_weights, args.device)
    print(f"[信息] 加载 C3k2_DCA 模型: {dca_weights}")
    dca_model = load_yolo_model(dca_weights, args.device)

    # 找到用于可视化的模块
    base_layer, base_name = find_last_module_by_class(baseline_model, args.baseline_layer)
    dca_layer, dca_name = find_last_module_by_class(dca_model, args.dca_layer)
    print(f"[信息] baseline 可视化层: {base_name} ({base_layer.__class__.__name__})")
    print(f"[信息] C3k2_DCA 可视化层: {dca_name} ({dca_layer.__class__.__name__})")

    images = gather_images(img_path)
    if not images:
        raise FileNotFoundError(f"在 {img_path} 下未找到可用图像。")

    # 随机抽取 3 张图像进行可视化，避免在大规模训练集上全部处理
    if len(images) > 3:
        images = random.sample(images, 3)
    print(f"[信息] 随机选择 {len(images)} 张图像用于可视化")

    for p in images:
        visualize_for_image(
            p,
            baseline_model,
            dca_model,
            base_layer,
            dca_layer,
            imgsz=args.imgsz,
            device=args.device,
            save_dir=out_dir,
        )


if __name__ == "__main__":
    main()


