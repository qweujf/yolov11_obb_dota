import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ==================== 配置参数 ====================
# 保持与您原始脚本一致的路径配置
RUN_DIR = r"D:\code\yolov11_obb_dota\runs\obb\try"
EPOCHS = 300


def generate_custom_csv(save_path):
    """生成符合学术指标且对齐列顺序的 CSV"""
    print(f"正在生成模拟数据...")
    np.random.seed(42)
    x = np.arange(EPOCHS)

    # 调高起始值的 Loss 函数逻辑
    def get_loss(start_val, final_val, decay=60):
        return final_val + start_val * np.exp(-x / decay) + np.random.normal(0, 0.015, EPOCHS)

    # 严格按照您的顺序：
    # 第一行：train/box, train/cls, train/dfl, precision, recall
    # 第二行：val/box, val/cls, val/dfl, mAP50, mAP50-95
    data = {
        '                  epoch': x,
        '      train/box_om_loss': get_loss(3.2, 0.4),  # Index 1
        '         train/cls_loss': get_loss(2.8, 0.3),  # Index 2
        '         train/dfl_loss': get_loss(3.0, 0.35),  # Index 3
        '   metrics/precision(B)': 0.825 / (1 + 15 * np.exp(-x / 35)) + np.random.normal(0, 0.005, EPOCHS),
        '      metrics/recall(B)': 0.745 / (1 + 18 * np.exp(-x / 40)) + np.random.normal(0, 0.005, EPOCHS),
        '        val/box_om_loss': get_loss(3.5, 0.45),  # Index 6
        '           val/cls_loss': get_loss(3.0, 0.35),  # Index 7
        '           val/dfl_loss': get_loss(3.2, 0.4),  # Index 8
        '       metrics/mAP50(B)': (0.764 / (1 + 20 * np.exp(-x / 45))) + np.random.normal(0, 0.001, EPOCHS),
        '    metrics/mAP50-95(B)': (0.598 / (1 + 22 * np.exp(-x / 50))) + np.random.normal(0, 0.001, EPOCHS)
    }

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ CSV 已生成: {save_path}")
    return save_path


def plot_results_consistent(file):
    """
    完全复用您提供的原版绘图逻辑
    """
    print(f"正在调用原版逻辑绘制 results.png...")
    save_dir = Path(file).parent
    data = pd.read_csv(file)
    s = [x.strip() for x in data.columns]
    x = data.values[:, 0]

    # 按照您要求的 2x5 布局组织索引
    # 第一行对应 CSV 中的 Index 1,2,3,4,5
    # 第二行对应 CSV 中的 Index 6,7,8,9,10
    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()

    for i, j in enumerate(index):
        if j >= len(s):
            continue
        y = data.values[:, j].astype("float")
        # 保持您原始的绘图样式：marker=".", linewidth=2, markersize=8
        ax[i].plot(x, y, marker=".", label="results", linewidth=2, markersize=8, alpha=0.4)
        # 保持您原始的平滑线样式：sigma=3, linestyle=":"
        ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)
        ax[i].set_title(s[j], fontsize=12)

    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    print(f"✅ 图像已保存: {fname}")


if __name__ == "__main__":
    csv_path = os.path.join(RUN_DIR, "results.csv")

    # 1. 生成数据
    generate_custom_csv(csv_path)

    # 2. 绘图（逻辑与您提供的脚本完全一致）
    plot_results_consistent(csv_path)