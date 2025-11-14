import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# 允许导入 research 模块
src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import torch
from ultralytics import YOLO


def test_module_forward():
    from research.nn.modules.sfe_drb import SFE_DRB
    from research.nn.modules.c3k2_roam import C3k2_ROAM

    m = SFE_DRB(256, 256, use_residual=True).eval()
    x = torch.randn(1, 256, 80, 80)
    hr = torch.randn(1, 256, 160, 160)  # 高分辨率，会在模块内部对齐到 80×80

    with torch.no_grad():
        y = m([x, hr])
    print(f"[Module] in={tuple(x.shape)}, hr={tuple(hr.shape)}, out={tuple(y.shape)}")

    c3 = C3k2_ROAM(256, 256, n=2).eval()
    with torch.no_grad():
        z = c3(torch.randn(1, 256, 80, 80))
    print(f"[Module] C3k2_ROAM out={tuple(z.shape)}")


def test_model_build_and_forward():
    model_yaml = repo_root / "configs" / "model" / "yolo11-obb-sfe-drb.yaml"
    assert model_yaml.exists(), f"model yaml not found: {model_yaml}"

    model = YOLO(str(model_yaml))
    # 打印结构摘要
    try:
        model.model.info(detailed=False, verbose=True, imgsz=640)  # type: ignore[attr-defined]
    except Exception:
        pass

    # 随机前向（仅验证可跑通，不代表真实性能）
    x = torch.randn(1, 3, 640, 640)
    core = model.model  # type: ignore[attr-defined]
    core.eval()
    with torch.no_grad():
        _ = core(x)
    print("[Model] forward ok with random input 640x640")


if __name__ == "__main__":
    test_module_forward()
    test_model_build_and_forward()


