import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

src_dir = repo_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import torch
from ultralytics import YOLO


def test_module_forward():
    from research.nn.modules.c3k2_roam import C3k2_ROAM

    module = C3k2_ROAM(256, 256, n=2).eval()
    x = torch.randn(1, 256, 80, 80)
    with torch.no_grad():
        y = module(x)
    print(f"[C3k2_ROAM] in={tuple(x.shape)}, out={tuple(y.shape)}")


def test_model_build_and_forward():
    model_yaml = repo_root / "configs" / "model" / "yolo11-obb-sfe-drb.yaml"
    assert model_yaml.exists(), f"模型定义不存在: {model_yaml}"

    model = YOLO(str(model_yaml))
    core = model.model  # type: ignore[attr-defined]
    core.eval()
    try:
        core.info(detailed=False, verbose=True, imgsz=640)  # type: ignore[attr-defined]
    except Exception:
        pass

    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        _ = core(dummy)
    print("[Model] forward ok with random input 640x640")


if __name__ == "__main__":
    test_module_forward()
    test_model_build_and_forward()

