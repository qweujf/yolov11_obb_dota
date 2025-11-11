import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Prefer Ultralytics' Conv/DWConv/LightConv
    from ultralytics.nn.modules.conv import Conv, DWConv, LightConv
except Exception:  # fallback if path differs
    from ultralytics.nn.modules import Conv, DWConv, LightConv  # type: ignore


class SFE_DRB(nn.Module):
    """
    Small-object Feature Enhancement with Dilated-Residual Bridge (SFE-DRB).

    This block is designed to replace MCAttention at the same backbone positions.
    It enhances fine-grained details for small objects by combining:
      1) Dilated-Depthwise Pyramid (DDP): parallel DWConv with dilations {1, 2, 3}
      2) Gated Fine-grained Fusion (GFF): gate-controlled fusion of a higher-resolution skip

    Inputs:
      - x: Tensor or List[Tensor]; if list, expects [current, high_res] where high_res is an earlier higher-res feature

    Args:
      c1 (int): input channels
      c2 (int): output channels
      use_residual (bool): whether to add input residual
    """

    def __init__(self, c1: int, c2: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        hidden = c1  # keep channel size

        # DDP: three parallel DWConv with different dilations, followed by 1x1 fuse
        self.ddp_dw1 = DWConv(c1, hidden, k=3, s=1, d=1, act=True)
        self.ddp_dw2 = DWConv(c1, hidden, k=3, s=1, d=2, act=True)
        self.ddp_dw3 = DWConv(c1, hidden, k=3, s=1, d=3, act=True)
        self.ddp_fuse = Conv(hidden * 3, hidden, k=1, s=1, act=True)

        # Align for high-resolution input (channel-adaptive)
        # 默认一个与 c1 匹配的对齐层，并在前向时按需为不同通道数的 hr 动态创建并缓存
        self.align_default = Conv(c1, hidden, k=1, s=1, act=True)
        self.align_bank = nn.ModuleDict()

        # Gate generator: squeeze -> 1x1 -> sigmoid (scalar per spatial position via broadcast)
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(hidden * 2, 1, k=1, s=1, act=False),
            nn.Sigmoid(),
        )

        # Refinement with lightweight conv
        self.refine = LightConv(hidden, hidden, k=3)
        self.out = Conv(hidden, c2, k=1, s=1, act=True)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x, hr = x[0], x[1]
        else:
            x, hr = x, None

        identity = x

        # DDP
        y1 = self.ddp_dw1(x)
        y2 = self.ddp_dw2(x)
        y3 = self.ddp_dw3(x)
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.ddp_fuse(y)

        # GFF: optional higher-resolution fusion with gating
        if hr is not None:
            # spatial match
            if hr.shape[-2:] != x.shape[-2:]:
                hr = F.interpolate(hr, size=x.shape[-2:], mode="nearest")
            # channel match: pick or build align layer for current hr channels
            hr_c = hr.shape[1]
            if hr_c == y.shape[1]:
                # already same channels as hidden, use identity projection
                hr_a = hr
            else:
                key = str(hr_c)
                align = self.align_bank.get(key, None)
                if align is None:
                    align = Conv(hr_c, y.shape[1], k=1, s=1, act=True)
                    self.align_bank[key] = align
                hr_a = align(hr)
            g = self.gate_gen(torch.cat([y, hr_a], dim=1))  # [B,1,1,1]
            y = y + g * hr_a

        # refine and project
        y = self.refine(y)
        y = self.out(y)

        if self.use_residual and identity.shape == y.shape:
            y = y + identity
        return y


