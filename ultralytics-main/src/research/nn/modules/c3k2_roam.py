import math
from typing import List, Union

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv


def _gn_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class OrientationSelectiveUnit(nn.Module):
    """Multi-direction depthwise convolutions to capture orientation cues."""

    def __init__(self, channels: int, k: int = 3, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                DWConv(channels, channels, (1, k), act=False),
                DWConv(channels, channels, (k, 1), act=False),
                DWConv(channels, channels, k, act=False),
                DWConv(channels, channels, k, d=2, act=False),
            ]
        )
        self.fuse = Conv(channels * len(self.branches), channels, 1, act=act)
        self.norm = nn.GroupNorm(_gn_groups(channels), channels)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        feats = self.fuse(feats)
        feats = self.norm(feats)
        return self.act(feats)


class RotationChannelReconstruction(nn.Module):
    """Channel refinement with grouped conv + selective kernel-style fusion."""

    def __init__(self, channels: int, groups: int = 4, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        groups = max(1, min(groups, channels))
        self.branch_a = Conv(channels, channels, 3, g=groups, act=act)
        self.branch_b = Conv(channels, channels, 5, g=groups, act=act)
        act_module = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.SiLU()
        hidden = max(channels // 4, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            act_module,
            nn.Conv2d(hidden, 2, 1, bias=True),
            nn.Softmax(dim=1),
        )
        self.point = Conv(channels, channels, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        weights = self.attn(a + b)
        fused = weights[:, 0:1] * a + weights[:, 1:2] * b
        return self.point(fused)


class RotationBottleneck(nn.Module):
    """Bottleneck with orientation conv + channel reconstruction."""

    def __init__(self, channels: int, shortcut: bool = True, g: int = 1, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        self.shortcut = shortcut
        self.orientation = OrientationSelectiveUnit(channels, act=act)
        self.reconstruct = RotationChannelReconstruction(channels, groups=max(1, g), act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.orientation(x)
        y = self.reconstruct(y)
        return x + y if self.shortcut else y


class RotationInvariantChannelAttention(nn.Module):
    """Channel attention using rotation-invariant pooling."""

    def __init__(self, channels: int, reduction: int = 4, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        hidden = max(channels // reduction, 8)
        act_module = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            act_module,
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.stack([self.pool(torch.rot90(x, k, dims=(2, 3))) for k in range(4)], dim=0).mean(0)
        return self.mlp(pooled)


class OrientationSpatialAttention(nn.Module):
    """Spatial attention with direction-sensitive kernels."""

    def __init__(self, channels: int, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        mid = max(channels // 2, 16)
        self.conv = nn.Sequential(
            Conv(channels, mid, 1, act=act),
            DWConv(mid, mid, 3, act=act),
            Conv(mid, 1, 1, act=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AngleAttention(nn.Module):
    """Angle-sensitive attention via gradient cues."""

    def __init__(self, channels: int, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        g = math.gcd(channels, channels)
        self.proj = nn.Sequential(
            Conv(2 * channels, channels, 1, act=act),
            Conv(channels, channels, 3, g=g, act=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_h = torch.abs(x - torch.roll(x, shifts=1, dims=2))
        grad_w = torch.abs(x - torch.roll(x, shifts=1, dims=3))
        feat = torch.cat([grad_h, grad_w], dim=1)
        return self.proj(feat)


class ROAM(nn.Module):
    """Rotation-Oriented Attention Module (channel + spatial + angle)."""

    def __init__(self, channels: int, reduction: int = 4, act: Union[nn.Module, bool] = nn.SiLU()):
        super().__init__()
        self.ca = RotationInvariantChannelAttention(channels, reduction=reduction, act=act)
        self.sa = OrientationSpatialAttention(channels, act=act)
        self.aa = AngleAttention(channels, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ca(x) * self.sa(x) * self.aa(x)


class C3k2_ROAM(nn.Module):
    """C3k2 module enhanced with Rotation-Oriented Adaptive Module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,  # kept for yaml compatibility
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        act: Union[nn.Module, bool] = nn.SiLU(),
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)
        self.blocks = nn.ModuleList(RotationBottleneck(self.c, shortcut=shortcut, g=g, act=act) for _ in range(n))
        self.roam = ROAM((2 + n) * self.c, act=act)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        feats: List[torch.Tensor] = [y[0], y[1]]
        for block in self.blocks:
            feats.append(block(feats[-1]))
        fused = torch.cat(feats, dim=1)
        fused = self.roam(fused)
        return self.cv2(fused)

