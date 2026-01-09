# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import *
    >>> import torch
    >>> import os
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DFC_Attention,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
# 从 research 目录导入自定义模块（保持兼容性）
try:
    import sys
    from pathlib import Path
    # 添加 research 模块到路径
    research_path = Path(__file__).parent.parent.parent.parent / "src" / "research"
    if research_path.exists():
        sys.path.insert(0, str(research_path.parent))
        from research.nn.modules.mca_attention import (
            MCAttention,
            AdaptiveAttention,
            SCAttention,
        )
        from research.nn.modules.afpn import (
            AFPN,
            AdaptiveFPN,
        )
        from research.nn.modules.sfe_drb import SFE_DRB
        from research.nn.modules.c3k2_roam import C3k2_ROAM
        from research.nn.modules.c3k2_dcn_ca import C3k2_DCN_CA
        from research.nn.modules.msff_small_head import (
            MSFF,
            LightMSFF,
            MSFFBlock,
        )
        from research.nn.modules.adaptive_detection_head import (
            AdaptiveScaleAwareHead,
            ASADH,
        )
        from research.nn.modules.adaptive_feature_alignment import (
            AdaptiveFeatureAlignment,
            AFA,
        )
    else:
        # 如果 research 目录不存在，尝试从本地导入（向后兼容）
        from .mca_attention import (
            MCAttention,
            AdaptiveAttention,
            SCAttention,
        )
        from .afpn import (
            AFPN,
            AdaptiveFPN,
        )
        from .sfe_drb import SFE_DRB
        from .c3k2_roam import C3k2_ROAM
        from .c3k2_dcn_ca import C3k2_DCN_CA
        from .msff_small_head import (
            MSFF,
            LightMSFF,
            MSFFBlock,
        )
        from .adaptive_detection_head import (
            AdaptiveScaleAwareHead,
            ASADH,
        )
        from .adaptive_feature_alignment import (
            AdaptiveFeatureAlignment,
            AFA,
        )
except ImportError:
    # 如果导入失败，尝试从本地导入（向后兼容）
    try:
        from .mca_attention import (
            MCAttention,
            AdaptiveAttention,
            SCAttention,
        )
        from .afpn import (
            AFPN,
            AdaptiveFPN,
        )
        from .sfe_drb import SFE_DRB
        from .c3k2_roam import C3k2_ROAM
        from .c3k2_dcn_ca import C3k2_DCN_CA
        from .msff_small_head import (
            MSFF,
            LightMSFF,
            MSFFBlock,
        )
        from .adaptive_detection_head import (
            AdaptiveScaleAwareHead,
            ASADH,
        )
        from .adaptive_feature_alignment import (
            AdaptiveFeatureAlignment,
            AFA,
        )
    except ImportError:
        # 如果都失败，定义占位符类
        MCAttention = None
        AdaptiveAttention = None
        SCAttention = None
        AFPN = None
        AdaptiveFPN = None
        SFE_DRB = None
        C3k2_ROAM = None
        C3k2_DCN_CA = None
        MSFF = None
        LightMSFF = None
        MSFFBlock = None
        AdaptiveScaleAwareHead = None
        ASADH = None
        AdaptiveFeatureAlignment = None
        AFA = None

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "DFC_Attention",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "LRPCHead",
    "ImagePoolingAttn",
    "MaxSigmoidAttnBlock",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "MCAttention",
    "AdaptiveAttention", 
    "SCAttention",
    "AFPN",
    "AdaptiveFPN",
    "SFE_DRB",
    "C3k2_ROAM",
    "C3k2_DCN_CA",
    "MSFF",
    "LightMSFF",
    "MSFFBlock",
    "AdaptiveScaleAwareHead",
    "ASADH",
    "AdaptiveFeatureAlignment",
    "AFA",
)
