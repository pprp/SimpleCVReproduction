from collections import OrderedDict
from  .operations import *


AUTOLA_PRIMITIVES = [
    "none",
    "skip_connect",
    "sep_conv_3x3",
    "dil_conv_3x3",
    "sep_conv_3x3_spatial",
    "dil_conv_3x3_spatial",
    "SE",
    "SE_A_M",
    "CBAM",
]

FULLPOOL_PRIMITIVES = [
    "none",
    "noise",
    "max_pool_3x3",
    "max_pool_5x5",
    "max_pool_7x7",
    "avg_pool_3x3",
    "avg_pool_5x5",
    "avg_pool_7x7",
    "strippool",
]

FULLCONV_PRIMITIVES = [
    "none",
    "noise",
    "dil_conv_3x3",
    "dil_conv_3x3_spatial",
    "dil_conv_5x5",
    "dil_conv_5x5_spatial",
    "sep_conv_3x3",
    "sep_conv_3x3_spatial",
    "sep_conv_5x5",
    "sep_conv_5x5_spatial",
    "conv_3x1_1x3",
    "conv_5x1_1x5",
]

HYBRID_PRIMITIVES = list(set([*FULLCONV_PRIMITIVES, *FULLPOOL_PRIMITIVES])) 


MAXPOOL_OPS = {
    # 最大池化 288
    # "max_pool": lambda C, K: nn.Sequential(
    #     nn.MaxPool2d(K, stride=1, padding=K // 2),
    #     nn.Conv2d(C, C, 3, 1, padding=1, bias=False, groups=C // 4),
    #     nn.BatchNorm2d(C, affine=False),
    # ),
    "max_pool_3x3": lambda C, stride, affine: nn.Sequential(
        nn.MaxPool2d(3, stride=1, padding=1),
        nn.Conv2d(C, C, 3, 1, padding=1, bias=False, groups=C // 4),
        nn.BatchNorm2d(C, affine=affine),
    ),
    "max_pool_5x5": lambda C, stride, affine: nn.Sequential(
        nn.MaxPool2d(5, stride=1, padding=2),
        nn.Conv2d(C, C, 3, 1, padding=1, bias=False, groups=C // 4),
        nn.BatchNorm2d(C, affine=affine),
    ),
    "max_pool_7x7": lambda C, stride, affine: nn.Sequential(
        nn.MaxPool2d(7, stride=1, padding=3),
        nn.Conv2d(C, C, 3, 1, padding=1, bias=False, groups=C // 4),
        nn.BatchNorm2d(C, affine=affine),
    ),
}

SEPCONV_OPS = {
    "sep_conv_3x3_spatial": lambda C, stride, affine: SepConvAttention(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5_spatial": lambda C, stride, affine: SepConvAttention(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
}

DILCONV_OPS = {
    "dil_conv_3x3_spatial": lambda C, stride, affine: DilConvAttention(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5_spatial": lambda C, stride, affine: DilConvAttention(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
}

SE_OPS = {
    "SE": lambda C, stride, affine: SE_A(C, reduction=4),  # channel attention
    "SE_A_M": lambda C, stride, affine: SE_A_M(C, reduction=4),  # channel attention v2
}

DECOMPOSE_OPS = {
    # 分解卷积 3x3 192
    "conv_3x1_1x3": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(
            C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False, groups=C // 4
        ),
        nn.Conv2d(
            C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False, groups=C // 4
        ),
        nn.BatchNorm2d(C, affine=affine),
    ),
    # 分解卷积 5x5 320
    "conv_5x1_1x5": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(
            C, C, (1, 5), stride=(1, stride), padding=(0, 2), bias=False, groups=C // 4
        ),
        nn.Conv2d(
            C, C, (5, 1), stride=(stride, 1), padding=(2, 0), bias=False, groups=C // 4
        ),
        nn.BatchNorm2d(C, affine=affine),
    ),
}

AVGPOOL_OPS = {
    # 平均池化 288
    # "avg_pool": lambda C, K: nn.Sequential(
    #     nn.AvgPool2d(K, stride=1, padding=K // 2),
    #     nn.Conv2d(C, C, 3, 1, 1, bias=False, groups=C // 4),
    #     nn.BatchNorm2d(C, affine=False),
    # ),
    "avg_pool_3x3": lambda C, stride, affine: nn.Sequential(
        nn.AvgPool2d(3, stride=1, padding=1),
        nn.Conv2d(C, C, 3, 1, 1, bias=False, groups=C // 4),
        nn.BatchNorm2d(C, affine=False),
    ),
    "avg_pool_5x5": lambda C, stride, affine: nn.Sequential(
        nn.AvgPool2d(5, stride=1, padding=2),
        nn.Conv2d(C, C, 3, 1, 1, bias=False, groups=C // 4),
        nn.BatchNorm2d(C, affine=False),
    ),
    "avg_pool_7x7": lambda C, stride, affine: nn.Sequential(
        nn.AvgPool2d(7, stride=1, padding=3),
        nn.Conv2d(C, C, 3, 1, 1, bias=False, groups=C // 4),
        nn.BatchNorm2d(C, affine=False),
    ),
}

SKIP_OPS = {
    "noise": lambda C, stride, affine: NoiseOp(stride, 0.0, 1.0),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
}


OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "CBAM": lambda C, stride, affine: CBAM(C, reduction_ratio=4),
    # 空洞卷积 d=1 or 2 or 4 or 8 304
    "conv_k3_dn": lambda C, N: ConvBnReLU(
        C, C, 3, 1, padding=N, dilation=N, groups=C // 4
    ),
    # Strip Pooling 232
    "strippool": lambda C, stride, affine: StripPool(C),
    # Global Average Pooling 288
    "gap": lambda C: GAPModule(C),
    **AVGPOOL_OPS,
    **MAXPOOL_OPS,
    **SE_OPS,
    **SEPCONV_OPS,
    **DILCONV_OPS,
    **DECOMPOSE_OPS,
    **SKIP_OPS,
}

spatial_spaces = {
    "autola": AUTOLA_PRIMITIVES,
    "fullpool": FULLPOOL_PRIMITIVES,
    "fullconv": FULLCONV_PRIMITIVES,
    "hybrid": HYBRID_PRIMITIVES,
}

PRIMITIVES = spatial_spaces["fullpool"]
