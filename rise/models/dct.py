# MBConv portions adapted from efficientnet-pytorch (Apache-2.0).
"""JPEG DCT encoder, retaining the trained checkpoint parameter names."""

import collections
import math
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch.utils import (
    get_same_padding_conv2d,
    MemoryEfficientSwish,
    Swish,
    drop_connect,
)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "num_repeat",
        "kernel_size",
        "stride",
        "expand_ratio",
        "input_filters",
        "output_filters",
        "se_ratio",
        "id_skip",
    ],
)
GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "width_coefficient",
        "depth_coefficient",
        "image_size",
        "dropout_rate",
        "num_classes",
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "drop_connect_rate",
        "depth_divisor",
        "min_depth",
        "include_top",
    ],
)
global_params = GlobalParams(
    width_coefficient=1.8,
    depth_coefficient=2.6,
    image_size=528,
    dropout_rate=0.0,
    num_classes=1000,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=0.001,
    drop_connect_rate=0.0,
    depth_divisor=8,
    min_depth=None,
    include_top=True,
)


def get_width_and_height_from_size(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=25):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = (
            1 - global_params.batch_norm_momentum
        )  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = (
            block_args.id_skip
        )  # whether to use skip connection and drop connect
        inp = self._block_args.input_filters  # number of input channels
        oup = (
            self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class DCTEncoder(nn.Module):
    def __init__(
        self,
        dct_n=21,
        dct_embed_dim=64,
        qt_n=64,
        qt_embed_dim=16,
        mbconv_dim=256,
        out_dim=None,
        patch_size=16,
        mbconv_repeat_n=1,
        momentum=0.01,
    ):
        super(DCTEncoder, self).__init__()
        self.dct_n = dct_n
        self.qt_n = qt_n
        self.patch_size = patch_size
        self.out_dim = mbconv_dim if out_dim is None else out_dim
        self.base_block_size = 8
        if (
            self.patch_size < self.base_block_size
            or self.patch_size % self.base_block_size != 0
        ):
            raise ValueError(
                f"patch_size must be a multiple of {self.base_block_size}, got {self.patch_size}"
            )
        self.pixel_unshuffle_factor = self.patch_size // self.base_block_size

        self.dct_proj = nn.Embedding(dct_n, dct_n).from_pretrained(torch.eye(dct_n))
        self.dct_proj.weight.requires_grad = False
        for param in self.dct_proj.parameters():
            param.requires_grad = False

        self.qt_proj = nn.Embedding(qt_n, qt_embed_dim)

        self.dct_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=dct_n,
                out_channels=dct_embed_dim,
                kernel_size=3,
                stride=1,
                dilation=8,
                padding=8,
            ),
            nn.BatchNorm2d(dct_embed_dim, momentum=momentum),
            nn.ReLU(inplace=True),
        )

        self.dct_conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dct_embed_dim,
                out_channels=qt_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(qt_embed_dim, momentum=momentum),
            nn.ReLU(inplace=True),
        )

        self.mbconv_blocks = nn.Sequential(
            nn.Conv2d(
                in_channels=qt_embed_dim * 2,
                out_channels=mbconv_dim,
                kernel_size=8,
                stride=8,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mbconv_dim, momentum=momentum),
            nn.ReLU(inplace=True),
            MBConvBlock(
                BlockArgs(
                    num_repeat=mbconv_repeat_n,
                    kernel_size=3,
                    stride=[1],
                    expand_ratio=6,
                    input_filters=mbconv_dim,
                    output_filters=mbconv_dim,
                    se_ratio=0.25,
                    id_skip=True,
                ),
                global_params,
            ),
            MBConvBlock(
                BlockArgs(
                    num_repeat=mbconv_repeat_n,
                    kernel_size=3,
                    stride=[1],
                    expand_ratio=6,
                    input_filters=mbconv_dim,
                    output_filters=mbconv_dim,
                    se_ratio=0.25,
                    id_skip=True,
                ),
                global_params,
            ),
            MBConvBlock(
                BlockArgs(
                    num_repeat=mbconv_repeat_n,
                    kernel_size=3,
                    stride=[1],
                    expand_ratio=6,
                    input_filters=mbconv_dim,
                    output_filters=mbconv_dim,
                    se_ratio=0.25,
                    id_skip=True,
                ),
                global_params,
            ),
        )

        self.spatial_reducer = (
            nn.Identity()
            if self.pixel_unshuffle_factor == 1
            else nn.PixelUnshuffle(self.pixel_unshuffle_factor)
        )

        self.projector = nn.Sequential(
            nn.Conv2d(
                mbconv_dim * (self.pixel_unshuffle_factor**2),
                self.out_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_dim, momentum=momentum),
        )

    def forward(self, dct, qt):
        dct_feat = self.dct_proj(dct).permute(0, 3, 1, 2).contiguous()
        dct_feat = self.dct_conv_2(self.dct_conv_1(dct_feat))

        b, c, h, w = dct_feat.shape
        # ---- QT modulation (block-wise) ----
        qt_feat = self.qt_proj(qt.long())  # (B, 8, 8, C)
        qt_feat = qt_feat.permute(0, 3, 1, 2)  # (B, C, 8, 8)

        # Tile each block to 8×8 pixels: (B, C, H, W), broadcast-friendly
        qh, qw = qt_feat.shape[-2], qt_feat.shape[-1]
        qt_feat = qt_feat.repeat_interleave(max(1, h // qh), dim=2).repeat_interleave(
            max(1, w // qw), dim=3
        )
        qt_feat = qt_feat[:, :, :h, :w]

        # Element-wise modulation; no extra reshapes / permutes
        times = dct_feat * qt_feat

        cat_feat = torch.cat([times, dct_feat], dim=1)

        output_feat = self.mbconv_blocks(cat_feat)

        output_feat = self.spatial_reducer(output_feat)

        target_h = max(1, h // self.patch_size)
        target_w = max(1, w // self.patch_size)
        if output_feat.shape[-2:] != (target_h, target_w):
            raise RuntimeError(
                "Unexpected DCT token grid after PixelUnshuffle: "
                f"got {output_feat.shape[-2:]}, expected {(target_h, target_w)}"
            )

        output_feat = self.projector(output_feat)
        output_feat = output_feat.flatten(2).transpose(1, 2).contiguous()

        return output_feat
