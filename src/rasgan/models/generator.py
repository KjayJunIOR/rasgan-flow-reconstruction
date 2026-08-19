from __future__ import annotations
from ..env import tf
from ..tf_layers import (
    Module, Conv2d, Conv1d, BatchNorm2d, Elementwise, UpSampling2d, Flatten, Sequential,
    GlobalAvgPool2d, MaxPool2d, LeakyReLU, Lambda, Concat, Swish, Linear,
    SubpixelConv2d, UpSampleBilinear2d,
)

# NOTE: Historically this project used pixel-shuffle (SubpixelConv2d). For SR on scientific fields,
# pixel-shuffle often produces periodic "stripe" / checkerboard artifacts. The generator below
# now defaults to resize-conv (bilinear + conv) upsampling via UpSampleBilinear2d.

from ..config import RRDB_DEPTH

# Keep original symbol name used in the extracted code.
rrdbdepth = RRDB_DEPTH

def _add_coords(x):
    # x: (N,C,H,W)
    shp = x.shape  # may contain None
    N = tf.shape(x)[0]
    H = tf.shape(x)[2] if shp[2] is None else shp[2]
    W = tf.shape(x)[3] if shp[3] is None else shp[3]

    y = tf.linspace(-1.0, 1.0, H)
    xlin = tf.linspace(-1.0, 1.0, W)
    yy, xx = tf.meshgrid(y, xlin, indexing="ij")  # (H,W)

    yy = tf.reshape(yy, (1, 1, H, W))
    xx = tf.reshape(xx, (1, 1, H, W))
    maps = tf.concat([yy, xx], axis=1)       # (1,2,H,W)
    maps = tf.tile(maps, [N, 1, 1, 1])          # (N,2,H,W)

    return tf.concat([x, maps], axis=1)     # (N,C+2,H,W)


class ECALayer(Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        k = int(k_size)
        if k % 2 == 0:  # ECA needs odd kernel
            k += 1
        self.channels = channels
        self.conv = Conv1d(out_channels=1, kernel_size=k, stride=1, padding='SAME', W_init=tf.keras.initializers.HeNormal(), data_format='channels_last', b_init=None) # we’ll feed (N, C, 1)

    def forward(self, x):
        # x: (N,C,H,W)
        # global avg pool over H,W -> (N,C)
        y = tf.reduce_mean(x, axis=[2, 3])         # (N,C)
        y = tf.expand_dims(y, axis=-1)             # (N,C,1)  (length=C, channels=1)
        y = self.conv(y)                            # (N,C,1)
        y = tf.sigmoid(y)                          # (N,C,1)
        # reshape to (N,C,1,1) and scale
        y = tf.expand_dims(y, axis=-1)
        return x * y

    def call(self, x, training=None):
        return self.forward(x)

class ResidualBlock(Module):
    """
    Dense-style residual block with symmetric kernel sizes [3,5,7,5,3] + ECA.
    The current implementation uses dilation=1 in every convolution and keeps
    64 input/output channels with dense internal concatenations.
    """
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr1 = Swish()
        self.concat1 = Concat(concat_dim=1)

        self.conv2 = Conv2d(
            out_channels=64, kernel_size=(5, 5), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr2 = Swish()
        self.concat2 = Concat(concat_dim=1)

        self.conv3 = Conv2d(
            out_channels=64, kernel_size=(7, 7), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr3 = Swish()
        self.concat3 = Concat(concat_dim=1)

        self.conv4 = Conv2d(
            out_channels=64, kernel_size=(5, 5), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        self.lr4 = Swish()
        self.concat4 = Concat(concat_dim=1)

        self.conv5 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1),
            dilation=1, act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(),
            data_format='channels_first', b_init=None
        )
        # Lightweight channel attention on the residual branch
        self.eca = ECALayer(channels=64, k_size=3)
        # Residual scaling (keeps block stable)
        self.lmd = Lambda(lambda x: 0.25 * x)

    def forward(self, x):
        z = self.conv1(x);  z = self.lr1(z)
        z1 = self.concat1([x, z])          # (N,128,H,W)
        z = self.conv2(z1); z = self.lr2(z)
        z2 = self.concat2([z1, z])         # (N,192,H,W)
        z = self.conv3(z2); z = self.lr3(z)
        z3 = self.concat3([z2, z])         # (N,256,H,W)
        z = self.conv4(z3); z = self.lr4(z)
        z4 = self.concat4([z3, z])         # (N,320,H,W)
        z = self.conv5(z4)                 # (N,64,H,W)

        # ECA attention on residual
        z = self.eca(z)
        z = self.lmd(z)

        return x + z

    def call(self, x, training=None):
        return self.forward(x)

# [ADD] Per-channel heads (2×res conv3x3 + 1×1 conv) — no BN
class _HeadRes(Module):
    def __init__(self, channels=256):
        super().__init__()
        self.c1 = Conv2d(out_channels=channels, kernel_size=(3,3), stride=(1,1),
                         padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(), data_format='channels_first', b_init=None)
        self.a1 = Swish()
        self.c2 = Conv2d(out_channels=channels, kernel_size=(3,3), stride=(1,1),
                         padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(), data_format='channels_first', b_init=None)
        self.scale = Lambda(lambda z: 0.5 * z)
    def forward(self, x):
        z = self.c1(x); z = self.a1(z)
        z = self.c2(z); z = self.scale(z)
        return x + z
    def call(self, x, training=None):
        return self.forward(x)

class _PerChannelHead(Module):
    def __init__(self, in_ch=256):
        super().__init__()
        self.b1 = _HeadRes(in_ch)
        self.b2 = _HeadRes(in_ch)
        self.out = Conv2d(out_channels=1, kernel_size=(1,1), stride=(1,1),
                          act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.Zeros(), data_format='channels_first')
    def forward(self, x):
        z = self.b1(x)
        z = self.b2(z)
        return self.out(z)
    def call(self, x, training=None):
        return self.forward(x)

class SRGAN_g(Module):
    """Generator.

    For POD->truth refinement tasks (same grid), use sr_scale=1 (default).
    For true super-resolution tasks where HR is 2× LR, use sr_scale=2.

    Tail order (as requested):
        subpixelconv2d (optional) -> conv2d (conv5) -> per-channel heads
    """
    def __init__(self, sr_scale: int = 1, upsample_mode: str = 'resizeconv'):
        super().__init__()
        self.g_arch = "rrdb"
        self.sr_scale = int(sr_scale)
        if self.sr_scale not in (1, 2):
            raise ValueError(f"sr_scale must be 1 or 2, got {self.sr_scale}")
        self.upsample_mode = str(upsample_mode)
        if self.upsample_mode not in ('resizeconv', 'pixelshuffle'):
            raise ValueError(f"upsample_mode must be 'resizeconv' or 'pixelshuffle', got {self.upsample_mode}")
        self.use_coords = False  # optionally append (y,x) coordinate channels

        self.conv1 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1),
            act=None, padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(), data_format='channels_first'
        )
        self.lrelu1 = Swish()

        self.residual_block = self.make_layer()
        self.lmd = Lambda(lambda x: 0.25 * x)
        self.add = Elementwise(combine_fn=tf.add)

        self.conv2 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1),
            padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.HeNormal(), data_format='channels_first', b_init=None
        )
        self.lrelu2 = Swish()

        # trunk to 256 channels
        self.conv3 = Conv2d(
            out_channels=256, kernel_size=(3, 3), stride=(1, 1),
            padding='SAME', padding_mode='REFLECT', W_init=tf.keras.initializers.TruncatedNormal(stddev=0.02), data_format='channels_first'
        )

        # Optional 2× upsample (LR -> HR) BEFORE final conv and heads.
        #
        # Stripe/checkerboard artifacts are most commonly introduced in this step.
        # Provide two implementations:
        #   - resizeconv: bilinear resize + conv (stable)
        #   - pixelshuffle: Conv + depth_to_space (fast), with ICNR init + blur to reduce artifacts
        if self.sr_scale == 2:
            if self.upsample_mode == 'pixelshuffle':
                self.up2x = SubpixelConv2d(
                    out_channels=256,
                    scale=2,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    act='swish',
                    padding='SAME',
                    W_init=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                    data_format='channels_first',
                    icnr=True,
                    blur=True,
                    name='up2x_pixelshuffle',
                )
            else:
                self.up2x = UpSampleBilinear2d(
                    out_channels=256,
                    scale=2,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding='SAME',
                    W_init=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                    data_format='channels_first',
                    act='swish',
                    # Explicitly match the resize convention used in conditioning/loss paths.
                    align_corners=False,
                    half_pixel_centers=False,
                    name='up2x_resizeconv',
                )
        else:
            self.up2x = None

        # conv5: post-(optional-upsample) refinement before heads
        self.conv5 = Conv2d(
            out_channels=256, kernel_size=(3, 3), stride=(1, 1),
            padding='SAME', W_init=tf.keras.initializers.TruncatedNormal(stddev=0.02), data_format='channels_first'
        )

        # per-channel heads then concat back to (N,3,H,W)
        self.headt = _PerChannelHead(in_ch=256)  # p
        self.headv = _PerChannelHead(in_ch=256)  # v
        self.headw = _PerChannelHead(in_ch=256)  # ωz
        self.fconcat = Concat(concat_dim=1)

        # For residual skip when sr_scale==2: upsample the INPUT (no conv, no channel change)
        self.up_in = UpSampling2d(scale=2, data_format='channels_first') if self.sr_scale == 2 else None

    def make_layer(self):
        layer_list = []
        for _ in range(rrdbdepth):
            layer_list.append(ResidualBlock())
        return Sequential(layer_list)

    # Accept **kwargs so loss wrappers can optionally pass sidecar info
    # (e.g. POD coefficients) without breaking this RRDB generator.
    def forward(self, x, **kwargs):
        x_in = x

        if self.use_coords:
            x = _add_coords(x)

        x = self.conv1(x)
        x = self.lrelu1(x)
        temp = x

        x = self.residual_block(x)
        x = self.lmd(x)
        x = self.add([temp, x])

        x = self.conv2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)

        # rearranged tail: subpixelconv2d -> conv2d -> heads
        if self.up2x is not None:
            x = self.up2x(x)

        x = self.conv5(x)

        xt = self.headt(x)
        xv = self.headv(x)
        xw = self.headw(x)
        res = self.fconcat([xt, xv, xw])
        base = self.up_in(x_in) if self.up_in is not None else x_in
        out = base + res

        return out
