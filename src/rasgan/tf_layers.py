
"""TensorFlow/Keras layers that mirror the small TensorLayerX subset used in this repo.

Goal: remove the external TensorLayerX dependency while keeping the model code
structure close to the original extracted TLX version.

These wrappers intentionally cover ONLY what this project imports/uses.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from .env import tf


# -------------------------
# Core base class
# -------------------------

class Module(tf.keras.Model):
    """Minimal drop-in replacement for `tensorlayerx.nn.Module`.

    - Uses `forward()` as the main implementation method.
    - Exposes `init_build()` to mimic TLX' build-by-example behavior.
    - Keeps Keras `training` context propagation so BatchNorm works even if
      sublayers are called without explicitly passing `training=...`.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        """Create a Module.

        Some Keras classes (notably `tf.keras.Sequential` / `Functional`) pass
        extra keyword args such as `autocast` to base initializers. The original
        TensorLayerX `Module` initializer is permissive, so we mirror that by
        accepting and safely ignoring unknown kwargs.
        """

        # `autocast` is used by TF-Keras internals; older TF builds may not
        # accept it in `tf.keras.Model.__init__`, so we strip it defensively.
        kwargs.pop("autocast", None)

        try:
            super().__init__(name=name, **kwargs)
        except TypeError:
            # Fall back for very strict / older signatures.
            super().__init__(name=name)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def call(self, inputs, *args, training: Optional[bool] = None, **kwargs):
        # Keep the first model input explicit so Keras can inspect the call
        # signature without inventing an ``args`` input during auto-build.
        # We intentionally do NOT pass `training` into forward(), so legacy
        # forward() signatures stay valid while Keras still propagates the
        # training context to nested layers.
        return self.forward(inputs, *args, **kwargs)

    def _get_weights(self, name: str, shape, init, trainable: bool = True):
        """TLX-style helper used by the spectral-norm layers."""
        initializer = init if init is not None else "glorot_uniform"
        return self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def init_build(self, example_inputs):
        """Mimic TLX's `init_build` by running a single forward pass."""
        _ = self(example_inputs, training=False)
        return self

    def set_train(self):
        self.trainable = True
        return self

    def set_eval(self):
        self.trainable = False
        return self


# -------------------------
# Simple layer wrappers
# -------------------------

def _maybe_act(act):
    if act is None:
        return None
    if isinstance(act, str):
        a = act.lower()
        if a == "relu":
            return tf.keras.layers.ReLU()
        if a in ("lrelu", "leakyrelu"):
            return tf.keras.layers.LeakyReLU(alpha=0.2)
        if a == "swish":
            return tf.keras.layers.Activation(tf.nn.swish)
        if a == "tanh":
            return tf.keras.layers.Activation("tanh")
        if a == "sigmoid":
            return tf.keras.layers.Activation("sigmoid")
        if a == "gelu":
            return tf.keras.layers.Activation(tf.nn.gelu)
        raise ValueError(f"Unsupported activation string: {act}")
    # Callable / layer
    if isinstance(act, tf.keras.layers.Layer):
        return act
    if callable(act):
        return tf.keras.layers.Activation(act)
    raise ValueError(f"Unsupported activation type: {type(act)}")


class Conv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        dilation: Union[int, Tuple[int, int]] = 1,
        act: Optional[Union[str, Callable]] = None,
        padding: str = "SAME",
        # How to implement SAME padding.
        # "ZERO" (default) uses Keras' built-in zero padding.
        # "REFLECT" or "SYMMETRIC" applies explicit tf.pad then uses VALID conv.
        padding_mode: str = "ZERO",
        W_init=None,
        b_init=None,
        data_format: str = "channels_first",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        pad = padding.lower()
        if pad not in ("same", "valid"):
            pad = "same" if padding.upper() == "SAME" else "valid"

        # Store params needed for explicit padding.
        self.padding_mode = (padding_mode or "ZERO").upper()
        self._pad = pad
        self._data_format = data_format
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation

        conv_pad = pad
        if pad == "same" and self.padding_mode in ("REFLECT", "SYMMETRIC"):
            # Apply explicit padding in call(), then use a VALID convolution.
            conv_pad = "valid"

        self.conv = tf.keras.layers.Conv2D(
            name=name,
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            dilation_rate=dilation,
            padding=conv_pad,
            data_format=("channels_first" if data_format == "channels_first" else "channels_last"),
            use_bias=(b_init is not None),
            kernel_initializer=W_init if W_init is not None else "glorot_uniform",
            bias_initializer=b_init if b_init is not None else "zeros",
        )
        # Keep a TensorLayerX-style alias for code that expects `.layer`.
        self.layer = self.conv
        self.act = _maybe_act(act)

    def build(self, input_shape):
        if not self.conv.built:
            self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        if self._pad == "same" and self.padding_mode in ("REFLECT", "SYMMETRIC"):
            # Explicit symmetric padding to avoid subtle asymmetries (especially with dilation).
            if isinstance(self._kernel_size, int):
                kh = kw = self._kernel_size
            else:
                kh, kw = self._kernel_size
            if isinstance(self._dilation, int):
                dh = dw = self._dilation
            else:
                dh, dw = self._dilation

            pad_h = (kh - 1) * dh
            pad_w = (kw - 1) * dw
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            if self._data_format == "channels_first":
                paddings = [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]]
            else:
                paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

            x = tf.pad(x, paddings, mode=self.padding_mode)

        y = self.conv(x)
        return self.act(y) if self.act is not None else y


class Conv1d(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        act: Optional[Union[str, Callable]] = None,
        padding: str = "SAME",
        W_init=None,
        b_init=None,
        data_format: str = "channels_last",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        pad = padding.lower()
        if pad not in ("same", "valid"):
            pad = "same" if padding.upper() == "SAME" else "valid"

        self.conv = tf.keras.layers.Conv1D(
            name=name,
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            dilation_rate=dilation,
	        # Conv1D expects padding in {"same","valid"}; we normalized to `pad` above.
	        padding=pad,
            data_format=("channels_last" if data_format == "channels_last" else "channels_first"),
            use_bias=(b_init is not None),
            kernel_initializer=W_init if W_init is not None else "glorot_uniform",
            bias_initializer=b_init if b_init is not None else "zeros",
        )
        self.act = _maybe_act(act)

    def call(self, x, training=None):
        y = self.conv(x)
        return self.act(y) if self.act is not None else y


class BatchNorm2d(tf.keras.layers.Layer):
    def __init__(self, act=None, gamma_init=None, data_format: str = "channels_first", name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        axis = 1 if data_format == "channels_first" else -1
        self.bn = tf.keras.layers.BatchNormalization(name=name, axis=axis, momentum=0.9, epsilon=1e-5, gamma_initializer=gamma_init)
        self.act = _maybe_act(act)

    def call(self, x, training=None):
        y = self.bn(x, training=training)
        return self.act(y) if self.act is not None else y


class UpSampling2d(tf.keras.layers.Layer):
    """Differentiable bilinear upsampling for NCHW tensors.

    Nearest-neighbor upsampling can create grid-aligned stripe artifacts in SR GANs.
    This layer upsamples using bilinear interpolation (via NHWC) and returns NCHW.
    """

    def __init__(self, scale: Union[int, Tuple[int, int]] = 2, data_format: str = "channels_first",
                 name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.scale = scale
        self.data_format = data_format

    def call(self, x, training=None):
        if self.data_format not in ("channels_first", "NCHW"):
            # NHWC path, using the same explicit resize convention as NCHW.
            h = tf.shape(x)[1]
            w = tf.shape(x)[2]
            if isinstance(self.scale, (tuple, list)):
                sh, sw = int(self.scale[0]), int(self.scale[1])
            else:
                sh = sw = int(self.scale)
            size = tf.cast(tf.stack([h * sh, w * sw]), tf.int32)
            try:
                return tf.raw_ops.ResizeBilinear(
                    images=x, size=size, align_corners=False, half_pixel_centers=False
                )
            except Exception:
                return tf.image.resize(x, [h * sh, w * sw], method="bilinear")

        # NCHW -> NHWC
        x_nhwc = tf.transpose(x, [0, 2, 3, 1])
        h = tf.shape(x_nhwc)[1]
        w = tf.shape(x_nhwc)[2]
        if isinstance(self.scale, (tuple, list)):
            sh = int(self.scale[0])
            sw = int(self.scale[1])
        else:
            sh = int(self.scale)
            sw = int(self.scale)

        size = tf.cast(tf.stack([h * sh, w * sw]), tf.int32)
        try:
            x_up = tf.raw_ops.ResizeBilinear(
                images=x_nhwc,
                size=size,
                align_corners=False, half_pixel_centers=False,
            )
        except Exception:
            x_up = tf.image.resize(x_nhwc, [h * sh, w * sw], method="bilinear", antialias=False)
        # NHWC -> NCHW
        return tf.transpose(x_up, [0, 3, 1, 2])


class UpSampleBilinear2d(tf.keras.layers.Layer):
    """Resize-conv upsampling (bilinear resize + Conv2D) for NCHW.

    Notes:
    - Pixel-shuffle and strided deconvs can create line/checkerboard artifacts.
      Using resize+conv is usually more stable.
    - We expose align_corners/half_pixel_centers to better match external
      interpolation conventions.

    Default alignment:
      align_corners=False, half_pixel_centers=False
    The flags are exposed explicitly so users can match the convention used to
    construct their paired grids.
    """

    def __init__(
        self,
        out_channels: int,
        scale: int = 2,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        W_init=None,
        data_format="channels_first",
        act: str | None = "lrelu",
        align_corners: bool = False,
        half_pixel_centers: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.scale = int(scale)
        self.data_format = data_format
        self.align_corners = bool(align_corners)
        self.half_pixel_centers = bool(half_pixel_centers)

        if padding not in ("SAME", "VALID"):
            raise ValueError(f"UpSampleBilinear2d padding must be 'SAME' or 'VALID', got: {padding}")

        # Use the project Conv2d wrapper so we can opt into REFLECT padding to
        # reduce seam/band artifacts caused by repeated SAME-zero padding.
        self.conv = Conv2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            act=None,
            padding=padding,
            padding_mode="REFLECT",
            W_init=W_init,
            b_init=None,
            data_format="channels_first",
        )

        self.act = _maybe_act(act)

    def call(self, x, training=False):
        # Expect NCHW
        if self.data_format != "channels_first":
            raise ValueError("UpSampleBilinear2d currently supports channels_first (NCHW) only")

        x_nhwc = tf.transpose(x, [0, 2, 3, 1])
        h = tf.shape(x_nhwc)[1]
        w = tf.shape(x_nhwc)[2]
        new_size = tf.stack([h * self.scale, w * self.scale])

        # tf.image.resize has fixed alignment semantics; use raw op for explicit control.
        x_up = tf.raw_ops.ResizeBilinear(
            images=x_nhwc,
            size=new_size,
            align_corners=self.align_corners,
            half_pixel_centers=self.half_pixel_centers,
        )
        x_up = tf.transpose(x_up, [0, 3, 1, 2])
        y = self.conv(x_up, training=training)

        return self.act(y) if self.act is not None else y

class SubpixelConv2d(tf.keras.layers.Layer):
    """Conv + PixelShuffle upsampling (TensorLayerX-compatible).

    PixelShuffle can create periodic stripes/grid artifacts when the preceding
    convolution is initialized naively. Two well-known mitigations are supported:

    - ICNR initialization (Aitken et al., 2017): initializes the Conv2D kernel so
      each sub-pixel group starts identically.
    - Post-shuffle blur (fixed 3x3 binomial kernel): lightly smooths the
      rearranged feature map to suppress residual checkerboard/stripe energy.

    This layer is *generic* and does not require explicit physical grids.

    Note: tf.nn.depth_to_space operates on NHWC. For NCHW we transpose to NHWC,
    apply, then transpose back.
    """

    def __init__(
        self,
        out_channels: int,
        scale: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        act: Optional[Union[str, Callable]] = None,
        padding: str = "SAME",
        W_init=None,
        data_format: str = "channels_first",
        b_init=None,
        icnr: bool = True,
        blur: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.out_channels = int(out_channels)
        self.scale = int(scale)
        self.padding = padding
        self.data_format = data_format
        self.act = _maybe_act(act)
        self.icnr = bool(icnr)
        self.blur = bool(blur)

        # Conv2d to produce out_channels*(scale^2)
        self.conv = Conv2d(
            out_channels=self.out_channels * (self.scale**2),
            kernel_size=kernel_size,
            stride=stride,
            act=None,
            padding=self.padding,
            padding_mode="REFLECT",
            W_init=W_init,
            data_format=data_format,
            b_init=b_init,
            name=name,
        )

    def build(self, input_shape):
        # Ensure underlying conv is built so kernel exists.
        self.conv.build(input_shape)
        if self.icnr:
            try:
                self._apply_icnr()
            except Exception:
                # If anything goes wrong, fall back to whatever initializer was used.
                pass
        super().build(input_shape)

    def _apply_icnr(self):
        """Apply ICNR init to the underlying Conv2D kernel."""
        import numpy as _np

        # Kernel is [kh, kw, in_ch, out_ch] in tf.keras Conv2D
        k = self.conv.conv.kernel
        k_shape = tuple(int(d) for d in k.shape)
        if len(k_shape) != 4:
            return
        kh, kw, in_ch, out_ch = k_shape
        r2 = self.scale * self.scale
        if out_ch % r2 != 0:
            return
        out_ch_base = out_ch // r2

        init = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'
        )
        base = init(shape=(kh, kw, in_ch, out_ch_base), dtype=k.dtype)
        base_np = base.numpy()
        tiled = _np.repeat(base_np, repeats=r2, axis=3)
        k.assign(tf.cast(tiled, k.dtype))

    @staticmethod
    def _binomial_blur_nhwc(x: tf.Tensor) -> tf.Tensor:
        # Fixed 3x3 binomial kernel (1 2 1)^T(1 2 1)/16
        k2 = tf.constant([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=x.dtype) / 16.0
        k2 = k2[:, :, None, None]  # (3,3,1,1)
        c = tf.shape(x)[-1]
        filt = tf.tile(k2, [1, 1, c, 1])
        return tf.nn.depthwise_conv2d(x, filt, strides=[1, 1, 1, 1], padding='SAME')

    def call(self, x):
        x = self.conv(x)

        # Pixel shuffle needs NHWC
        if self.data_format == "channels_first":
            x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.depth_to_space(x, self.scale)

        if self.blur:
            x = self._binomial_blur_nhwc(x)

        if self.data_format == "channels_first":
            x = tf.transpose(x, [0, 3, 1, 2])

        return self.act(x) if self.act is not None else x

class Flatten(tf.keras.layers.Layer):
    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.f = tf.keras.layers.Flatten(name=name)

    def call(self, x, training=None):
        return self.f(x)


class GlobalAvgPool2d(tf.keras.layers.Layer):
    def __init__(self, data_format: str = "channels_first", name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=name, data_format=data_format)

    def call(self, x, training=None):
        return self.gap(x)


class MaxPool2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding="SAME", data_format: str = "channels_first", name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        pad = padding.lower()
        if pad not in ("same", "valid"):
            pad = "same" if padding.upper() == "SAME" else "valid"
        self.mp = tf.keras.layers.MaxPooling2D(name=name, pool_size=kernel_size, strides=stride, padding=pad, data_format=data_format)

    def call(self, x, training=None):
        return self.mp(x)


class Linear(tf.keras.layers.Layer):
    def __init__(self, out_features: int, act=None, W_init=None, b_init=None, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.dense = tf.keras.layers.Dense(
            out_features,
            name=name,
            activation=None,
            use_bias=(b_init is not None),
            kernel_initializer=W_init if W_init is not None else "glorot_uniform",
            bias_initializer=b_init if b_init is not None else "zeros",
        )
        self.act = _maybe_act(act)

    def call(self, x, training=None):
        y = self.dense(x)
        return self.act(y) if self.act is not None else y


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.2, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.l = tf.keras.layers.LeakyReLU(alpha=alpha, name=name)

    def call(self, x, training=None):
        return self.l(x)


class Swish(tf.keras.layers.Layer):
    def call(self, x, training=None):
        return tf.nn.swish(x)


class Lambda(tf.keras.layers.Layer):
    def __init__(self, fn: Callable, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.fn = fn

    def call(self, x, training=None):
        return self.fn(x)


class Concat(tf.keras.layers.Layer):
    def __init__(self, concat_dim: int = 1, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.axis = concat_dim

    def call(self, xs: Sequence[tf.Tensor], training=None):
        return tf.concat(list(xs), axis=self.axis)


class Elementwise(tf.keras.layers.Layer):
    def __init__(self, combine_fn: Callable = tf.add, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.combine_fn = combine_fn

    def call(self, xs: Sequence[tf.Tensor], training=None):
        a, b = xs
        return self.combine_fn(a, b)


class Sequential(tf.keras.Sequential, Module):
    """Sequential that also behaves like Module (forward)."""

    def __init__(self, layers: Optional[Sequence[tf.keras.layers.Layer]] = None, name: Optional[str] = None, **kwargs):
        tf.keras.Sequential.__init__(self, layers=list(layers) if layers is not None else [], name=name)

    def forward(self, x):
        return tf.keras.Sequential.call(self, x)


def Input(shape):
    # Lightweight helper to keep older init_build patterns readable.
    return tf.keras.Input(shape=shape[1:])
