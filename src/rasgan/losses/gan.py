from __future__ import annotations

import numpy as np

from ..config import D_SOLO_CHAN_P, EDGE_TANH_GAIN
from ..env import tf

# Sobel kernels for a single NCHW channel.
_SOBEL_KX_C1 = tf.constant(
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32).reshape(
        3, 3, 1, 1
    )
)
_SOBEL_KY_C1 = tf.constant(
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32).reshape(
        3, 3, 1, 1
    )
)


def _sobel_mag_channel(residual, idx):
    """Return the Sobel magnitude for one channel of an NCHW residual."""
    ch = residual[:, idx : idx + 1, :, :]
    ch_nhwc = tf.transpose(ch, [0, 2, 3, 1])
    gx = tf.nn.depthwise_conv2d(
        ch_nhwc, _SOBEL_KX_C1, strides=[1, 1, 1, 1], padding="SAME"
    )
    gy = tf.nn.depthwise_conv2d(
        ch_nhwc, _SOBEL_KY_C1, strides=[1, 1, 1, 1], padding="SAME"
    )
    mag = tf.sqrt(gx * gx + gy * gy + 1e-9)
    return tf.transpose(mag, [0, 3, 1, 2])


def _norm01(x):
    mean = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
    std = tf.math.reduce_std(x, axis=[2, 3], keepdims=True)
    return (x - mean) / (std + 1e-6)


COORD_IN_D = False

def _coord_maps_like(x):
    """Return NCHW y/x coordinate planes in [-1, 1] matching ``x``."""
    height = tf.shape(x)[2]
    width = tf.shape(x)[3]
    y = tf.linspace(-1.0, 1.0, height)
    xlin = tf.linspace(-1.0, 1.0, width)
    yy, xx = tf.meshgrid(y, xlin, indexing="ij")
    base = tf.stack([yy, xx], axis=0)[None, ...]  # (1, 2, H, W)
    return tf.tile(tf.cast(base, x.dtype), [tf.shape(x)[0], 1, 1, 1])


def _resize_like_nchw(x: tf.Tensor, ref: tf.Tensor, method: str = "nearest") -> tf.Tensor:
    """Resize x (N,C,H,W) to match ref's spatial size, if needed."""
    x_h = tf.shape(x)[2]
    x_w = tf.shape(x)[3]
    r_h = tf.shape(ref)[2]
    r_w = tf.shape(ref)[3]
    if x.shape.rank == 4 and ref.shape.rank == 4:
        # Fast-path when static sizes match
        if (
            x.shape[2] is not None
            and ref.shape[2] is not None
            and x.shape[2] == ref.shape[2]
            and x.shape[3] is not None
            and ref.shape[3] is not None
            and x.shape[3] == ref.shape[3]
        ):
            return x

    def _do_resize():
        # Keep the same centered resize convention used by generator upsampling.
        return resize_nchw(x, [r_h, r_w], method=method)
    return tf.cond(tf.logical_and(tf.equal(x_h, r_h), tf.equal(x_w, r_w)), lambda: x, _do_resize)


def resize_nchw(x: tf.Tensor, size_hw, method: str = "nearest") -> tf.Tensor:
    """Resize an NCHW tensor with a centered (SR-friendly) convention.

    The supplied implementation uses:
      align_corners=False, half_pixel_centers=False
    and applies the same convention throughout the conditioning and loss paths
    to avoid inconsistent LR-to-HR alignment.
    """
    x_nhwc = tf.transpose(x, [0, 2, 3, 1])
    size = tf.cast(tf.stack([size_hw[0], size_hw[1]]), tf.int32)
    m = method.lower()

    if m in ("bilinear", "linear"):
        try:
            y = tf.raw_ops.ResizeBilinear(
                images=x_nhwc, size=size, align_corners=False, half_pixel_centers=False
            )
        except Exception:
            y = tf.image.resize(x_nhwc, size_hw, method="bilinear")
    elif m == "nearest":
        try:
            y = tf.raw_ops.ResizeNearestNeighbor(
                images=x_nhwc, size=size, align_corners=False, half_pixel_centers=False
            )
        except Exception:
            y = tf.image.resize(x_nhwc, size_hw, method="nearest")
    else:
        y = tf.image.resize(x_nhwc, size_hw, method=method)

    return tf.transpose(y, [0, 3, 1, 2])

def _make_d_input(x, lr, edge_lambda, p_drop, training):
    # If LR is genuinely low-res (e.g. 96×96) but x is HR (e.g. 192×192),
    # upsample LR for conditioning/residual so shapes match.
    lr = _resize_like_nchw(lr, x, method="bilinear")
    # Residual and edge map (channels_first: N,C,H,W)
    r = tf.stop_gradient(x) - lr  # detach residual branch from G
    gain = EDGE_TANH_GAIN
    edge_lambda = tf.cast(edge_lambda, x.dtype)
    p_drop = tf.cast(p_drop, x.dtype)

    # `training` is intentionally kept as a Python bool here.
    # All current callers pass True/False explicitly, which lets the
    # solo-channel branch be selected at tf.function trace time.
    training_flag = bool(training)

    # --- "solo" per-channel passes into D ------------------------------------
    solo_p = float(globals().get("D_SOLO_CHAN_P", 0.0))
    if training_flag and (solo_p > 0.0):
        B = tf.shape(x)[0]
        do_solo = tf.random.uniform([B, 1, 1, 1]) < solo_p          # per-sample coin
        ci = tf.random.uniform([B], 0, 3, dtype=tf.int32)           # per-sample channel id
        mask = tf.one_hot(ci, depth=3, dtype=x.dtype)               # [B,3]
        mask = tf.reshape(mask, [B, 3, 1, 1])

        # apply mask only where do_solo==True, else pass-through
        solo_mask = tf.where(do_solo, mask, tf.ones_like(mask))     # [B,3,1,1]
        x  = x  * solo_mask
        lr = lr * solo_mask
        r  = x - lr  # recompute residual from masked streams

    # edge from (possibly masked) residual
    ew = tf.tanh(gain * _norm01(_sobel_mag_channel(r, 2))) * edge_lambda
    eT = tf.tanh(gain * _norm01(_sobel_mag_channel(r, 0))) * edge_lambda
    ev = tf.tanh(gain * _norm01(_sobel_mag_channel(r, 1))) * edge_lambda

    # --- per-sample conditioning dropout with "at least one" guard ----------
    def do_dropout():
        B = tf.shape(x)[0]
        keep_r  = tf.cast(tf.random.uniform([B,1,1,1]) >= p_drop, x.dtype)
        keep_lr = tf.cast(tf.random.uniform([B,1,1,1]) >= p_drop, x.dtype)
        both_zero = tf.cast(tf.equal(keep_r + keep_lr, 0.0), x.dtype)
        keep_lr = tf.maximum(keep_lr, both_zero)

        r2  = r  * keep_r
        lr2 = lr * keep_lr
        return (lr2,
                r2,
                ew * keep_r,
                eT * keep_r,
                ev * keep_r)

    def no_dropout():
        return (lr, r, ew, eT, ev)

    drop_pred = tf.logical_and(
        tf.constant(training_flag, dtype=tf.bool),
        p_drop > 0.0,
    )

    lr, r, ew, eT, ev = tf.cond(
        drop_pred,
        do_dropout,
        no_dropout,
    )
    # --- pack inputs for D ---
    edges3 = tf.concat([eT, ev, ew], axis=1)           # 3 learned edge planes
    # x(3) + lr(3) + r(3) + edges(3) = 12 channels → D.init_build(...,12,...)
    inp = tf.concat([x, lr, r, edges3], axis=1)
    if COORD_IN_D:
        coords = _coord_maps_like(inp)
        inp = tf.concat([inp, coords], axis=1)  # -> 14 channels when enabled
    return inp

# Logit post-processing
def _clamp_logits(z, limit=30.0):
    return tf.clip_by_value(z, -limit, limit)

def _sanitize_logits(z):
    return tf.where(tf.math.is_finite(z), z, tf.zeros_like(z))

# --------------------------- GAN losses -------------------------------------------
def bce_d_loss(real_logits, fake_logits, real_t=0.8, fake_t=0.2, scale=1.5):
    r = _sanitize_logits(real_logits) * scale
    f = _sanitize_logits(fake_logits) * scale
    d_real = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(r) * real_t,
        logits=r,
    )
    d_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(f) + fake_t,
        logits=f,
    )
    return tf.reduce_mean(d_real + d_fake)

def bce_g_loss(fake_logits, scale=1.5):
    f = _sanitize_logits(fake_logits) * scale
    g_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f), logits=f)
    return tf.reduce_mean(g_fake)

def rpsgan_d_loss(real_logits, fake_logits, diff_limit=30.0):
    r = _sanitize_logits(real_logits)
    f = _sanitize_logits(fake_logits)
    diff = _clamp_logits(r - f, diff_limit)          # r>f desired
    # logistic (non-saturating) with softplus on logits
    return tf.reduce_mean(tf.nn.softplus(-diff))

def rpsgan_g_loss(real_logits, fake_logits, diff_limit=30.0):
    r = _sanitize_logits(real_logits)
    f = _sanitize_logits(fake_logits)
    diff = _clamp_logits(r - f, diff_limit)          # G wants f>r ⇒ minimize softplus(r-f)
    return tf.reduce_mean(tf.nn.softplus(diff))

# -------- RA-SGAN (AVERAGE) with softplus + optional EMA running means --------------
def rasgan_d_loss(real_logits, fake_logits, ema_real=None, ema_fake=None, mom=0.9):
    # Use logits (no sigmoid in D). Softplus is numerically stable.
    r = _sanitize_logits(real_logits)
    f = _sanitize_logits(fake_logits)
    # batch means
    r_mean = tf.reduce_mean(r)
    f_mean = tf.reduce_mean(f)

    # optional running means for RA-AVERAGE (smoother switch)
    if ema_real is not None:
        ema_real.assign(mom * ema_real + (1.0 - mom) * tf.stop_gradient(r_mean))
        r_bar = tf.stop_gradient(ema_real)
    else:
        r_bar = r_mean
    if ema_fake is not None:
        ema_fake.assign(mom * ema_fake + (1.0 - mom) * tf.stop_gradient(f_mean))
        f_bar = tf.stop_gradient(ema_fake)
    else:
        f_bar = f_mean
    # D wants real > fake
    loss = (
        tf.reduce_mean(tf.nn.softplus(-(r - f_bar)))
        + tf.reduce_mean(tf.nn.softplus(f - r_bar))
    )
    return loss


def rasgan_g_loss(real_logits, fake_logits, ema_real=None, ema_fake=None, mom=0.9):
    r = _sanitize_logits(real_logits)
    f = _sanitize_logits(fake_logits)
    r_mean = tf.reduce_mean(r)
    f_mean = tf.reduce_mean(f)

    if ema_real is not None:
        ema_real.assign(mom * ema_real + (1.0 - mom) * tf.stop_gradient(r_mean))
        r_bar = tf.stop_gradient(ema_real)
    else:
        r_bar = r_mean
    if ema_fake is not None:
        ema_fake.assign(mom * ema_fake + (1.0 - mom) * tf.stop_gradient(f_mean))
        f_bar = tf.stop_gradient(ema_fake)
    else:
        f_bar = f_mean
    # G wants fake > real
    loss = (
        tf.reduce_mean(tf.nn.softplus(-(f - r_bar)))
        + tf.reduce_mean(tf.nn.softplus(r - f_bar))
    )
    return loss

# Instance noise

def add_instance_noise(x, std):
    std = tf.cast(std, x.dtype)
    return tf.cond(
        std > 0,
        lambda: x + tf.random.normal(tf.shape(x), stddev=std, dtype=x.dtype),
        lambda: x
    )

# R1 on REAL (mean patch logit stabilizes scale)
def r1_penalty_safe(D, real, clip_per_sample_norm=50.0):
    with tf.GradientTape() as t:
        t.watch(real)
        logits_map = D(real)
        s_per = tf.reduce_mean(logits_map, axis=[1,2,3])
        s = tf.reduce_sum(s_per)
    grads = t.gradient(s, real)
    if grads is None:
        return tf.constant(0.0, dtype=real.dtype)
    grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
    g_flat = tf.reshape(grads, [tf.shape(grads)[0], -1])
    norms = tf.sqrt(tf.reduce_sum(g_flat * g_flat, axis=1) + 1e-12)
    norms = tf.minimum(norms, clip_per_sample_norm)
    return tf.reduce_mean(norms * norms)

# ---- NCHW <-> NHWC helpers ----
def nchw_to_nhwc(x):
    return tf.transpose(x, (0, 2, 3, 1))


def nhwc_to_nchw(x):
    return tf.transpose(x, (0, 3, 1, 2))

# ---- Low-pass penalty on delta using avg-pool (NHWC internally) ----
def lowpass_l1_on_delta(delta_nchw, k=4):
    d = nchw_to_nhwc(delta_nchw)
    # SAME padding, stride=1 - true blur (not downsample)
    lp = tf.nn.avg_pool2d(d, ksize=k, strides=1, padding="SAME")
    return tf.reduce_mean(tf.abs(lp))

# ====================================== Loss Wrappers ====================================== #
