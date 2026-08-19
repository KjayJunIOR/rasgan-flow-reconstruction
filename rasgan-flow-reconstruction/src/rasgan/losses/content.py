from __future__ import annotations

from ..env import tf
from typing import Optional, Sequence, Tuple

import numpy as np
from ..config import CHARB_EPS, EDGE_TANH_GAIN


def _to_f32(x):
    return tf.cast(x, tf.float32) if x.dtype != tf.float32 else x


def _finite(x, name: str):
    # zero-out non-finite contributions instead of crashing
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    return x


def charbonnier(x, y, eps: float = CHARB_EPS):
    x = _to_f32(x); y = _to_f32(y)
    diff = x - y
    return tf.reduce_mean(tf.sqrt(diff * diff + eps * eps))


def charbonnier_balanced(x, y, hr_inv, eps: float = CHARB_EPS):
    """Channel-balanced Charbonnier using dataset HR std inverse (broadcastable 1,C,1,1)."""
    x = _to_f32(x); y = _to_f32(y)
    diff = (x - y) * hr_inv
    return tf.reduce_mean(tf.sqrt(diff * diff + eps * eps))


# Ported verbatim from the original script, but made explicit + standalone.
def charbonnier_balanced_weighted(
    pred,
    gt,
    weights: Optional[Sequence[float]] = None,
    eps: float = CHARB_EPS,
    alpha: float = 0.5,
    use_channel_balance: bool = True,
    w_clip: Tuple[float, float] = (0.5, 2.0),
    use_residual_scale: bool = False,
):
    """
    Balanced Charbonnier:
      - Per-channel robust scale s_c is estimated on the fly (stop_grad).
      - Residuals are normalized by s_c, and channels are reweighted by ~1/s_c.
      - w_clip prevents over/under-weighting a channel.
    """
    pred = _to_f32(pred); gt = _to_f32(gt)
    r = pred - gt  # [N,C,H,W]

    if use_channel_balance:
        if use_residual_scale:
            s = tf.reduce_mean(tf.abs(r), axis=[0, 2, 3], keepdims=True)  # [1,C,1,1]
        else:
            s = tf.reduce_mean(tf.abs(gt), axis=[0, 2, 3], keepdims=True) # [1,C,1,1]
        s = tf.stop_gradient(s + 1e-6)
        r_n = r / s

        # reweight ~ 1/s, normalized to mean 1
        w_bal = 1.0 / s
        w_bal = w_bal / tf.reduce_mean(w_bal)
        w_bal = tf.clip_by_value(w_bal, w_clip[0], w_clip[1])
        w_bal = tf.stop_gradient(w_bal)  # don't backprop through weights
    else:
        r_n = r
        w_bal = 1.0

    if weights is not None:
        w_user = tf.reshape(tf.constant(list(weights), dtype=tf.float32), [1, -1, 1, 1])
    else:
        w_user = 1.0

    diff = r_n * w_bal * w_user
    out = tf.sqrt(diff * diff + eps * eps)
    return tf.reduce_mean(out)


def _grad_xy(x):
    # x: (N,C,H,W)
    x = _to_f32(x)
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # pad to keep same shape
    gx = tf.pad(gx, [[0,0],[0,0],[0,0],[0,1]])
    gy = tf.pad(gy, [[0,0],[0,0],[0,1],[0,0]])
    return gx, gy


def _grad_mag(x):
    gx, gy = _grad_xy(x)
    return tf.sqrt(gx*gx + gy*gy + 1e-9)


def tv_loss(x):
    x = _to_f32(x)
    gx, gy = _grad_xy(x)
    return tf.reduce_mean(tf.abs(gx)) + tf.reduce_mean(tf.abs(gy))


def denorm_hr_np(x: np.ndarray, hr_mean_np: np.ndarray, hr_std_np: np.ndarray) -> np.ndarray:
    """Denormalize HR numpy batches using broadcastable (1,C,1,1) arrays."""
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    out = x * hr_std_np + hr_mean_np
    # Be defensive: stats can contain NaNs early if computed from a tiny subset.
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def mae_rmse_psnr_denorm(pred_nchw: np.ndarray, gt_nchw: np.ndarray, hr_mean_np: np.ndarray, hr_std_np: np.ndarray):
    """Return MAE, RMSE, PSNR on denormalized arrays."""
    p = denorm_hr_np(pred_nchw, hr_mean_np, hr_std_np)
    g = denorm_hr_np(gt_nchw, hr_mean_np, hr_std_np)
    # Be defensive: early training may produce NaNs/Infs from unstable activations.
    # Clamp them away so validation metrics remain informative.
    p = np.nan_to_num(p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    diff = p - g
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff * diff)))
    # avoid divide by zero
    mse = max(1e-12, rmse * rmse)
    peak = float(np.percentile(np.abs(g), 99.9))
    peak = max(1e-6, peak)
    psnr = float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))
    return mae, rmse, psnr

def _edge_align_penalty(dx_a, dy_a, dx_b, dy_b, edge_gain=EDGE_TANH_GAIN, eps=1e-6):
    """
    Penalize alignment of two edge fields (A vs B). We compute cos^2(theta)
    between the gradient vectors and gently weight by 'edge presence'
    using tanh(magnitude) so flat areas contribute ~0.

    Inputs are (N,C,H,W) tensors for dx, dy of each field.
    Returns a scalar tf.reduce_mean(...) penalty.
    """
    # magnitudes
    mag_a = tf.sqrt(dx_a * dx_a + dy_a * dy_a + eps)
    mag_b = tf.sqrt(dx_b * dx_b + dy_b * dy_b + eps)

    # cosine similarity in [-1, 1]
    dot   = dx_a * dx_b + dy_a * dy_b
    denom = (mag_a * mag_b + eps)
    cos   = dot / denom

    # focus penalty where both have edges (range ~0..1)
    w = tf.tanh(edge_gain * mag_a) * tf.tanh(edge_gain * mag_b)

    # decorrelate directions (sign-agnostic: cos^2)
    return tf.reduce_mean(w * (cos * cos))


# ---------- Mean absolute off-diagonal correlation (diagnostic) ----------


def _corr_abs(feat, eps=1e-6):
    """Return mean absolute off-diagonal correlation between channels.

    This is used as a validation diagnostic to detect channel collapse /
    redundancy. It computes correlation matrices per-sample across spatial
    positions (treating each channel as a vector), then averages the absolute
    value of off-diagonal entries.

    Args:
        feat: Tensor of shape [N, C, H, W] (NCHW).
        eps: Numerical stability term.

    Returns:
        Scalar tensor (float32).
    """
    x = tf.cast(feat, tf.float32)
    # Flatten spatial dims: [N, C, S]
    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
    # Center per channel
    x = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    # Covariance: [N, C, C]
    s = tf.cast(tf.shape(x)[-1], tf.float32)
    cov = tf.matmul(x, x, transpose_b=True) / tf.maximum(s - 1.0, 1.0)
    # Correlation
    var = tf.linalg.diag_part(cov)
    std = tf.sqrt(tf.maximum(var, 0.0) + eps)
    denom = std[:, :, None] * std[:, None, :] + eps
    corr = cov / denom

    # Zero diagonal and average absolute off-diagonal entries
    c = tf.shape(corr)[-1]
    mask = tf.ones([c, c], dtype=tf.float32) - tf.eye(c, dtype=tf.float32)
    corr_off = corr * mask[None, :, :]
    abs_sum = tf.reduce_sum(tf.abs(corr_off), axis=[1, 2])
    denom_off = tf.cast(c * (c - 1), tf.float32) + eps
    return tf.reduce_mean(abs_sum / denom_off)

#[ADD] ---------- Off-diagonal covariance penalty ----------


def _offdiag_cov_penalty(feat, as_correlation=True, eps=1e-6):
    """
    Batched off-diagonal penalty on covariance/correlation between channels.

    feat: (N, C, H, W) or (N, C, S) tensor
    Returns a scalar mean over batch of sumsq(off-diagonals).
    """
    # Flatten spatial dims → (N, C, S)
    feat = tf.convert_to_tensor(feat)
    x = feat
    x_shape = tf.shape(x)
    # (N,C,H,W) → (N,C,S)
    x = tf.reshape(x, [x_shape[0], x_shape[1], -1])

    x = tf.cast(x, tf.float32)

    # Center across samples S
    x = x - tf.reduce_mean(x, axis=2, keepdims=True)

    S = tf.cast(tf.shape(x)[2], tf.float32)
    cov = tf.matmul(x, x, transpose_b=True) / (S + eps)   # (N,C,C)

    if as_correlation:
        # Normalize to correlation to avoid scale issues
        var = tf.reduce_sum(tf.square(x), axis=2) / (S + eps)   # (N,C)
        std = tf.sqrt(tf.maximum(var, 0.0) + eps)               # (N,C)
        # Outer product of std: (N,C,C)
        std_outer = tf.expand_dims(std, 2) * tf.expand_dims(std, 1)
        mat = cov / (std_outer + eps)                           # correlation matrix
    else:
        mat = cov

    # Zero the diagonal in a batched-safe way
    mat_off = tf.linalg.set_diag(mat, tf.zeros_like(tf.linalg.diag_part(mat)))

    # Penalize off-diagonals (squared)
    return tf.reduce_mean(tf.square(mat_off))

def lowpass_l1_on_delta(delta_p, k: int = 5):
    """Low-pass penalty on delta_p.
    delta_p expected in NCHW. Uses average pooling (SAME) to extract low-frequency component.
    Returns mean(|lowpass(delta_p)|).
    """
    x = _to_f32(delta_p)
    k = int(k)
    if k <= 1:
        return tf.reduce_mean(tf.abs(x))
    # TF pooling expects NHWC
    x_nhwc = tf.transpose(x, [0, 2, 3, 1])
    low_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize=k, strides=1, padding="SAME")
    low = tf.transpose(low_nhwc, [0, 3, 1, 2])
    return tf.reduce_mean(tf.abs(low))
