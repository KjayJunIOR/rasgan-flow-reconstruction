"""Validation helpers.

Validation should **reuse the exact same loss definitions** as training.

During training, the generator is often wrapped in a `WithLoss_*` module from
`losses/wrappers.py`. Those wrappers expose `eval_terms(...)`, which returns a
deterministic, per-term breakdown (plus `total`) suitable for validation.

This file therefore focuses on:
  * basic reconstruction metrics (MAE / denorm MAE / RMSE / PSNR)
  * delegating *all* training-loss computation to `net_g.eval_terms(...)`
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .env import tf
from .losses.content import mae_rmse_psnr_denorm
from .losses.spectrum import energy_spectrum2d

def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert tensor/np/scalar to python float; replace NaN/Inf with default."""
    try:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_div(num: Any, den: int, default: float = 0.0) -> float:
    if den is None or den == 0:
        return float(default)
    try:
        v = float(num) / float(den)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def _grad_mag_nchw(x: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    """Simple gradient magnitude for NCHW tensors."""
    # x: [N,C,H,W]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = tf.pad(dx, [[0, 0], [0, 0], [0, 0], [0, 1]])
    dy = tf.pad(dy, [[0, 0], [0, 0], [0, 1], [0, 0]])
    mag = tf.sqrt(dx * dx + dy * dy + eps)
    return mag

def _edge_bleed_metric(sr: tf.Tensor, hr: tf.Tensor) -> tf.Tensor:
    """Edge 'bleed' metric: extra edges in SR where HR is smooth.

    This is *not* a training loss; it's a sanity-check metric.
    We measure gradient magnitude in SR vs HR and only penalize
    positive excess in SR, weighted to focus on regions where HR
    has little/no edge.
    """
    mag_sr = tf.reduce_mean(_grad_mag_nchw(sr), axis=1, keepdims=True)
    mag_hr = tf.reduce_mean(_grad_mag_nchw(hr), axis=1, keepdims=True)
    # weight close to 1 in smooth regions, close to 0 at strong HR edges
    w = 1.0 - tf.tanh(5.0 * mag_hr)
    bleed = tf.reduce_mean(w * tf.nn.relu(mag_sr - mag_hr))
    return tf.where(tf.math.is_finite(bleed), bleed, tf.zeros_like(bleed))

def _as_mean_std_tf(mean_np: Optional[np.ndarray], std_np: Optional[np.ndarray]):
    """(mean_tf, std_tf) shaped [1,C,1,1] or (None, None) if unavailable."""
    if mean_np is None or std_np is None:
        return None, None
    try:
        m = np.asarray(mean_np, dtype=np.float32).reshape(1, -1, 1, 1)
        s = np.asarray(std_np, dtype=np.float32).reshape(1, -1, 1, 1)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
        s = np.where(np.abs(s) < 1e-12, 1.0, s)
        return tf.constant(m, tf.float32), tf.constant(s, tf.float32)
    except Exception:
        return None, None


def _validate_epoch(
    val_loader,
    net_g,
    hr_mean_np: np.ndarray,
    hr_std_np: np.ndarray,
    lr_mean_np: Optional[np.ndarray] = None,
    lr_std_np: Optional[np.ndarray] = None,
    *,
    phys_params: Optional[Any] = None,
    pod_params: Optional[Any] = None,
    score_weights: Optional[Any] = None,
    # Legacy/unused knobs kept for API compatibility with older training loops.
    net_d=None,
    M_STACK=None,
    WEIGHTS=None,
    DEN_BINS=None,
    LOW_BINS=None,
    LOW_MASK=None,
):
    """Run one validation epoch.

    Notes
    -----
    * If `net_g` exposes `eval_terms`, we use it to compute training-loss terms.
    * `score_weights` is used only as a stage hint:
        - None  => init stage: `val_score = -MAE` (unchanged behavior)
        - else  => adv stage:  `val_score = -val_total` from wrapper terms
    """

    # Underlying generator for reconstruction metrics.
    gen = getattr(net_g, "G", net_g)

    # Optionally configure wrapper (if provided) with physics/POD params.
    if hasattr(net_g, "set_physics") and phys_params is not None:
        mean_tf, std_tf = _as_mean_std_tf(hr_mean_np, hr_std_np)
        if mean_tf is not None and std_tf is not None:
            try:
                net_g.set_physics(mean_tf, std_tf, phys_params)
            except Exception:
                pass

    if hasattr(net_g, "set_pod") and pod_params is not None:
        lr_mean_tf, lr_std_tf = _as_mean_std_tf(lr_mean_np, lr_std_np)
        if lr_mean_tf is not None and lr_std_tf is not None:
            try:
                net_g.set_pod(lr_mean_tf, lr_std_tf, pod_params)
            except Exception:
                pass

    use_wrapper_terms = hasattr(net_g, "eval_terms")

    n_batches = 0
    agg_terms: Dict[str, float] = {}
    mae_denorm_vals = []
    rmse_denorm_vals = []
    psnr_denorm_vals = []
    mae_sum = 0.0; mae_ch_sum = None
    edge_val = 0.0
    spec_rh = 0.0

    # Make safe numpy stats for denorm metrics.
    try:
        hr_mean_np_safe = np.nan_to_num(np.asarray(hr_mean_np, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        hr_std_np_safe = np.nan_to_num(np.asarray(hr_std_np, dtype=np.float32), nan=1.0, posinf=1.0, neginf=1.0)
        hr_std_np_safe = np.where(np.abs(hr_std_np_safe) < 1e-8, 1.0, hr_std_np_safe)
    except Exception:
        hr_mean_np_safe = None
        hr_std_np_safe = None

    # Tensor versions (NCHW) for stable in-graph denormalization during validation.
    mean_tf = std_tf = None
    if hr_mean_np_safe is not None and hr_std_np_safe is not None:
        try:
            m = np.asarray(hr_mean_np_safe, dtype=np.float32).reshape(1, -1, 1, 1)
            s = np.asarray(hr_std_np_safe, dtype=np.float32).reshape(1, -1, 1, 1)
            # Avoid divide-by-zero / exploding denorm when std has zeros.
            s = np.where(np.isfinite(s) & (np.abs(s) > 1e-12), s, 1.0)
            mean_tf = tf.constant(m, dtype=tf.float32)
            std_tf = tf.constant(s, dtype=tf.float32)
        except Exception:
            mean_tf = None
            std_tf = None

    for batch in val_loader:
        # Support optional POD coeffs in validation loader: (lr, hr, tim_u, tim_v, tim_p)
        if isinstance(batch, (tuple, list)) and len(batch) == 5:
            lr_b, hr_b, tim_u, tim_v, tim_p = batch
            pod_coeffs = (tim_u, tim_v, tim_p)
        else:
            lr_b, hr_b = batch
            pod_coeffs = None

        # Forward generator for reconstruction metrics.
        try:
            sr_b = gen(lr_b, training=False)
        except TypeError:
            # Some generators accept (lr, hr, training=...)
            sr_b = gen(lr_b, hr_b, training=False)

        # Safety: keep validation robust even if model produces NaNs/Infs.
        sr_b = tf.where(tf.math.is_finite(sr_b), sr_b, tf.zeros_like(sr_b))
        hr_b = tf.where(tf.math.is_finite(hr_b), hr_b, tf.zeros_like(hr_b))

        # Optional denorm tensors (used by metrics like edge_bleed).
        if mean_tf is not None and std_tf is not None:
            sr_den_tf = sr_b * std_tf + mean_tf
            hr_den_tf = hr_b * std_tf + mean_tf
        else:
            sr_den_tf = sr_b
            hr_den_tf = hr_b

        if (M_STACK is not None and WEIGHTS is not None and DEN_BINS is not None):
            # Edge-bleed metric (extra edges in SR where HR is smooth).
            bleed_tf = _edge_bleed_metric(sr_den_tf, hr_den_tf)
            edge_val += _safe_float(bleed_tf)
            Ey = energy_spectrum2d(sr_b, M_STACK, WEIGHTS, DEN_BINS)
            Et = energy_spectrum2d(hr_b, M_STACK, WEIGHTS, DEN_BINS)

            # Spectrum relative error split into low/high k bands (for logging).
            # Uses LOW_MASK when available; otherwise splits at 25% of bins.
            try:
                relerr = tf.abs(Ey - Et) / (tf.abs(Et) + 1e-12)
                relerr_flat = tf.reshape(relerr, [-1, tf.shape(relerr)[-1]])
                if LOW_MASK is not None:
                    low_mask = tf.convert_to_tensor(LOW_MASK, dtype=tf.bool)
                else:
                    K = tf.shape(relerr_flat)[-1]
                    low_mask = tf.range(K) < tf.cast(K // 4, tf.int32)
                high_mask = tf.logical_not(low_mask)
                high_vals = tf.boolean_mask(relerr_flat, high_mask, axis=1)
                high_re = tf.reduce_mean(high_vals)
                spec_rh += _safe_float(high_re)
            except Exception:
                pass

        mae_tf = tf.reduce_mean(tf.abs(sr_b - hr_b))
        mae_tf = tf.where(tf.math.is_finite(mae_tf), mae_tf, tf.constant(0.0, dtype=mae_tf.dtype))
        mae = _safe_float(mae_tf)
        mae_sum += mae

        # --- per-channel MAE (NCHW) ---
        # reduce over N,H,W -> keep C
        mae_ch_tf = tf.reduce_mean(tf.abs(sr_b - hr_b), axis=[0, 2, 3])  # [C]
        mae_ch_tf = tf.where(tf.math.is_finite(mae_ch_tf), mae_ch_tf, tf.zeros_like(mae_ch_tf))
        mae_ch = mae_ch_tf.numpy().astype(np.float64)  # [C]
        if mae_ch_sum is None:
            mae_ch_sum = np.zeros_like(mae_ch)
        mae_ch_sum += mae_ch

        # Denormalized metrics for logging/sanity checking.
        if hr_mean_np_safe is not None and hr_std_np_safe is not None:
            try:
                mae_d, rmse_d, psnr_d = mae_rmse_psnr_denorm(
                    sr_b.numpy(), hr_b.numpy(), hr_mean_np_safe, hr_std_np_safe
                )
            except Exception:
                mae_d, rmse_d, psnr_d = mae, float("nan"), float("nan")
        else:
            mae_d, rmse_d, psnr_d = mae, float("nan"), float("nan")

        mae_denorm_vals.append(float(mae_d))
        rmse_denorm_vals.append(float(rmse_d))
        psnr_denorm_vals.append(float(psnr_d))

        # Wrapper loss breakdown (exact training terms, deterministic).
        if use_wrapper_terms:
            try:
                terms = net_g.eval_terms(lr_b, hr_b, pod_coeffs=pod_coeffs)
                for k, v in terms.items():
                    agg_terms[k] = agg_terms.get(k, 0.0) + _safe_float(v)
            except Exception:
                # If wrapper breakdown fails, keep going with metrics only.
                use_wrapper_terms = False

        n_batches += 1

    if n_batches == 0:
        return {
            "mae": float("nan"),
            "mae_denorm": float("nan"),
            "rmse_denorm": float("nan"),
            "psnr": float("nan"),
            "val_score": float("nan"),
            "val_mae": float("nan"),
            "val_mae_denorm": float("nan"),
            "val_psnr": float("nan"),
        }

    out: Dict[str, float] = {}
    out["mae"] = _safe_div(mae_sum, n_batches, default=0.0)

    # Per-channel MAE averages
    if mae_ch_sum is not None:
        mae_ch_avg = mae_ch_sum / max(1, n_batches)  # [C]
        # If you know your channel order is (p, v, w):
        names = ["p", "v", "w"]
        for i in range(int(mae_ch_avg.shape[0])):
            key = names[i] if i < len(names) else f"ch{i}"
            out[f"mae_{key}"] = float(mae_ch_avg[i])

    out["mae_denorm"] = float(np.mean(mae_denorm_vals)) if mae_denorm_vals else float("nan")
    out["rmse_denorm"] = float(np.mean(rmse_denorm_vals)) if rmse_denorm_vals else float("nan")
    out["psnr"] = float(np.mean(psnr_denorm_vals)) if psnr_denorm_vals else float("nan")

    # Training logger aliases.
    out["val_mae"] = out["mae"]
    out["val_mae_denorm"] = out["mae_denorm"]
    out["val_psnr"] = out["psnr"]

    # Training logger aliases for per-channel
    for k in list(out.keys()):
        if k.startswith("mae_") and k not in ("mae", "mae_denorm"):
            out[f"val_{k}"] = out[k]

    out["edge_bleed"] = _safe_div(edge_val, n_batches, default=0.0)
    out["spec_relerr_high"] = _safe_div(spec_rh, n_batches, default=0.0)

    # Wrapper terms -> val_* logging keys.
    if agg_terms:
        for k, v in agg_terms.items():
            out[f"val_{k}"] = _safe_div(v, n_batches, default=0.0)
        out["val_total_loss"] = float(out.get("val_total", 0.0))

    # Stage-dependent score (keeps init behavior unchanged).
    if score_weights is None:
        out["val_score"] = -out["mae"]
    else:
        # Prefer exact training objective when available; fall back to MAE.
        if "val_total" in out:
            out["val_score"] = -out["val_total"]
        else:
            out["val_score"] = -out["mae"]

    return out
