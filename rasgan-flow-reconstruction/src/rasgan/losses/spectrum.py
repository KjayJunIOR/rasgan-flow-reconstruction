from __future__ import annotations

from ..env import tf
import math
from typing import Tuple

import numpy as np
from ..config import LOW_BINS, ESPEC_ALPHA

SPEC_BORDER = 4
_SPEC_BAKED = None      # do not touch directly

def set_spec_border(b: int):
    """Freeze the border used for all spectral ops/masks during this run."""
    global _SPEC_BAKED
    _SPEC_BAKED = int(b)

def _crop_border(x, b=SPEC_BORDER):
    if b <= 0: return x
    return x[:, :, b:-b, b:-b]

# --- GAN-logit crop (use baked spectral border) ---
def _crop_logits(z, b=None):
    b = int(_SPEC_BAKED if b is None else b)
    if b <= 0:
        return z
    return z[:, :, b:-b, b:-b]

# ====================== FFT & ENERGY SPECTRUM (vectorized) ===================== #
# --- Graph-safe Hann window (no Python dict, works with dynamic shapes) ---

def _make_hann2d(H, W, dtype=tf.float32, periodic=True):
    """Return a [1,1,H,W] Hann window built in-graph."""
    H = tf.cast(H, tf.int32)
    W = tf.cast(W, tf.int32)
    wH = tf.signal.hann_window(H, periodic=periodic, dtype=dtype)   # [H]
    wW = tf.signal.hann_window(W, periodic=periodic, dtype=dtype)   # [W]
    win2d = tf.tensordot(wH, wW, axes=0)                            # [H,W]
    return tf.reshape(win2d, [1, 1, H, W])

def _apply_hann(x):
    """x: [N,C,H,W] float tensor. Multiplies by Hann[1,1,H,W]."""
    s = tf.shape(x)
    w = _make_hann2d(s[2], s[3], dtype=x.dtype)
    return x * w

def log_amp_fft(x, eps=1e-6, border=None):
    """
    x: (N, C, H, W) float32
    Crops by `border` pixels on all sides (H and W), applies Hann, then rFFT2d.
    Returns (N, C, Hc, Wrc) where Hc = H - 2*b, Wrc = (W - 2*b)//2 + 1
    """
    # choose border
    if border is None:
        border = _SPEC_BAKED if _SPEC_BAKED is not None else 0

    x = tf.cast(x, tf.float32)
    if border:
        x = _crop_border(x, b=border)  # centralize the crop here

    # Hann window AFTER crop
    x = _apply_hann(x)

    s = tf.shape(x)
    N, C, H, W = s[0], s[1], s[2], s[3]

    x_nc_hw = tf.reshape(x, [N * C, H, W])
    U = tf.signal.rfft2d(x_nc_hw)  # (N·C, H, W_r)
    mag = tf.abs(U)
    mag = tf.where(tf.math.is_finite(mag), mag, tf.zeros_like(mag))

    # hard bound extremes before log compression (prevents rare huge spikes)
    mag = tf.clip_by_value(mag, 0.0, 1e3)  # 1e3 is a safe starting point; tune if needed

    # smoother near 0 than log(mag)
    la_nc = tf.math.log1p(mag)  # in [0, log1p(1e3)] approx [0, 6.9]
    la = tf.reshape(la_nc, [N, C, H, tf.shape(la_nc)[2]])  # (N, C, H, W_r)
    la = tf.where(tf.math.is_finite(la), la, tf.zeros_like(la))
    return la

def safe_energy_spectrum(E, eps=1e-6):
    E = tf.cast(E, tf.float32)
    E = tf.where(tf.math.is_finite(E), E, tf.zeros_like(E))
    return tf.maximum(E, eps)

def build_radial_masks(H, W):
    """Given real-space H,W AFTER crop, build masks for rFFT grid (H, W_r)."""
    H = int(H); W = int(W)
    W_r = W // 2 + 1
    # radial coordinates on rFFT grid
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.rfftfreq(W) * W
    KY, KX = np.meshgrid(ky, kx, indexing='ij')  # (H, W_r)
    R = np.sqrt(KX**2 + KY**2)

    nbins = int(min(H, W_r) // 2)
    edges = np.linspace(0.0, R.max() + 1e-6, nbins + 1).astype(np.float32)

    masks = []
    for i in range(nbins):
        m = ((R >= edges[i]) & (R < edges[i+1])).astype(np.float32)  # (H, W_r)
        masks.append(m)
    M_STACK = np.stack(masks, axis=-1)  # (H, W_r, nbins)

    # gentle high-k emphasis (tune ESPEC_ALPHA as before)
    weights = 1.0 + ESPEC_ALPHA * (np.arange(nbins, dtype=np.float32) / max(1, nbins-1))

    # (broadcast shapes used by the loss)
    M_STACK_tf = tf.constant(M_STACK[None, None, ...])          # (1,1,H,W_r,nbins)
    WEIGHTS_tf = tf.constant(weights)                           # (nbins,)
    DEN_BINS_tf = tf.reduce_sum(M_STACK_tf, axis=[0,1,2,3])     # (nbins,)

    # low-freq mask (sum first LOW_BINS masks)
    if LOW_BINS > 0:
        low_mask_np = np.sum(masks[:LOW_BINS], axis=0).astype(np.float32)  # (H, W_r)
    else:
        low_mask_np = np.zeros_like(masks[0], dtype=np.float32)
    LOW_MASK_tf = tf.constant(low_mask_np[None, None, ...])     # (1,1,H,W_r)

    return W_r, M_STACK_tf, WEIGHTS_tf, DEN_BINS_tf, LOW_MASK_tf

@tf.function
def spectrum_loss_from_LA(LAy, LAt,
                          clip=12.0,
                          eps=1e-6,
                          top_frac=0.30,
                          soft=True,
                          k=20.0,
                          renorm=True,
                          z_clip=10.0,
                          std_floor=1e-2,
                          huber_delta=1.0):
    """
    LAy, LAt: log-amplitude maps (N, C, H, W).
    Robust high-band weighted mismatch with bounded z-score + bounded gradient.
    """

    # Guard numerics
    LAy = tf.where(tf.math.is_finite(LAy), LAy, tf.zeros_like(LAy))
    LAt = tf.where(tf.math.is_finite(LAt), LAt, tf.zeros_like(LAt))
    if clip is not None:
        LAy = tf.clip_by_value(LAy, -clip, clip)
        LAt = tf.clip_by_value(LAt, -clip, clip)

    # Per-sample, per-channel mean/std over H,W (target stats)
    mu  = tf.reduce_mean(LAt, axis=[2, 3], keepdims=True)
    std = tf.math.reduce_std(LAt, axis=[2, 3], keepdims=True)

    # IMPORTANT: prevent tiny std from creating huge z-scores
    std = tf.maximum(std, tf.cast(std_floor, std.dtype))
    std = tf.stop_gradient(std)
    z = ((LAy - mu) - (LAt - mu)) / std

    # IMPORTANT: bound influence of any single batch / bin
    z = tf.clip_by_value(z, -z_clip, z_clip)

    # pseudo-Huber on z (bounded gradient for large residuals)
    # huber(z) = d^2 (sqrt(1 + (z/d)^2) - 1)
    d = tf.cast(huber_delta, z.dtype)
    base = d * d * (tf.sqrt(1.0 + (z / d) ** 2) - 1.0)

    if top_frac is None or top_frac <= 0.0:
        return tf.reduce_mean(base)

    # ---- radial high-band mask on (H,W) ----
    H = tf.shape(base)[2]
    W = tf.shape(base)[3]
    yy = tf.linspace(-1.0, 1.0, H)
    xx = tf.linspace(-1.0, 1.0, W)
    YY, XX = tf.meshgrid(yy, xx, indexing="ij")
    RR = tf.sqrt(XX * XX + YY * YY)

    # NOTE: Avoid walrus operator (:=) — AutoGraph can't reliably transform it.
    top_frac_f = tf.cast(top_frac, tf.float32)
    top_frac_f = tf.clip_by_value(top_frac_f, 0.0, 1.0)
    r_cut = tf.sqrt(tf.maximum(0.0, 1.0 - top_frac_f))  # outer ring area ~ top_frac

    if soft:
        mask_hw = 1.0 / (1.0 + tf.exp(-k * (RR - r_cut)))
    else:
        mask_hw = tf.cast(RR >= r_cut, tf.float32)

    mask = mask_hw[None, None, :, :]  # (1,1,H,W)
    weighted = base * mask

    if renorm:
        mask_mean = tf.reduce_mean(mask)
        return tf.reduce_mean(weighted) / (mask_mean + tf.cast(eps, weighted.dtype))
    else:
        return tf.reduce_mean(weighted)


def energy_spectrum_loss_from_E(Ey, Et, M_STACK, WEIGHTS, DEN_BINS, eps=1e-8, eclip=None, top_frac=0.30, soft=True, k=20.0):
    # Casts + sanity
    Ey = tf.cast(Ey, tf.float32)
    Et = tf.cast(Et, tf.float32)
    M  = tf.cast(M_STACK,  tf.float32)   # (1,1,Hc,Wrc,nbins)
    Wt = tf.cast(WEIGHTS,  tf.float32)   # (nbins,)
    Db = tf.cast(DEN_BINS, tf.float32)   # (nbins,)

    Ey = tf.where(tf.math.is_finite(Ey), Ey, tf.zeros_like(Ey))
    Et = tf.where(tf.math.is_finite(Et), Et, tf.zeros_like(Et))
    if eclip is not None:
        Ey = tf.clip_by_value(Ey, 0.0, eclip)
        Et = tf.clip_by_value(Et, 0.0, eclip)

    Ey_b = Ey[..., None]  # (N,C,Hc,Wrc,1)
    Et_b = Et[..., None]

    num_y = tf.reduce_sum(Ey_b * M, axis=[1,2,3])  # (N, nbins)
    num_t = tf.reduce_sum(Et_b * M, axis=[1,2,3])

    C = tf.cast(tf.shape(Ey)[1], tf.float32)
    den = tf.maximum(Db * C, eps)
    Sy = num_y / den; St = num_t / den

    # Energy spectra can span orders of magnitude; operating in energy-space
    # makes the loss (and gradients) extremely sensitive to outliers.
    # Use log-energy to stabilize and prevent finite-but-huge explosions.
    Sy = tf.math.log(tf.maximum(Sy, 1e-12))
    St = tf.math.log(tf.maximum(St, 1e-12))

    # Use direct log-spectrum difference. Standardizing by std(St) can still
    # create sharp gain / gradient spikes and has no real physical meaning here.
    scale = tf.stop_gradient(tf.reduce_mean(tf.abs(St), axis=-1, keepdims=True))
    scale = tf.maximum(scale, 1.0)  # or 0.5, tune
    diff = (Sy - St) / scale

    # Optional: cap to avoid a single batch dominating optimizer moments.
    diff = tf.clip_by_value(diff, -10.0, 10.0)

    # ---- high-bin emphasis on the last top_frac of bins ----
    nbins = tf.shape(Db)[0]
    # Use cumulative density so the split respects bin density, not just index
    cum = tf.cumsum(Db) / (tf.reduce_sum(Db) + eps)           # (nbins,)
    cut = 1.0 - tf.cast(top_frac, tf.float32)

    if soft:
        mask_bins = 1.0 / (1.0 + tf.exp(-k * (cum - cut)))    # smooth
    else:
        mask_bins = tf.cast(cum >= cut, tf.float32)           # hard

    w = Wt * mask_bins                                        # (nbins,)

    out = tf.reduce_mean(tf.abs(diff) * w[None, :])
    return tf.where(tf.math.is_finite(out), out, tf.constant(0.0, tf.float32))

#
# -----------------------------------------------------------------------------
# Compatibility helpers (validate.py / training.py)
# -----------------------------------------------------------------------------

_SPEC_CACHE = {}


def build_spec_cache(H, W, border=None):
    """Build (or fetch) cached spectrum masks for given spatial dims.

    Returns a dict with:
      - M_STACK, WEIGHTS, DEN_BINS, LOW_MASK

    Notes:
      * Uses the same radial binning as build_radial_masks().
      * The returned tensors are float32 and broadcast-friendly.
    """
    b = SPEC_BORDER if border is None else int(border)
    key = (int(H), int(W), b)
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]

    Hc = int(H) - 2 * b
    Wc = int(W) - 2 * b
    if Hc <= 0 or Wc <= 0:
        raise ValueError(f"Invalid spectrum border={b} for H={H}, W={W}")

    M_STACK, WEIGHTS, DEN_BINS = build_radial_masks(Hc, Wc)

    # LOW_MASK: bins < LOW_BINS (in terms of radial bin index)
    # M_STACK is (1,1,Hc,Wrc,nbins)
    LOW_MASK = tf.reduce_sum(M_STACK[..., :LOW_BINS], axis=-1)  # (1,1,Hc,Wrc)

    out = {
        'border': b,
        'Hc': Hc,
        'Wc': Wc,
        'M_STACK': M_STACK,
        'WEIGHTS': WEIGHTS,
        'DEN_BINS': DEN_BINS,
        'LOW_MASK': LOW_MASK,
    }
    _SPEC_CACHE[key] = out
    return out


def _crop_nchw(x, border):
    if border <= 0:
        return x
    return x[:, :, border:-border, border:-border]


_HANN_CACHE = {}


def _hann2d(H, W):
    key = (int(H), int(W))
    if key in _HANN_CACHE:
        return _HANN_CACHE[key]
    wh = tf.signal.hann_window(H, periodic=True, dtype=tf.float32)  # (H,)
    ww = tf.signal.hann_window(W, periodic=True, dtype=tf.float32)  # (W,)
    w2 = wh[:, None] * ww[None, :]                                  # (H,W)
    w2 = tf.reshape(w2, [1, 1, H, W])
    _HANN_CACHE[key] = w2
    return w2


def energy_spectrum2d(x, M_STACK=None, WEIGHTS=None, DEN_BINS=None, border=None, eps=1e-8):
    """Compute 2D energy map (|FFT|^2) in rFFT coordinates.

    The binning into 1D radial spectra is handled elsewhere via M_STACK, etc.

    Args:
      x: (N,C,H,W) NCHW tensor.
      border: optional int border crop; defaults to SPEC_BORDER.

    Returns:
      E: (N,C,Hc,Wrc) float32, where Wrc = Wc//2 + 1.
    """
    b = SPEC_BORDER if border is None else int(border)
    x = tf.cast(x, tf.float32)
    # Prevent NaNs/Infs in residuals from propagating through FFT and then
    # getting silently zeroed by downstream _finite() wrappers (which makes
    # low-k / spectrum stabilizers disappear right when they're needed).
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    x = _crop_nchw(x, b)

    Hc = tf.shape(x)[2]
    Wc = tf.shape(x)[3]

    # Hann taper (matches log_amp_fft)
    Hs = x.shape[2]
    Ws = x.shape[3]
    if (Hs is not None) and (Ws is not None):
        w2 = _make_hann2d(Hc, Wc, dtype=tf.float32, periodic=True)
    else:
        wh = tf.signal.hann_window(Hc, periodic=True, dtype=tf.float32)
        ww = tf.signal.hann_window(Wc, periodic=True, dtype=tf.float32)
        w2 = wh[:, None] * ww[None, :]
        w2 = tf.reshape(w2, [1, 1, Hc, Wc])
    xw = x * w2

    N = tf.shape(xw)[0]
    C = tf.shape(xw)[1]

    x2 = tf.reshape(xw, [N * C, Hc, Wc])
    U = tf.signal.rfft2d(x2)
    # Energy / power map
    E = tf.math.real(U * tf.math.conj(U))
    # Normalize by number of spatial points to reduce scale dependence
    E = E / (tf.cast(Hc * Wc, tf.float32) + eps)
    # Final numeric guard
    E = tf.where(tf.math.is_finite(E), E, tf.zeros_like(E))

    Wrc = tf.shape(E)[2]
    E = tf.reshape(E, [N, C, Hc, Wrc])
    return tf.cast(E, tf.float32)

def lowk_loss_from_E(Ey, Et, LOW_MASK, eps=1e-8):
    Ey = tf.cast(Ey, tf.float32)
    Et = tf.cast(Et, tf.float32)
    mask = tf.cast(LOW_MASK, tf.float32)

    Ey = tf.where(tf.math.is_finite(Ey), Ey, tf.zeros_like(Ey))
    Et = tf.where(tf.math.is_finite(Et), Et, tf.zeros_like(Et))

    Ey = tf.math.log1p(tf.maximum(Ey, 0.0))
    Et = tf.math.log1p(tf.maximum(Et, 0.0))

    diff = tf.abs(Ey - Et) * mask
    msum = tf.reduce_sum(mask)
    denom = msum * tf.cast(tf.shape(Ey)[0] * tf.shape(Ey)[1], tf.float32) + eps
    out = tf.reduce_sum(diff) / denom
    out = tf.where(msum > 0.0, out, tf.constant(0.0, tf.float32))
    return tf.where(tf.math.is_finite(out), out, tf.constant(0.0, tf.float32))

def residual_loss_from_E(Ey, Et, LOW_BINS=LOW_BINS, eps=1e-8):
    """Residual mismatch outside the low-k region.

    We build a high-k mask on-the-fly using the same binning rule as
    build_radial_masks(): bin = round(|k|). The threshold is LOW_BINS.
    """
    Ey = tf.cast(Ey, tf.float32)
    Et = tf.cast(Et, tf.float32)

    Hc = tf.shape(Ey)[2]
    Wrc = tf.shape(Ey)[3]

    # Frequency grid for rFFT2d along last axis
    ky = tf.range(Hc, dtype=tf.float32)
    ky = tf.minimum(ky, tf.cast(Hc, tf.float32) - ky)
    kx = tf.range(Wrc, dtype=tf.float32)
    KY, KX = tf.meshgrid(ky, kx, indexing='ij')
    kmag = tf.sqrt(KY * KY + KX * KX)
    bin_idx = tf.cast(tf.round(kmag), tf.int32)

    high = tf.cast(bin_idx >= int(LOW_BINS), tf.float32)  # (Hc,Wrc)
    high = tf.reshape(high, [1, 1, Hc, Wrc])

    diff = tf.abs(Ey - Et) * high
    denom = tf.reduce_sum(high) * tf.cast(tf.shape(Ey)[0] * tf.shape(Ey)[1], tf.float32) + eps
    return tf.reduce_sum(diff) / denom
