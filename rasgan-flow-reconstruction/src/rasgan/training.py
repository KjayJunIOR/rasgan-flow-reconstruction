from __future__ import annotations

import os
import math
import time
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
from .config import TrainConfig, d_in_channels, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, ADV_MIN, ADV_MAX, COND_DROP_P
from .env import RuntimeFlags, configure_runtime, tf
from .data import load_h5, make_loaders
from .runtime import build_stats
from .models.factory import build_generator
from .models.discriminator import CondPatchD
from .ema import EMA
from .steerer import ValSteerer


class TrainOneStep:
    def __init__(self, net, optimizer, train_weights, clip_norm: float | None = 5.0,
                 agc: bool = True, agc_clip: float = 0.01, skip_on_bad_grads: bool = True):
        self.net = net
        self.optimizer = optimizer
        self.train_weights = list(train_weights)
        self.clip_norm = clip_norm
        self.agc = agc
        self.agc_clip = float(agc_clip)      # typical: 0.01 to 0.05
        self.skip_on_bad_grads = bool(skip_on_bad_grads)

    # NOTE:
    # `reduce_retracing=True` helps avoid building new concrete functions when
    # only non-shape aspects of inputs change. On PTX-JIT GPU paths (e.g. SM90
    # with a TF build lacking native cubins), excessive retracing can cause
    # unbounded graph/kernel-cache growth and eventual SIGKILL/OOM-like exits.
    @tf.function(reduce_retracing=True)
    def __call__(self, *inputs):
        with tf.GradientTape() as tape:
            loss = self.net(*inputs, training=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.train_weights)
        grads_vars = [(g, v) for g, v in zip(grads, self.train_weights) if g is not None]
        if not grads_vars:
            return loss

        grads, var_list = zip(*grads_vars)
        grads = list(grads)
        var_list = list(var_list)
        # (1) sanitize grads (NaN/Inf -> 0)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]

        # (2) optional: Adaptive Gradient Clipping (AGC) per-variable
        if self.agc:
            clipped = []
            for g, w in zip(grads, var_list):
                w_norm = tf.norm(tf.reshape(w, [-1]))
                g_norm = tf.norm(tf.reshape(g, [-1]))
                # max allowed grad norm proportional to weight norm
                max_g = tf.maximum(w_norm, 1e-3) * self.agc_clip
                scale = tf.minimum(1.0, max_g / (g_norm + 1e-6))
                clipped.append(g * scale)
            grads = clipped

        # (3) global norm clip (keep, but after sanitization/AGC)
        if self.clip_norm is not None:
            grads, gn = tf.clip_by_global_norm(grads, self.clip_norm)
        else:
            gn = tf.linalg.global_norm(grads)

        # (4) skip step if something is still bad (prevents optimizer state poisoning)
        if self.skip_on_bad_grads:
            ok = tf.math.is_finite(loss) & tf.math.is_finite(gn)
            ok_f = tf.cast(ok, tf.float32)
            grads = [g * ok_f for g in grads]

        grads_and_vars = tuple(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)
        return loss

class TrainOneStepGradReweight:
    """Low-memory gradient-norm loss reweighting for generator.

    Why this exists:
      - A persistent GradientTape over a big GAN graph can blow up memory.
      - Instead, we *recompute* forward/backward per loss term to estimate grad norms,
        then do one normal update step using the computed multipliers.

    Tradeoff: more compute, much lower peak memory.
    """

    def __init__(
        self,
        net_g: WithLoss_G,
        optimizer: tf.keras.optimizers.Optimizer,
        train_weights,
        *,
        keys: Tuple[str, ...],
        ema: float = 0.95,
        power: float = 0.5,
        clip: float = 5.0,
        eps: float = 1e-8,
        every: int = 1,
        deterministic_norms: bool = True,
        clip_norm: float | None = 5.0,
    ):
        self.net_g = net_g
        self.optimizer = optimizer
        self.train_weights = list(train_weights)
        self.keys = tuple(keys)
        self.ema = float(ema)
        self.power = float(power)
        self.clip = float(clip)
        self.eps = float(eps)
        self.every = max(1, int(every))
        self.deterministic_norms = bool(deterministic_norms)
        self.clip_norm = clip_norm

        # State: EMA grad norms + cached alphas
        self._step = tf.Variable(0, trainable=False, dtype=tf.int64, name="grw_step")
        self._gn_ema = {k: tf.Variable(1.0, trainable=False, dtype=tf.float32, name=f"grw_gn_ema_{k}") for k in self.keys}
        self._alpha = {k: tf.Variable(1.0, trainable=False, dtype=tf.float32, name=f"grw_alpha_{k}") for k in self.keys}

    def _term_means(self, lr, hr, pod_coeffs):
        terms = self.net_g._compute_terms(lr, hr, pod_coeffs=pod_coeffs, deterministic=self.deterministic_norms)
        # Reduce each to scalar for grad computation
        means = {k: tf.reduce_mean(terms[k]) for k in terms.keys() if k in self.keys or k == "total"}
        return means

    @tf.function(reduce_retracing=True)
    def __call__(self, lr, hr, pod_coeffs=None):
        # Increment step
        self._step.assign_add(1)
        do_update = tf.equal(tf.math.floormod(self._step - 1, self.every), 0)

        # (A) Optionally recompute grad norms and update alphas (low-memory: one tape per term)
        def _update_alphas():
            # Compute grad norms per term sequentially
            for k in self.keys:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.train_weights)
                    means = self._term_means(lr, hr, pod_coeffs)
                    lk = means.get(k, None)
                    lk = tf.cast(lk if lk is not None else 0.0, tf.float32)
                grads_k = tape.gradient(lk, self.train_weights)
                grads_k = [g for g in grads_k if g is not None]
                gn = tf.linalg.global_norm(grads_k) if grads_k else tf.constant(0.0, tf.float32)
                gn = tf.where(tf.math.is_finite(gn), gn, tf.constant(0.0, tf.float32))
                # EMA update
                self._gn_ema[k].assign(self.ema * self._gn_ema[k] + (1.0 - self.ema) * tf.cast(gn, tf.float32))

            # Target = mean EMA norm across selected terms
            gns = tf.stack([self._gn_ema[k] for k in self.keys])
            target = tf.reduce_mean(gns)
            for k in self.keys:
                denom = self._gn_ema[k] + self.eps
                a = tf.pow(target / denom, self.power)
                a = tf.clip_by_value(a, 1.0 / self.clip, self.clip)
                self._alpha[k].assign(tf.cast(a, tf.float32))
            return 0

        tf.cond(do_update, _update_alphas, lambda: 0)

        # (B) Actual update step (single tape)
        with tf.GradientTape() as tape:
            terms = self.net_g._compute_terms(lr, hr, pod_coeffs=pod_coeffs, deterministic=False)
            total = tf.constant(0.0, tf.float32)
            for k, v in terms.items():
                if k == "total":
                    continue
                v = tf.reduce_mean(v)
                if k in self.keys:
                    total += tf.stop_gradient(self._alpha[k]) * v
                else:
                    total += v
            total = tf.where(tf.math.is_finite(total), total, tf.ones_like(total))
        grads = tape.gradient(total, self.train_weights)
        grads_vars = [(g, v) for g, v in zip(grads, self.train_weights) if g is not None]
        if grads_vars:
            grads, vars_ = zip(*grads_vars)
            if self.clip_norm is not None:
                grads, _ = tf.clip_by_global_norm(list(grads), self.clip_norm)
            self.optimizer.apply_gradients(zip(grads, vars_))

        # For logging: alpha stats
        a_stack = tf.stack([self._alpha[k] for k in self.keys]) if self.keys else tf.constant([1.0], tf.float32)
        return total, tf.reduce_min(a_stack), tf.reduce_mean(a_stack), tf.reduce_max(a_stack)

class TrainOneStepGradReweightLowMem:
    """Gradient-magnitude loss reweighting for the generator (single-forward-pass).

    Key design goal: do **not** add extra forward passes (which can OOM on GPU).
    We take one forward pass to compute all loss terms, then (optionally)
    compute per-term grad norms from the SAME tape to update multipliers.

    When `every>1`, multipliers are updated only every N steps to reduce
    extra backward passes.
    """

    def __init__(
        self,
        net_g: "WithLoss_G",
        optimizer: tf.keras.optimizers.Optimizer,
        train_weights,
        *,
        keys: Tuple[str, ...],
        every: int = 4,
        ema: float = 0.95,
        power: float = 0.5,
        clip: float = 5.0,
        eps: float = 1e-8,
        deterministic_norms: bool = True,
        clip_norm: float | None = 5.0,
    ):
        self.net_g = net_g
        self.optimizer = optimizer
        self.train_weights = list(train_weights)
        self.keys = tuple(keys)
        self.every = max(1, int(every))
        self.ema = float(ema)
        self.power = float(power)
        self.clip = float(clip)
        self.eps = float(eps)
        self.deterministic_norms = bool(deterministic_norms)
        self.clip_norm = clip_norm

        self._step = tf.Variable(0, trainable=False, dtype=tf.int64, name="grw_step")
        self._gn_ema = {k: tf.Variable(1.0, trainable=False, dtype=tf.float32, name=f"grw_gn_ema_{k}") for k in self.keys}
        self._alpha = {k: tf.Variable(1.0, trainable=False, dtype=tf.float32, name=f"grw_alpha_{k}") for k in self.keys}

    @tf.function(reduce_retracing=True)
    def __call__(self, lr, hr, pod_coeffs=None):
        # Step counter
        self._step.assign_add(1)
        do_update = tf.equal(tf.math.floormod(self._step - 1, self.every), 0)

        # One forward pass under a persistent tape so we can query multiple gradients.
        with tf.GradientTape(persistent=True) as tape:
            terms = self.net_g._compute_terms(
                lr,
                hr,
                pod_coeffs=pod_coeffs,
                deterministic=bool(self.deterministic_norms),
            )
            # Reduce to scalars
            means = {k: tf.reduce_mean(v) for k, v in terms.items() if k != "total"}
            # Weighted total (use current alphas; stop_gradient => first-order)
            total = tf.constant(0.0, tf.float32)
            for k, v in means.items():
                if k in self._alpha:
                    total += tf.stop_gradient(self._alpha[k]) * tf.cast(v, tf.float32)
                else:
                    total += tf.cast(v, tf.float32)
            total = tf.where(tf.math.is_finite(total), total, tf.ones_like(total))

        # Update alphas every N steps (extra backward passes, no extra forward pass)
        def _update_alphas():
            for k in self.keys:
                lk = tf.cast(means.get(k, 0.0), tf.float32)
                grads_k = tape.gradient(lk, self.train_weights)
                grads_k = [g for g in grads_k if g is not None]
                gn = tf.linalg.global_norm(grads_k) if grads_k else tf.constant(0.0, tf.float32)
                gn = tf.where(tf.math.is_finite(gn), gn, tf.constant(0.0, tf.float32))
                self._gn_ema[k].assign(self.ema * self._gn_ema[k] + (1.0 - self.ema) * tf.cast(gn, tf.float32))

            gns = tf.stack([self._gn_ema[k] for k in self.keys])
            target = tf.reduce_mean(gns)
            for k in self.keys:
                denom = self._gn_ema[k] + self.eps
                a = tf.pow(target / denom, self.power)
                a = tf.clip_by_value(a, 1.0 / self.clip, self.clip)
                self._alpha[k].assign(tf.cast(a, tf.float32))
            return 0

        tf.cond(do_update, _update_alphas, lambda: 0)

        # Apply gradients for weighted total
        grads = tape.gradient(total, self.train_weights)
        del tape

        grads_vars = [(g, v) for g, v in zip(grads, self.train_weights) if g is not None]
        if grads_vars:
            grads, vars_ = zip(*grads_vars)
            if self.clip_norm is not None:
                grads, _ = tf.clip_by_global_norm(list(grads), self.clip_norm)
            self.optimizer.apply_gradients(zip(grads, vars_))

        a_stack = tf.stack([self._alpha[k] for k in self.keys]) if self.keys else tf.constant([1.0], tf.float32)
        return total, tf.reduce_min(a_stack), tf.reduce_mean(a_stack), tf.reduce_max(a_stack)

def _make_adamw(learning_rate: float, weight_decay: float) -> tf.keras.optimizers.Optimizer:
    """Create AdamW with a safe fallback across TF/Keras variants.

    TF 2.17 should provide tf.keras.optimizers.AdamW. Some installs expose it
    under tf.keras.optimizers.experimental.AdamW. If neither exists, fall back
    to Adam (no decoupled decay) so the code still runs.
    """
    # Prefer standard Keras API (TF 2.17+)
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
    except Exception:
        pass
    # Some versions keep AdamW in the experimental namespace
    try:
        return tf.keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
    except Exception:
        # Last-resort fallback (no decoupled decay)
        print("[warn] AdamW not available; falling back to Adam (no weight decay).")
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )


def _lr_value(opt: tf.keras.optimizers.Optimizer, default: float) -> float:
    """Best-effort numeric LR for checkpoint metadata."""
    for attr in ("learning_rate", "lr"):
        if hasattr(opt, attr):
            v = getattr(opt, attr)
            try:
                return float(v)
            except Exception:
                try:
                    return float(v.numpy())
                except Exception:
                    pass
    return float(default)

from .schedules import adv_weight_schedule, edge_lambda_at_epoch, physics_ramp_factor, scale_physics_weights, noise_at_epoch, pod_vel_blend_alpha, fft_deramp_factor
from .validate import _validate_epoch
from .losses.spectrum import set_spec_border, build_radial_masks
from .losses.wrappers import WithLoss_init, WithLoss_D, WithLoss_G
from .losses.physics import PhysicsParams
from .pod_sidecar import load_pod_sidecar
from .losses.pod import PodParams
from .utils.checkpoint import _save_checkpoint, _load_checkpoint, _restore_train_state

def _to_float(x, default=float("nan")) -> float:
    """Best-effort scalar extraction for python floats, numpy scalars, tf.Tensor/tf.Variable."""
    try:
        if x is None:
            return default
        # tf.Variable / tf.Tensor
        if hasattr(x, "numpy"):
            v = x.numpy()
            # numpy scalar / 0-d array
            try:
                return float(v)
            except Exception:
                pass
        return float(x)
    except Exception:
        return default

def _attr_float(obj, name: str, default=float("nan")) -> float:
    try:
        return _to_float(getattr(obj, name), default)
    except Exception:
        return default

def _opt_lr(opt, default=float("nan")) -> float:
    """Extract current optimizer LR for tf.keras optimizers (var or schedule)."""
    try:
        lr = getattr(opt, "learning_rate", None)
        if lr is None:
            lr = getattr(opt, "lr", None)
        # If schedule, it may be callable with iterations
        if callable(lr):
            it = getattr(opt, "iterations", None)
            return _to_float(lr(it), default)
        return _to_float(lr, default)
    except Exception:
        return default

def _fmt_knobs(net_g) -> str:
    """Compact string of live generator knobs (only includes ones that exist)."""
    keys = [
        ("aw", "adv_weight"),
        ("noise", "inst_noise_std"),
        ("pdrop", "cond_drop_p"),
        ("w_spec", "w_spec"),
        ("w_espec", "w_espec"),
        ("w_res", "w_res"),
        ("w_low", "w_low"),
        ("grad_w", "grad_w"),
        ("tv_w", "tv_w"),
        ("decor_w", "decorr_weight"),
        ("cov_w", "cov_weight"),
    ]
    parts = []
    for label, attr in keys:
        v = _attr_float(net_g, attr, default=float("nan"))
        if math.isfinite(v):
            parts.append(f"{label} {v:.3e}")
    return " | ".join(parts)

def _batch_stats(x, name):
    # x is NCHW float32
    x = tf.cast(x, tf.float32)
    return (f"{name}: min {_to_float(tf.reduce_min(x)):.3e} "
            f"max {_to_float(tf.reduce_max(x)):.3e} "
            f"mean {_to_float(tf.reduce_mean(x)):.3e} "
            f"absmax {_to_float(tf.reduce_max(tf.abs(x))):.3e}")

def _build_models(cfg: TrainConfig, lr_shape: Tuple[int,int,int], hr_shape: Tuple[int,int,int]):
    c_lr, h_lr, w_lr = lr_shape
    c_hr, h_hr, w_hr = hr_shape

    # Infer and validate scale from paired spatial shapes. The data contract is
    # authoritative because a mismatch cannot be repaired safely inside the model.
    if (h_hr, w_hr) == (h_lr, w_lr):
        sr_scale = 1
    elif (h_hr, w_hr) == (2 * h_lr, 2 * w_lr):
        sr_scale = 2
    else:
        raise ValueError(
            "RASGAN currently supports same-grid refinement or 2x super-resolution; "
            f"got LR={(h_lr, w_lr)} and HR={(h_hr, w_hr)}."
        )
    if int(getattr(cfg, "scale", sr_scale)) != sr_scale:
        print(
            f"[data] --scale={getattr(cfg, 'scale', None)} does not match the paired arrays; "
            f"using scale={sr_scale} inferred from LR/HR shapes."
        )
    G = build_generator(cfg, sr_scale=sr_scale, lr_shape=lr_shape)
    G.init_build(tf.zeros((cfg.batch_size, c_lr, h_lr, w_lr), dtype=tf.float32))

    d_in = d_in_channels(use_coords=getattr(G, "use_coords", False))
    D = CondPatchD()
    D.init_build(tf.zeros((cfg.batch_size, d_in, h_hr, w_hr), dtype=tf.float32))
    return G, D

def _load_weights_from_ckpt_dir(resume_dir: str, G, D, ema) -> None:
    """Load generator/discriminator weights from a checkpoint directory.

    Mirrors the original script's behavior:
      - If generator_ema.npz exists: load it to live G, register EMA shadow, then swap live to RAW if present.
      - Else: load RAW and register as EMA shadow.
      - Load discriminator.npz if present (best-effort).
    """
    p_ema = os.path.join(resume_dir, "generator_ema.weights.h5")
    p_raw = os.path.join(resume_dir, "generator_raw.weights.h5")

    if os.path.exists(p_ema):
        G.load_weights(p_ema)
        try:
            ema.register()
        except Exception:
            pass
        if os.path.exists(p_raw):
            G.load_weights(p_raw)
    else:
        if os.path.exists(p_raw):
            G.load_weights(p_raw)
        try:
            ema.register()
        except Exception:
            pass

    p_d = os.path.join(resume_dir, "discriminator.weights.h5")
    if os.path.exists(p_d):
        try:
            D.load_weights(p_d)
        except Exception as e:
            print(f"[!] Failed to load D from {p_d}: {e}")

def train(cfg: TrainConfig) -> None:
    # Runtime toggles
    configure_runtime(RuntimeFlags(mixed=cfg.mixed, xla=cfg.xla, tlx_verbose=cfg.tlx_verbose))

    # Data
    h5 = load_h5(cfg.data_path)
    stats = build_stats(h5.means_hr, h5.stds_hr, h5.means_lr, h5.stds_lr)

    phys_params = PhysicsParams(dx=cfg.dx, dy=cfg.dy, nu=cfg.nu, rho=cfg.rho,
                               bc=cfg.bc, poisson_method=cfg.poisson_method, poisson_iters=cfg.poisson_iters)

    # Optional POD sidecar (modes + per-sample coeffs for training)
    pod_sidecar = None
    if getattr(cfg, "pod_mat_path", None):
        pod_sidecar = load_pod_sidecar(cfg.pod_mat_path, k=cfg.pod_k)

    # If the H5 preprocessing shuffled snapshots, we must apply the same
    # permutation to the POD time-coefficients so they line up with the
    # training samples.
    pod_tim_u = None if pod_sidecar is None else pod_sidecar.tim_u
    pod_tim_v = None if pod_sidecar is None else pod_sidecar.tim_v
    pod_tim_p = None if pod_sidecar is None else pod_sidecar.tim_p
    if pod_sidecar is not None and getattr(h5, "perm", None) is not None:
        perm = np.asarray(h5.perm, dtype=np.int64).reshape(-1)
        # H5 preprocessing permuted snapshots and then split into train/test.
        # Reorder POD coefficients into the SAME permuted order (train followed by test)
        # so both training and validation see aligned coefficients.
        idx_all = perm

        def _reindex_coeff(name: str, arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim != 2:
                raise ValueError(f"POD coeff '{name}' must be 2D [N,K], got {arr.shape}")
            if arr.shape[0] < int(np.max(idx_all)) + 1:
                raise ValueError(
                    f"POD coeff '{name}' has N={arr.shape[0]} rows but needs at least {int(np.max(idx_all))+1} "
                    f"to match the H5 permutation (max idx overall={int(np.max(idx_all))}). "
                    "Regenerate the POD sidecar on the full snapshot set, or export coeffs for all N."
                )
            return arr[idx_all]

        pod_tim_u = _reindex_coeff("tim_u", pod_tim_u)
        pod_tim_v = _reindex_coeff("tim_v", pod_tim_v)
        pod_tim_p = _reindex_coeff("tim_p", pod_tim_p)
    elif pod_sidecar is not None and getattr(h5, "perm", None) is None:
        raise ValueError(
            f"POD coeff must be ordered to match training snapshots"
        )

    train_loader, val_loader = make_loaders(
        h5,
        batch_size=cfg.batch_size,
        shuffle=True,
        pod_tim_u=pod_tim_u,
        pod_tim_v=pod_tim_v,
        pod_tim_p=pod_tim_p,
    )

    # Build models at the actual training spatial size.
    lr_shape = tuple(h5.lr_train.shape[1:])  # (C,H,W)
    hr_shape = tuple(h5.hr_train.shape[1:])  # (C,H,W)

    # Prepare POD params (modes live in LR space)
    pod_params_tf: Optional[PodParams] = None
    if pod_sidecar is not None:
        h_lr = int(lr_shape[1]); w_lr = int(lr_shape[2])
        hw = h_lr * w_lr
        if int(pod_sidecar.phiu.shape[0]) != hw or int(pod_sidecar.phiv.shape[0]) != hw or int(pod_sidecar.phip.shape[0]) != hw:
            raise ValueError(
                f"POD sidecar modes first dim must match HW={hw} (H={h_lr},W={w_lr}). "
                f"Got phiu={pod_sidecar.phiu.shape}, phiv={pod_sidecar.phiv.shape}, phip={pod_sidecar.phip.shape}."
            )
        pod_params_tf = PodParams(
            phiu=tf.convert_to_tensor(pod_sidecar.phiu, dtype=tf.float32),
            phiv=tf.convert_to_tensor(pod_sidecar.phiv, dtype=tf.float32),
            phip=tf.convert_to_tensor(pod_sidecar.phip, dtype=tf.float32),
            h_lr=h_lr,
            w_lr=w_lr,
        )
    G, D = _build_models(cfg, lr_shape, hr_shape)
    # If the generator supports POD/time-coefficient conditioning, inject
    # dataset-specific HR normalization so coefficients are scaled correctly
    if hasattr(G, "set_coeff_norm_from_hr"):
        try:
            G.set_coeff_norm_from_hr(stats.hr_mean, stats.hr_std)
        except Exception:
            pass

    # Spectrum masks based on cropped size
    set_spec_border(4)
    border = 4
    Hc = int(hr_shape[1] - 2 * border)
    Wc = int(hr_shape[2] - 2 * border)
    Wrc, M_STACK, WEIGHTS, DEN_BINS, LOW_MASK = build_radial_masks(Hc, Wc)
    # Keep as TF tensors (validation/loss code expects tf tensors)
    M_STACK  = tf.convert_to_tensor(M_STACK,  dtype=tf.float32)
    WEIGHTS  = tf.convert_to_tensor(WEIGHTS,  dtype=tf.float32)
    DEN_BINS = tf.convert_to_tensor(DEN_BINS, dtype=tf.float32)
    LOW_MASK = tf.convert_to_tensor(LOW_MASK, dtype=tf.float32)

    # EMA + steerer
    ema = EMA(G)
    steerer = None  # initialized after net_g is built
    #d_ema_real = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="d_ema_real")
    #d_ema_fake = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="d_ema_fake")

    # Prefer composite validation score in adversarial stage steering
    os.environ.setdefault("VAL_PRIMARY", "score")

    # Optimizers (default; can be overridden by checkpoint restore)
    # Use AdamW (decoupled weight decay). Weight decay values are in cfg.
    g_opt_init = _make_adamw(cfg.g_lr_i, getattr(cfg, "g_weight_decay", 0.0))
    g_opt_adv  = _make_adamw(cfg.g_lr_adv, getattr(cfg, "g_weight_decay", 0.0))
    d_opt_adv  = _make_adamw(cfg.d_lr_adv, getattr(cfg, "d_weight_decay", 0.0))

    # Build loss wrappers
    net_init = WithLoss_init(G, low_mask=LOW_MASK, w_res_init=cfg.init_weights.w_res, w_low_init=cfg.init_weights.w_low,
                             p_up=cfg.init_weights.pix_w[0], lp_kernel=11)
    net_init.grad_w = float(cfg.init_weights.grad_w)
    net_init.tv_w = float(cfg.init_weights.tv_w)
    net_init.content_w = float(cfg.init_weights.content_w)
    # Configure physics losses (disabled unless weights > 0)
    net_init.set_physics(stats.hr_mean, stats.hr_std, phys_params)
    net_init.w_vomega = float(cfg.init_weights.w_vomega)
    net_init.w_omcons = float(cfg.init_weights.w_omcons)
    net_init.w_div    = float(cfg.init_weights.w_div)
    net_init.w_mom    = float(cfg.init_weights.w_mom)
    net_init.w_ppois  = float(cfg.init_weights.w_ppois)

    if pod_params_tf is not None:
        net_init.set_pod(stats.lr_mean, stats.lr_std, pod_params_tf)
        net_init.w_pod_vel = float(cfg.init_weights.w_pod_vel)
        net_init.w_pod_p   = float(cfg.init_weights.w_pod_p)
        net_init.w_pod_w   = float(cfg.init_weights.w_pod_w)

    train_init = TrainOneStep(net_init, optimizer=g_opt_init, train_weights=G.trainable_weights, clip_norm=5.0)

    net_d = WithLoss_D(D, G)
    net_g = WithLoss_G(
        D, G,
        adv_weight_schedule(0, cfg.adv_epochs, steerer=None, vm=None),
        cfg.adv_weights.w_spec, cfg.adv_weights.w_espec, cfg.adv_weights.w_res, cfg.adv_weights.w_low,
        M_STACK, WEIGHTS, DEN_BINS, LOW_MASK, inst_noise_std=noise_at_epoch(1)
    )
    # weights for G loss
    net_g.pix_w  = tuple(float(x) for x in cfg.adv_weights.pix_w)
    net_g.grad_w = float(cfg.adv_weights.grad_w)
    net_g.tv_w   = float(cfg.adv_weights.tv_w)
    net_g.content_w = float(cfg.adv_weights.content_w)
    net_g.set_physics(stats.hr_mean, stats.hr_std, phys_params)
    net_g.w_vomega = float(cfg.adv_weights.w_vomega)
    net_g.w_omcons = float(cfg.adv_weights.w_omcons)
    net_g.w_div    = float(cfg.adv_weights.w_div)
    net_g.w_mom    = float(cfg.adv_weights.w_mom)
    net_g.w_ppois  = float(cfg.adv_weights.w_ppois)

    if pod_params_tf is not None:
        net_g.set_pod(stats.lr_mean, stats.lr_std, pod_params_tf)
        net_g.w_pod_vel = float(cfg.adv_weights.w_pod_vel)
        net_g.w_pod_p   = float(cfg.adv_weights.w_pod_p)
        net_g.w_pod_w   = float(cfg.adv_weights.w_pod_w)

    train_d = TrainOneStep(net_d, optimizer=d_opt_adv, train_weights=D.trainable_weights, clip_norm=5.0)
    # Optionally reweight selected generator loss terms by gradient magnitude.
    if cfg.grad_reweight:
        train_g = TrainOneStepGradReweightLowMem(
            net_g,
            optimizer=g_opt_adv,
            train_weights=G.trainable_weights,
            keys=cfg.grad_reweight_keys,
            ema=cfg.grad_reweight_ema,
            power=cfg.grad_reweight_power,
            clip=cfg.grad_reweight_clip,
            eps=cfg.grad_reweight_eps,
            every=cfg.grad_reweight_every,
            deterministic_norms=cfg.grad_reweight_deterministic_norms,
            clip_norm=5.0,
        )
        print(
            "[grad_reweight] enabled (low-mem recompute). "
            f"terms={','.join(cfg.grad_reweight_keys)} | "
            f"every={cfg.grad_reweight_every} ema={cfg.grad_reweight_ema} "
            f"power={cfg.grad_reweight_power} clip={cfg.grad_reweight_clip} "
            f"det_norms={cfg.grad_reweight_deterministic_norms}"
        )
    else:
        train_g = TrainOneStep(
            net_g, optimizer=g_opt_adv, train_weights=G.trainable_weights, clip_norm=5.0
        )
    steerer = ValSteerer(g_opt_adv, d_opt_adv, net_g, net_d, cfg.g_lr_adv, cfg.d_lr_adv)

    # Resume (best-effort)
    # If resume_stage is explicitly provided on the CLI, we honor it.
    # Otherwise we resume the stage stored in the checkpoint metadata.
    start_stage = cfg.resume_stage
    start_epoch_init = 1
    start_epoch_adv = 1
    if cfg.resume_from:
        meta = _load_checkpoint(cfg.resume_from)
        # rebuild models from meta if shapes mismatch? we assume compatible
        _restore_train_state(meta, g_opt_adv, d_opt_adv, steerer, net_g, net_d)
        # Load weights from checkpoint directory
        resume_dir = cfg.resume_from
        if os.path.isfile(resume_dir):
            resume_dir = os.path.dirname(resume_dir)
        _load_weights_from_ckpt_dir(resume_dir, G, D, ema)
        # Determine where to resume
        # (This is handled inside _restore_train_state in the original script's checkpointing code.)
        # If your checkpoint doesn't restore weights, use `Model.load_weights(...)` here.
        stage = meta.get("stage")
        epoch = int(meta.get("epoch", 0))

        # Only follow the checkpoint's stage if the user didn't override it.
        if cfg.resume_stage is None:
            start_stage = "adv" if str(stage).startswith("adv") else stage
        # Epoch counters: if we are resuming *the same stage* as the checkpoint,
        # continue from epoch+1. If we are forcing a different stage, start at 1.
        if start_stage == "init":
            start_epoch_init = max(1, (epoch + 1) if stage == "init" else 1)
        elif start_stage == "adv":
            is_adv_family = str(stage).startswith("adv")
            start_epoch_adv  = max(1, (epoch + 1) if is_adv_family else 1)

    # Decide what to run
    do_init = (start_stage in (None, "init"))
    do_adv  = (start_stage in (None, "adv"))
    steps_per_epoch = h5.lr_train.shape[0] // cfg.batch_size
    # ---------------- Init stage ----------------
    if do_init:
        G.set_train(); D.set_eval()
        print(f"\n=== Pretrain G for {cfg.init_epochs} epochs ===")
        best_val_init_obj = float('inf')
        # Track last validation metrics so the epoch summary can include them
        # even when we don't validate every epoch.
        last_val_mae = float('nan')
        last_psnr = float('nan')
        last_vomega = 0.0
        last_omcons = 0.0
        last_div = 0.0
        last_mom = 0.0
        last_ppois = 0.0

        es_wait = 0

        # Use a single persistent iterator over the repeated training dataset.
        #
        # Rationale: repeatedly creating new iterators / `.take(...)` datasets
        # per epoch can leak tf.data resources (threads/buffers) on some TF
        # versions and looks like a repeatable "hard exit" after N epochs.
        # Since `train_loader` already includes `.repeat()`, consuming exactly
        # `steps_per_epoch` batches per epoch still corresponds to one full pass.
        train_iter = iter(train_loader)

        for epoch in range(start_epoch_init, cfg.init_epochs + 1):
            t0 = time.time()
            run_loss = 0.0
            nb = 0
            net_init.set_pod_vel_blend_alpha(pod_vel_blend_alpha(epoch, ramp_epochs=10))
            for step in range(steps_per_epoch):
                batch = next(train_iter)
                # batch may be (lr, hr) or (lr, hr, tim_u, tim_v, tim_p)
                if isinstance(batch, (tuple, list)) and len(batch) == 5:
                    lr, hr, tim_u, tim_v, tim_p = batch
                    pod_coeffs = (tim_u, tim_v, tim_p)
                else:
                    lr, hr = batch
                    pod_coeffs = None
                loss = train_init(lr, hr, pod_coeffs)
                run_loss += float(loss)
                nb += 1
                if (step + 1) % cfg.print_every_steps == 0:
                    print(f"[init] ep {epoch:04d} step {step+1:05d} loss {run_loss/nb:.5f}")
            # Validate
            if (epoch % cfg.val_every) == 0:
                vm = _validate_epoch(
                    val_loader,
                    net_init,
                    stats.hr_mean_np,
                    stats.hr_std_np,
                    stats.lr_mean_np,
                    stats.lr_std_np,
                    phys_params=None,
                    pod_params=pod_params_tf,
                    score_weights=asdict(cfg.init_weights),
                )
                last_val_mae = float(vm.get('val_mae', float('nan')))
                # validator uses val_psnr; keep backward-compat with older key names
                last_psnr = float(vm.get('val_psnr', vm.get('psnr', float('nan'))))
                mp = float(vm.get("val_mae_p", float("nan")))
                mv = float(vm.get("val_mae_v", float("nan")))
                mw = float(vm.get("val_mae_w", float("nan")))
                last_vomega = float(vm.get('vomega', 0.0))
                last_omcons = float(vm.get('wcons', 0.0))
                last_div    = float(vm.get('div', 0.0))
                last_mom    = float(vm.get('mom', 0.0))
                last_ppois  = float(vm.get('ppois', 0.0))
                print(f"[init] ep {epoch:04d} val mae {last_val_mae:.6f}  psnr {last_psnr:.3f}")
                # Early stopping on validation MAE (lower is better)
                vobj = float(vm.get("val_mae", float("inf")))
                if (best_val_init_obj - vobj) > EARLY_STOP_MIN_DELTA:
                    best_val_init_obj = vobj
                    es_wait = 0
                    # Save best init checkpoint (lower val_mae is better)
                    if cfg.save_best:
                        best_dir = os.path.join(cfg.ckpt_dir, cfg.best_ckpt_init_name)
                        _save_checkpoint(
                            best_dir, "init_best", epoch, G, D, ema,
                            input_shape=(cfg.batch_size,) + lr_shape,
                            g_lr=_lr_value(g_opt_init, cfg.g_lr_adv),
                            d_lr=None,
                            state={'cfg': asdict(cfg), 'epoch': epoch, "val_total": float(vm.get("val_total", float("nan"))),
                            "val_mae": float(vm.get("val_mae", float("nan")))}
                        )
                    best_val_mae = float(vm.get("val_total", float("nan")))
                else:
                    es_wait += 1
                    if es_wait >= 25:
                        print(f"[init] early stop: val_mae did not improve for 10 validations (best {best_val_mae:.6f})")
                        break
            # Save ckpt
            if (epoch % cfg.save_every == 0) or (epoch == cfg.init_epochs):

                save_dir = os.path.join(cfg.ckpt_dir, f"ckpt_init_ep{epoch:04d}")

                _save_checkpoint(
                             save_dir, "init", epoch, G, D, ema,
                             # Generator input is LR (N,C,H,W). Store LR shape in meta so
                             # inference scripts can build the model correctly.
                             input_shape=(cfg.batch_size,)+lr_shape,
                             g_lr=_lr_value(g_opt_init, cfg.g_lr_adv),
                             d_lr=None,
                             state={'cfg': asdict(cfg)})
            print(
                f"[init] epoch {epoch} done in {time.time()-t0:.1f}s | "
                f"val_mae {last_val_mae:.6f} (p {mp:.6f} v {mv:.6f} w {mw:.6f}) (denorm {vm.get('val_mae_denorm', float('nan')):.6f}) psnr {last_psnr:.3f} "
            )

        # After init/pretrain, the *best* checkpoint can occur well before the final
        # pretrain epoch (as in your logs where best_init is around epoch 6 but epoch 30 is worse).
        # If we roll straight into adversarial training from the last pretrain weights, we start
        # adv training from a degraded generator and can bake in artifacts.
        #
        # Fix: always restore best_init weights (if available) before continuing.
        best_init_dir = os.path.join(cfg.ckpt_dir, cfg.best_ckpt_init_name)
        best_init_w = os.path.join(best_init_dir, 'generator_raw.weights.h5')
        if os.path.exists(best_init_w):
            try:
                G.load_weights(best_init_w)
                # Reset EMA shadow to the restored generator weights so subsequent EMA updates
                # (during adversarial training) start from the same point.
                try:
                    ema.register(G)
                except Exception:
                    pass
                print(f"[*] Restored best_init → {best_init_dir}")
            except Exception as e:
                print(f"[warn] Failed to restore best_init from {best_init_dir}: {e}")

        start_epoch_adv = 1  # after pretrain

    # ---------------- Adv stage ----------------
    if do_adv:
        G.set_train(); D.set_train()
        print(f"\n=== Adversarial training {cfg.adv_epochs} epochs ===")
        # seed adv weight
        aw_base = adv_weight_schedule(start_epoch_adv, cfg.adv_epochs, steerer=None, vm=None)
        aw_curr = float(min(ADV_MAX, max(ADV_MIN, float(aw_base))))
        net_g.set_adv_weight(aw_curr)
        # Inform steerer of the *scheduled* ramp target (upper bound) for this epoch
        steerer.set_schedule(aw_base)

        best_score_adv = float('-inf')
        es_wait_adv = 0
        # Same rationale as init stage: keep a persistent iterator rather than
        # rebuilding `.take(...)` datasets / iterators every epoch.
        train_iter = iter(train_loader)

        for epoch in range(start_epoch_adv, cfg.adv_epochs + 1):
            t0 = time.time()
            run_d = 0.0; run_g = 0.0
            nb = 0

            # update scheduled knobs
            aw_base = adv_weight_schedule(epoch, cfg.adv_epochs, steerer=None, vm=None)
            # `adv_w_sched` is interpreted as a *cap* produced by guards from the previous epoch.
            # If no guard reduced adv_w, steerer sets this to ADV_MAX so the ramp can continue.
            aw_cap = float(getattr(steerer, "adv_w_sched", float(ADV_MAX)))
            if epoch == start_epoch_adv:
                aw_cap = float(ADV_MAX)

            # Apply ramp (upper bound) and guard cap (can only push down).
            aw_curr = min(float(aw_base), float(aw_cap))
            aw_curr = float(min(ADV_MAX, max(ADV_MIN, aw_curr)))
            net_g.set_adv_weight(aw_curr)
            # FFT deramp: gradually reduce spectral penalties to 0 over first 80 adv epochs
            fft_fac = fft_deramp_factor(epoch, deramp_epochs=80)
            net_g.set_loss_weights(
                w_spec  = float(cfg.adv_weights.w_spec)  * fft_fac,
                w_espec = float(cfg.adv_weights.w_espec) * fft_fac,
                w_low   = float(cfg.adv_weights.w_low)   * fft_fac,
            )
            # Log LIVE knobs (not just scheduled aw_curr)
            print(f"[adv] ep {epoch:04d} knobs (live): "
                  f"g_lr {_opt_lr(g_opt_adv):.3e} d_lr {_opt_lr(d_opt_adv):.3e} | "
                  f"{_fmt_knobs(net_g)}")

            net_g.set_pod_vel_blend_alpha(pod_vel_blend_alpha(epoch, ramp_epochs=30))
            # Inform the steerer of the *scheduled* target (upper bound) for this epoch.
            steerer.set_schedule(aw_base)

            # physics ramp (like adv_w): ramp physics weights from 0→configured over 20 adv epochs
            phys_fac = physics_ramp_factor(epoch, ramp_epochs=20)
            adv_w_scaled = scale_physics_weights(cfg.adv_weights, phys_fac)
            net_g.w_vomega = float(adv_w_scaled.w_vomega)
            net_g.w_omcons = float(adv_w_scaled.w_omcons)
            net_g.w_div    = float(adv_w_scaled.w_div)
            net_g.w_mom    = float(adv_w_scaled.w_mom)
            net_g.w_ppois  = float(adv_w_scaled.w_ppois)

            # edge anneal
            edge_lam = edge_lambda_at_epoch(epoch, edge_start_epoch=1)
            try:
                net_d.set_conditioning(edge_lambda=edge_lam, cond_drop_p=COND_DROP_P)
                net_g.set_conditioning(edge_lambda=edge_lam, cond_drop_p=COND_DROP_P)
            except Exception:
                pass

            for step in range(steps_per_epoch):
                batch = next(train_iter)
                if isinstance(batch, (tuple, list)) and len(batch) == 5:
                    lr, hr, tim_u, tim_v, tim_p = batch
                    pod_coeffs = (tim_u, tim_v, tim_p)
                else:
                    lr, hr = batch
                    pod_coeffs = None
                # D then G
                d_loss = train_d(lr, hr, pod_coeffs)
                g_out = train_g(lr, hr, pod_coeffs)
                if isinstance(g_out, (tuple, list)) and len(g_out) == 4:
                    g_loss, a_min, a_mean, a_max = g_out
                else:
                    g_loss = g_out; a_mean = None
                # Update EMA every G step (matches reference behavior)
                try:
                    ema.update()
                except Exception:
                    pass
                run_d += float(d_loss); run_g += float(g_loss)
                nb += 1
                if (step + 1) % cfg.print_every_steps == 0:
                    # IMPORTANT: print LIVE adv_weight (may differ from aw_curr if steerer modifies net_g)
                    aw_live = _attr_float(net_g, "adv_weight", default=aw_curr)
                    g_lr_live = _opt_lr(g_opt_adv)
                    d_lr_live = _opt_lr(d_opt_adv)
                    if a_mean is not None:
                        print(f"[adv] ep {epoch:04d} step {step+1:05d} "
                              f"d {run_d/nb:.5f} g {run_g/nb:.5f} "
                              f"aw {aw_live:.2e} g_lr {g_lr_live:.2e} d_lr {d_lr_live:.2e} "
                              f"a[min/mean/max] {float(a_min):.3g}/{float(a_mean):.3g}/{float(a_max):.3g}")
                    else:
                        print(f"[adv] ep {epoch:04d} step {step+1:05d} "
                              f"d {run_d/nb:.5f} g {run_g/nb:.5f} "
                              f"aw {aw_live:.2e} g_lr {g_lr_live:.2e} d_lr {d_lr_live:.2e}")
                    # --- DEBUG: term breakdown near the known blow-up window ---
                    # Trigger window: steps 45..80 OR if avg G loss already large.
#                    try:
#                        avg_g = float(run_g / max(1, nb))
#                    except Exception:
#                        avg_g = 0.0
#                    if (45 <= (step + 1) <= 80) or (avg_g > 1.0):
#                        try:
                            # IMPORTANT: do not call `_compute_terms()` eagerly in a hot loop.
                            # It contains multiple `tf.cond(...)` blocks whose branch functions are
                            # created inside the call, which can cause graph/kernels-cache growth.
                            # Use the cached `tf.function` wrapper instead.
#                            terms = net_g.train_terms(lr, hr, pod_coeffs=pod_coeffs)
                            # terms is dict[str, tensor]; print a compact subset
#                            def tget(k):
#                                v = terms.get(k, None)
#                                return _to_float(v, default=0.0)
#                            print("[adv][terms] "
#                                  f"pix {tget('pix'):.3e} res {tget('res'):.3e} "
#                                  f"spec {tget('spec'):.3e} espec {tget('espec'):.3e} "
#                                  f"lowk {tget('lowk'):.3e} grad {tget('grad'):.3e} "
#                                  f"cov {tget('cov'):.3e} decor {tget('decor'):.3e} "
#                                  f"gan {tget('gan'):.3e} tv {tget('tv'):.3e}")

                            # Also print input ranges; bad batches usually show up here first.
#                            print("[adv][batch] " + _batch_stats(lr, "lr") + " | " + _batch_stats(hr, "hr"))
#                        except Exception as e:
#                            print(f"[adv][terms] debug failed: {type(e).__name__}: {e}")
            # Validate + steer
            vm = _validate_epoch(
                val_loader,
                net_g,
                stats.hr_mean_np,
                stats.hr_std_np,
                stats.lr_mean_np,
                stats.lr_std_np,
                phys_params=phys_params,
                pod_params=pod_params_tf,
                score_weights=asdict(adv_w_scaled),
                M_STACK=M_STACK,
                WEIGHTS=WEIGHTS,
                DEN_BINS=DEN_BINS,
                LOW_MASK=LOW_MASK,
            )

            # Instance noise
            s = max(noise_at_epoch(epoch), 0.0)
            net_g.set_inst_noise(s); net_d.set_inst_noise(s)

            # Steer LRs / adversarial pressure using composite validation score
            steerer.update(vm, epoch=epoch, g_loss_epoch=(run_g / max(1, nb)), d_loss_epoch=(run_d / max(1, nb)))
            # After steering, print live knobs again so logs reflect what was actually applied
            print(f"[adv] ep {epoch:04d} steer-applied (live): "
                  f"g_lr {_opt_lr(g_opt_adv):.3e} d_lr {_opt_lr(d_opt_adv):.3e} | "
                  f"{_fmt_knobs(net_g)}")
            # Early stopping on composite validation score (higher is better)
            vscore = float(vm.get('val_score', float('-inf')))
            mp = float(vm.get("val_mae_p", float("nan")))
            mv = float(vm.get("val_mae_v", float("nan")))
            mw = float(vm.get("val_mae_w", float("nan")))
            vpix = vm.get("val_pix", float("nan"))
            vspec = vm.get("val_spec", float("nan"))
            vespec = vm.get("val_espec", float("nan"))
            vres = float(vm.get("val_res", float("nan")))
            vtv = float(vm.get("val_tv", float("nan")))
            vcov = float(vm.get("val_cov", float("nan")))
            vdecor = float(vm.get("val_decor", float("nan")))
            vgrad = float(vm.get("val_grad", float("nan")))
            vlowk = float(vm.get("val_lowk", float("nan")))
            vfm = float(vm.get("val_fm", float("nan")))
            vgan = vm.get("val_gan", float("nan"))
            vpod = vm.get("val_pod", float("nan"))
            if (vscore - best_score_adv) > EARLY_STOP_MIN_DELTA:
                best_score_adv = vscore
                es_wait_adv = 0
                if epoch < 10:
                    best_score_adv = float('-inf')
                # Save best adv checkpoint (higher val_score is better)
                if cfg.save_best:
                    best_dir = os.path.join(cfg.ckpt_dir, cfg.best_ckpt_adv_name)
                    _save_checkpoint(
                        best_dir, "adv_best", epoch, G, D, ema,
                        input_shape=(cfg.batch_size,) + lr_shape,
                        g_lr=_lr_value(g_opt_adv, cfg.g_lr_adv),
                        d_lr=_lr_value(d_opt_adv, cfg.d_lr_adv),
                        state={
                            'cfg': asdict(cfg),
                            'epoch': epoch,
                            'val_score': float(vscore),
                            'val_mae': float(vm.get('val_mae', float('nan'))),
                            # Persist steerer internals so resume does not restart warmups/cooldowns.
                            'steerer_state': steerer.state_dict() if hasattr(steerer, 'state_dict') else {},
                            # Also persist the adv weights actually in effect.
                            'adv': {
                                'eff': float(getattr(net_g, 'adv_weight', 0.0)),
                                'sched': float(getattr(steerer, 'scheduled_adv_w', 0.0)),
                            },
                        }
                    )
            else:
                es_wait_adv += 1
                if es_wait_adv >= EARLY_STOP_PATIENCE:
                    print(f"[adv] early stop: val_score did not improve for {EARLY_STOP_PATIENCE} validations (best {best_score_adv:.6f})")
                    break

            # Save ckpt
            if (epoch % cfg.save_every == 0) or (epoch == cfg.adv_epochs):

                save_dir = os.path.join(cfg.ckpt_dir, f"ckpt_adv_ep{epoch:04d}")

                _save_checkpoint(
                             save_dir, "adv", epoch, G, D, ema,
                             # Generator input is LR (N,C,H,W). Store LR shape in meta so
                             # inference scripts can build the model correctly.
                             input_shape=(cfg.batch_size,)+lr_shape,
                             g_lr=_lr_value(g_opt_adv, cfg.g_lr_adv),
                             d_lr=_lr_value(d_opt_adv, cfg.d_lr_adv),
                             state={
                                 'cfg': asdict(cfg),
                                 'steerer_state': steerer.state_dict() if hasattr(steerer, 'state_dict') else {},
                                 'adv': {
                                     'eff': float(getattr(net_g, 'adv_weight', 0.0)),
                                     'sched': float(getattr(steerer, 'scheduled_adv_w', 0.0)),
                                 },
                             })
            print(
                f"[adv] epoch {epoch} done in {time.time()-t0:.1f}s | "
                f"val_score {vm.get('val_score', float('nan')):.6f} "
                f"losses: pix {vpix:.3e} | res {vres:.3e} | spec {vspec:.3e} | espec {vespec:.3e} | tv {vtv:.3e} | cov {vcov:.3e} | pod {vpod:.3e} | gan {vgan:.3e} | lowk {vlowk:.3e} | grad {vgrad:.3e} | decor {vdecor:.3e} " #| fm {vfm:.3e} "
                f"val_mae {vm.get('val_mae', float('nan')):.4f}  (p {mp:.4f} v {mv:.4f} w {mw:.4f}) (denorm {vm.get('val_mae_denorm', float('nan')):.2f}) "
                f"psnr {vm.get('val_psnr', float('nan')):.3f} "
                f"bleed {vm.get('edge_bleed', float('nan')):.4g} "
            )
