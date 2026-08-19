"""Project configuration.

This module keeps *defaults* close to the original single-file script,
but training code should prefer the dataclasses below (TrainConfig, etc.)
instead of mutating globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List


# ------------------------ Default training schedule ------------------------ #
N_EPOCH_INIT_DEFAULT: int = 100
N_EPOCH_ADV_DEFAULT: int = 300
BATCH_SIZE_DEFAULT: int = 8

VAL_EVERY_DEFAULT: int = 1
PRINT_EVERY_STEPS_DEFAULT: int = 10

# ------------------------ Model hyperparameters ------------------------ #
RRDB_DEPTH: int = 6
D_DIMS: int = 32  # base channel width for discriminator

# ------------------------ Pretrain knobs ------------------------ #
PRETRAIN_P_UPWEIGHT: float = 1.5
PRETRAIN_W_LP_DELTA_P: float = 0.0
PRETRAIN_LP_KERNEL: int = 11

# ------------------------ Adversarial weight schedule ------------------------ #
ADV_INIT: float = 5e-7
ADV_MIN: float = 1e-7
ADV_MAX: float = 1e-6
ADV_WARMUP_NO_DECREASE: int = 40

# ------------------------ Content / regularizers ------------------------ #
TV_W_DEFAULT: float = 1e-4
CHARB_EPS: float = 1e-3

W_CONTENT_DEFAULT = 0.001
W_RES_DEFAULT: float = 0.2
W_LOW_DEFAULT: float = 0.0005
LOW_BINS: int = 8

# Spectrum targets and ramp
W_SPEC_TARGET: float = 0.0005
W_ESPEC_TARGET: float = 0.00025
ESPEC_ALPHA: float = 1.0
ENABLE_SPEC_RAMP: bool = True
SPEC_RAMP_EPOCHS: int = 40

# Per-channel pixel weights [P, v, ωz] (adv stage default)
PIX_W_DEFAULT: Tuple[float, float, float] = (1.0, 1.0, 1.0)

# R1 regularization (0 during warmup in script; scheduled in training)
R1_GAMMA_BASE: float = 0.25

# Instance noise base
INST_NOISE_INIT: float = 0.005

# Master LR bands
G_MIN_LR: float = 5e-7
G_MAX_LR: float = 5e-4
G_LR_INIT: float = 1e-5
D_LR_INIT: float = 2e-6
D_STABLE_MIN: float = 1e-7
D_STABLE_MAX: float = 5e-5

# Temporary D LR increase when D loss trends upward (but not yet above ceiling)
D_TREND_WINDOW: int = 4          # epochs to look back for trend detection
D_TREND_MIN_DELTA: float = 0.005  # minimum avg increase per epoch to count as rising
D_TREND_LR_INCREASE_FACTOR: float = 1.05  # multiplicative bump when trend detected
D_TREND_COOLDOWN: int = 4        # epochs to wait between successive bumps

LR_PLATEAU_PATIENCE: int = 20
LR_REDUCE_FACTOR: float = 0.6
LR_INCREASE_FACTOR: float = 1.4

EARLY_STOP_PATIENCE: int = 200
EARLY_STOP_MIN_DELTA: float = 1e-4

PSNR_EPS: float = 0.05
ADV_PATIENCE: int = 4
ADV_COOLDOWN: int = 4
ADV_WINDOW: int = 4
STEER_LIMIT: float = 0.10

# Output
SAVE_DIR_DEFAULT: str = "outputs"
CKPT_DIR_DEFAULT: str = "checkpoints"
DATA_PATH_DEFAULT: str = "data/flow_fields.h5"

# ------------------------ Edge / conditioning controls ------------------------ #
EDGE_MODE: str = "anneal"  # 'on' | 'off' | 'anneal'
EDGE_INIT: float = 0.2
EDGE_MIN: float = 0.01
EDGE_DECAY_EPOCHS: int = 40

D_CLF_TARGET_LOW: float = 0.6
D_CLF_TARGET_HIGH: float = 1.5
ADA_NOISE_MAX: float = 0.02

COND_DROP_P_INIT: float = 0.60
COND_DROP_P_MIN: float = 0.30
COND_DROP_P_MAX: float = 0.90

EDGE_TANH_GAIN: float = 1.0
COND_DROP_P: float = 0.35
D_SOLO_CHAN_P: float = 0.35
COV_W = 8e-5
DECORR_W = 6e-5
LP_W = 0.0

def d_in_channels(use_coords: bool = False, base_channels: int = 12) -> int:
    """D input channels: [x, lr, residual, learned-edges] = 12; +2 coord maps optionally."""
    return int(base_channels + (2 if use_coords else 0))

@dataclass(frozen=True)
class LossWeights:
    content_w: float = W_CONTENT_DEFAULT
    pix_w: Tuple[float, float, float] = PIX_W_DEFAULT
    grad_w: float = 0.002
    tv_w: float = TV_W_DEFAULT
    w_spec: float = W_SPEC_TARGET
    w_espec: float = W_ESPEC_TARGET
    w_res: float = W_RES_DEFAULT
    w_low: float = W_LOW_DEFAULT
    # Physics-informed terms (set to >0 to enable)
    w_vomega: float = 0.0   # ||v_pred - v(ω_pred)||
    w_omcons: float = 0.0   # ||ω_pred - curl(u(ω_pred), v_pred)||
    w_div: float = 0.0      # ||div(u(ω_pred), v_pred)||
    w_mom: float = 0.0      # Navier–Stokes momentum residual
    w_ppois: float = 0.0    # Pressure Poisson residual

    # POD sidecar consistency terms (set >0 to enable)
    # Operate in LR space (e.g. 96x96) where POD was computed.
    # - vel: velocity POD cycle (u inferred from omega + v) + coefficient match
    # - p:   pressure POD cycle + coefficient match
    # - w:   vorticity consistency between POD-reconstructed (u,v) and LR omega
    w_pod_vel: float = 0.0
    w_pod_p: float = 0.0
    w_pod_w: float = 0.0

@dataclass(frozen=True)
class TrainConfig:
    data_path: str = DATA_PATH_DEFAULT
    ckpt_dir: str = CKPT_DIR_DEFAULT
    save_dir: str = SAVE_DIR_DEFAULT

    # Runtime toggles
    mixed: bool = False
    xla: bool = False
    tlx_verbose: bool = False  # retained name for backward-compatible checkpoint configs
    # Grid / physics parameters (used by physics-informed losses)
    dx: float = 0.0000265
    dy: float = 0.0000265
    nu: float = 0.000066
    rho: float = 0.5
    bc: str = "replicate"          # 'replicate' or 'periodic'
    poisson_method: str = "jacobi"    # 'fft' (periodic) or 'jacobi'
    poisson_iters: int = 100

    # Training schedule
    batch_size: int = BATCH_SIZE_DEFAULT
    val_every: int = VAL_EVERY_DEFAULT
    save_every: int = 10  # save checkpoints every N epochs
    # Save "best" checkpoints whenever validation improves
    save_best: bool = True
    best_ckpt_init_name: str = "best_init"  # best during init phase (lowest val_mae)
    best_ckpt_adv_name: str = "best_adv"    # best during adversarial phase (highest val_score)
    print_every_steps: int = PRINT_EVERY_STEPS_DEFAULT
    init_epochs: int = N_EPOCH_INIT_DEFAULT
    adv_epochs: int = N_EPOCH_ADV_DEFAULT
    # Super-resolution scale factor
    scale: int = 2
    # Generator upsampling: 'resizeconv' (bilinear+conv) or 'pixelshuffle' (ICNR+blur)
    g_upsample: str = 'resizeconv'

    # Generator architecture
    # - 'rrdb' (default): RRDB-style SRGAN_g
    # - 'transformer': TransformerSR_g (2× SR)
    g_arch: str = 'rrdb'

    # Transformer generator hyperparams (used when g_arch='transformer')
    g_patch: int = 8
    g_embed_dim: int = 192
    g_depth: int = 10
    g_heads: int = 6
    g_mlp_ratio: float = 4.0
    g_dropout: float = 0.1
    g_use_film: bool = True
    # If provided, overrides coeff_dim used to condition the transformer.
    # Default: 3 * pod_k (concatenated tim_u, tim_v, tim_p)
    g_coeff_dim: Optional[int] = None

    # POD sidecar (optional): MATLAB .mat file containing POD modes and (optionally)
    # per-sample coefficients aligned with the training set order.
    pod_mat_path: Optional[str] = None
    pod_k: int = 5

    # LRs
    g_lr_i: float = G_LR_INIT
    g_lr_adv: float = G_LR_INIT
    d_lr_adv: float = D_LR_INIT
    d_band_min: float = D_STABLE_MIN
    d_band_max: float = D_STABLE_MAX

    # AdamW (decoupled weight decay)
    # - Applied in both init and adv stages.
    # - Default keeps D decay off (common in GANs).
    g_weight_decay: float = 1e-4
    d_weight_decay: float = 0.0

    # Resume
    resume_stage: Optional[str] = None  # 'init' | 'adv' | None
    resume_from: Optional[str] = None   # path to ckpt dir or meta.json
    resume_epoch: int = 0

    # ---------------- Gradient-magnitude loss reweighting (adv stage, G only) ----------------
    # When enabled, selected generator loss terms are dynamically rescaled so their
    # EMA-smoothed gradient norms w.r.t. G weights are similar. This helps prevent
    # one term dominating while others steadily worsen.
    grad_reweight: bool = False
    # Loss-term keys taken from WithLoss_G._compute_terms().
    # NOTE: 'gan' is intentionally excluded by default (reweighting GAN pressure can
    # destabilize training).
    grad_reweight_keys: Tuple[str, ...] = (
        "pix",
        "res",
        "grad",
        "spec",
        "espec",
        "lowk",
        "tv",
        "decor",
        "cov",
#        "lp",
#        "phys",
#        "pod",
#        "fm",
    )
    # Apply reweighting update every N G steps (1 = every step). Larger saves compute/memory churn.
    grad_reweight_every: int = 1
    # Use deterministic term computation for grad-norm estimation (no inst noise/dropout).
    grad_reweight_deterministic_norms: bool = False
    # EMA factor for gradient norms.
    grad_reweight_ema: float = 0.95
    # Power/exponent in multiplier: alpha=(target/(gn+eps))**power
    grad_reweight_power: float = 0.5
    # Clamp range for alpha: [1/clip, clip]
    grad_reweight_clip: float = 10.0
    grad_reweight_eps: float = 1e-8

    # Stage weights
    init_weights: LossWeights = field(default_factory=lambda: LossWeights(
        pix_w=(PRETRAIN_P_UPWEIGHT, 1.0, 1.0),
        grad_w=0.0,
        tv_w=5e-5,
        w_res=0.0,
        w_low=PRETRAIN_W_LP_DELTA_P,
        content_w=1.0,
    ))
    adv_weights: LossWeights = field(default_factory=lambda: LossWeights())
