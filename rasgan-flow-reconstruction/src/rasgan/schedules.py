from __future__ import annotations

import math
from typing import Optional

from .config import (
    ADV_MIN, ADV_MAX, ADV_INIT,
    EDGE_MODE, EDGE_INIT, EDGE_MIN, EDGE_DECAY_EPOCHS,
)


def adv_weight_schedule(
    epoch: int,
    total_adv_epochs: int,
    steerer=None,
    vm=None,
    adv_min: float = ADV_MIN,
    adv_max: float = ADV_MAX,
) -> float:
    """Adversarial weight schedule.

    Ramps from ADV_INIT to ADV_MAX over the first 20 adversarial epochs, then
    holds at ADV_MAX (before optional steerer/guard adjustments).
    """
    if total_adv_epochs <= 0:
        return adv_min

    # Clamp ADV_INIT into [adv_min, adv_max] first.
    adv_init = float(min(adv_max, max(adv_min, ADV_INIT)))

    ramp_epochs = 40
    if ramp_epochs <= 1:
        base = float(adv_max)
    elif epoch <= ramp_epochs:
        # epoch=1 -> adv_init, epoch=ramp_epochs -> adv_max
        t = float(epoch - 1) / float(ramp_epochs - 1)
        base = float(adv_init + t * (adv_max - adv_init))
    else:
        base = float(adv_max)

    # Global clamp to keep within configured limits.
    base = float(min(adv_max, max(adv_min, base)))

    if (steerer is not None) and (vm is not None) and hasattr(steerer, "plan_next_adv_weight"):
        return float(steerer.plan_next_adv_weight(base, epoch, vm, extra_decay_mul=1.0))
    return float(base)


def edge_lambda_at_epoch(epoch: int, edge_start_epoch: int) -> float:
    if EDGE_MODE == "off":
        return 0.0
    if EDGE_MODE == "on":
        return float(EDGE_INIT)

    # anneal
    if epoch <= edge_start_epoch:
        return float(EDGE_INIT)
    t = (epoch - edge_start_epoch) / max(1, EDGE_DECAY_EPOCHS)
    if t >= 1.0:
        return float(EDGE_MIN)
    return float(EDGE_INIT + (EDGE_MIN - EDGE_INIT) * t)

def noise_at_epoch(e):
    # hold ~0.008 through 45 epochs, then taper to 0 by 80
    if e <= 45: return 0.0008
    if e <= 80: return 0.0008 * (1 - (e-45)/35)
    return 0.0

def pod_vel_blend_alpha(epoch: int, ramp_epochs: int = 40) -> float:
    r = max(1, int(ramp_epochs))
    if r <= 1: return 0.0
    if epoch <= 1: return 1.0
    if epoch >= r: return 0.0
    t = float(epoch - 1) / float(r - 1)
    return float(1.0 - t)

from .config import LossWeights

def fft_deramp_factor(epoch: int, deramp_epochs: int = 80) -> float:
    """Multiplier in [0,1] to *deramp* FFT-based spectral weights to 0.

    This is intended to gradually turn off spec/espec/lowk penalties over the
    first `deramp_epochs` adversarial epochs (epoch is 1-indexed).

    - epoch=1   -> 1.0
    - epoch=deramp_epochs -> 0.0
    - epoch>deramp_epochs -> 0.0
    """
    r = int(max(1, deramp_epochs))
    if r <= 1:
        return 0.0
    if epoch <= 1:
        return 1.0
    if epoch >= r:
        return 0.0
    t = float(epoch - 1) / float(r - 1)  # 0..1
    return float(max(0.0, min(1.0, 1.0 - t)))

def physics_ramp_factor(epoch: int, ramp_epochs: int = 20) -> float:
    """Multiplier in [0,1] to ramp physics weights (epoch is 1-indexed in adv)."""
    ramp_epochs = int(max(1, ramp_epochs))
    if ramp_epochs <= 1:
        return 1.0
    t = 0.0 if epoch <= 1 else float(epoch - 1) / float(ramp_epochs - 1)
    return float(min(1.0, max(0.0, t)))


def scale_physics_weights(w: LossWeights, factor: float) -> LossWeights:
    """Scale only physics terms of LossWeights by `factor` (clamped to >=0)."""
    f = float(max(0.0, factor))
    return LossWeights(
        pix_w=w.pix_w,
        grad_w=w.grad_w,
        tv_w=w.tv_w,
        w_spec=w.w_spec,
        w_espec=w.w_espec,
        w_res=w.w_res,
        w_low=w.w_low,
        w_vomega=float(max(0.0, w.w_vomega * f)),
        w_omcons=float(max(0.0, w.w_omcons * f)),
        w_div=float(max(0.0, w.w_div * f)),
        w_mom=float(max(0.0, w.w_mom * f)),
        w_ppois=float(max(0.0, w.w_ppois * f)),
    )
