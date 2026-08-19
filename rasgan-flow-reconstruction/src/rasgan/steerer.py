from __future__ import annotations

import os
import math
import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import (
    G_MIN_LR, G_MAX_LR, D_STABLE_MIN, D_STABLE_MAX, STEER_LIMIT,
    ADV_MIN, ADV_MAX, ADV_PATIENCE, ADV_WINDOW, ADV_COOLDOWN,
    LR_PLATEAU_PATIENCE, LR_REDUCE_FACTOR, LR_INCREASE_FACTOR,
    EARLY_STOP_MIN_DELTA, PSNR_EPS, ADV_WARMUP_NO_DECREASE, ADV_INIT,
    D_TREND_WINDOW, D_TREND_MIN_DELTA, D_TREND_LR_INCREASE_FACTOR,
    D_TREND_COOLDOWN,
)

class ValSteerer:
    def __init__(self, g_opt, d_opt, withloss_g, withloss_d,
                 init_g_lr, init_d_lr,
                 g_min=G_MIN_LR, g_max=G_MAX_LR,
                 d_min=D_STABLE_MIN, d_max=D_STABLE_MAX,
                 factor_down=LR_REDUCE_FACTOR, factor_up=LR_INCREASE_FACTOR,
                 patience=LR_PLATEAU_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA,
                 psnr_eps=PSNR_EPS, adv_patience=ADV_PATIENCE, adv_cooldown=ADV_COOLDOWN, adv_window=ADV_WINDOW,
                 no_decrease_until=ADV_WARMUP_NO_DECREASE, steer_limit=STEER_LIMIT):
        # [PATCH][Steerer] Freeze D LR flag for ADA
        self.hold_d = False
        self.g_opt, self.d_opt = g_opt, d_opt
        self.withloss_g = withloss_g; self.withloss_d = withloss_d
        self.g_lr = float(getattr(g_opt, 'learning_rate', getattr(g_opt, 'lr', init_g_lr)))
        self.d_lr = float(getattr(d_opt, 'learning_rate', getattr(d_opt, 'lr', init_d_lr)))
        self.g_min, self.g_max = g_min, g_max
        self.d_min, self.d_max = d_min, d_max
        self.factor_down, self.factor_up = factor_down, factor_up
        # First-time steeper drop for "D too strong"
        self.d_first_strong_factor = 0.25   # steeper one-time drop (e.g., 0.25)
        self.d_first_strong_drop_done = False; self.patience, self.min_delta = int(patience), float(min_delta)
        self.best = np.inf; self.wait = 0
        self.psnr_eps=float(psnr_eps); self.adv_patience=int(adv_patience); self.adv_cooldown=int(adv_cooldown)
        self.adv_wait=0; self.epochs_since_adv_change=10**9
        self.psnr_hist = collections.deque(maxlen=max(3,int(adv_window)))
        # --- [NEW] tiny histories for metric-aware gating ---
        self.h1_hist       = collections.deque(maxlen=max(3, int(adv_window)))
        self.spec_hi_hist  = collections.deque(maxlen=max(3, int(adv_window)))
        self.edge_hist     = collections.deque(maxlen=max(3, int(adv_window)))
        self.psnr_p05_hist = collections.deque(maxlen=max(3, int(adv_window)))
        self.mae_p95_hist  = collections.deque(maxlen=max(3, int(adv_window)))
        # [NEW] track covariance off-diagonal metric from validation
        self.cov_hist      = collections.deque(maxlen=max(3, int(adv_window)))
        # thresholds (small, conservative)
        self.h1_eps      = 1e-3; self.spec_hi_eps = 1e-2
        self.edge_eps    = 1e-4; self.cov_eps     = 1e-5
        # Primary metric selection: "score" (composite) or "psnr"
        self.primary = os.environ.get("VAL_PRIMARY", "score")
        self.score_hist = collections.deque(maxlen=max(3, int(adv_window)))
        self.score_eps = 0.01  # ~1% movement considered meaningful
        self.prev_psnr=None; self.prev_pix=None; self.epoch=0;
        # Adv weight schedule (current + next-epoch handoff)
        #
        # Training loop reads `steerer.adv_w_sched` to carry a drift-guard/steering adjustment
        # forward to the next epoch. Keep `scheduled_adv_w` as the internal name used by
        # this steerer, but also expose `adv_w_sched` as an alias for compatibility.
        self.scheduled_adv_w = float(ADV_INIT)
        self.adv_w_sched = float(ADV_INIT)
        self.steer_limit = float(steer_limit)
        self.d_floor = 0.2; self.d_ceil  = 1.8; self.stall_eps = 0.15
        self.stall_patience = 3; self._stall_cnt = 0
        self._apply_g_lr(self.g_lr); self._apply_d_lr(self.d_lr)
        # Gate G-LR up-bumps; and require a meaningful relative improvement if enabled
        self.g_lr_up_enabled = True
        self.g_up_rel_delta = 0.05  # 5% relative improvement threshold for LR up-bump (only used when enabled)
        # [NEW] keep a baseline and cooldown for structural penalties (COV_W)
        try:
            self.cov_base = float(self.withloss_g.get_cov_weight())
        except Exception:
            self.cov_base = float(os.environ.get("COV_W", 0.0))
        self.cov_floor = 0.25 * self.cov_base  # never decay below this
        self.cov_cap   = float(os.environ.get("COV_CAP", 10.0))  # safety cap (× from base)
        self.epochs_since_penalty_change = 10**9; self.penalty_wait = 0
        self.penalty_patience = max(2, int(adv_window) // 2)
        self.penalty_cooldown = int(os.environ.get("PENALTY_COOLDOWN", 5))
        # --- Stability knobs (caps + streaks) ---
        # Require consecutive "weak/strong" signals before changing D LR
        self.d_streak_req = int(os.environ.get("D_STREAK_REQ", 3))
        # --- Temporary D LR trend bump (separate from ceiling-based logic) ---
        self.d_trend_window = int(os.environ.get("D_TREND_WINDOW", D_TREND_WINDOW))
        self.d_trend_min_delta = float(os.environ.get("D_TREND_MIN_DELTA", D_TREND_MIN_DELTA))
        self.d_trend_lr_increase_factor = float(os.environ.get("D_TREND_LR_INCREASE_FACTOR", D_TREND_LR_INCREASE_FACTOR))
        self.d_trend_cooldown = int(os.environ.get("D_TREND_COOLDOWN", D_TREND_COOLDOWN))
        self._d_trend_cooldown_rem = 0
        self._d_epoch_hist = collections.deque(maxlen=max(2, self.d_trend_window))

        self._d_weak_streak = 0; self._d_strong_streak = 0
        # Hard cap on adversarial pressure (below global ADV_MAX)
        self.adv_cap   = float(os.environ.get("ADV_CAP", 1e-5))
        self.adv_decay = float(os.environ.get("ADV_DECAY", 0.5))
        # Absolute drift thresholds to trigger extra caution
        self.bleed_hi     = float(os.environ.get("BLEED_HI", 0.0004))
        self.spec_hi_bad  = float(os.environ.get("SPEC_HI_BAD", 0.85))
        # --- Epoch-aware adversarial guard state (replaces _apply_adv_guard/* helpers) ---
        self._spec_ema = None; self._bad_streak = 0; self.warmup_epochs = int(no_decrease_until) # no clamping before this; matches the old helper

    # ---------------------------------------------------------------------
    # Checkpointing helpers
    #
    # This steerer is stateful (EMA stats, cooldown counters, short-window
    # histories, etc.). If we don't persist that state, a resume will restart
    # its internal warmups and periodic adjustments, which can yield
    # repeatable "blow-ups" after a fixed number of epochs.
    #
    # We keep this as pure-Python (JSON-serializable) state because TF/Keras
    # checkpoints do not automatically capture deques / numpy scalars.
    # ---------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the steerer state."""

        def _flt(x, default=None):
            if x is None:
                return default
            try:
                return float(x)
            except Exception:
                return default

        def _int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        def _lst(dq):
            try:
                return list(dq)
            except Exception:
                return []

        return {
            # core
            "epoch": _int(self.epoch),
            # JSON does not have NaN/Inf; store None and reconstruct on load.
            "best": (_flt(self.best, None) if (self.best is not None and np.isfinite(self.best)) else None),
            "wait": _int(self.wait),
            "g_lr": _flt(self.g_lr, 0.0),
            "d_lr": _flt(self.d_lr, 0.0),
            "hold_d": bool(getattr(self, "hold_d", False)),

            # adv schedule + gating
            "scheduled_adv_w": _flt(getattr(self, "scheduled_adv_w", ADV_INIT), ADV_INIT),
            "adv_w_sched": _flt(getattr(self, "adv_w_sched", ADV_INIT), ADV_INIT),
            "adv_wait": _int(getattr(self, "adv_wait", 0)),
            "epochs_since_adv_change": _int(getattr(self, "epochs_since_adv_change", 0)),
            "prev_psnr": _flt(getattr(self, "prev_psnr", None), None),
            "prev_pix": _flt(getattr(self, "prev_pix", None), None),

            # penalty tuning
            "epochs_since_penalty_change": _int(getattr(self, "epochs_since_penalty_change", 0)),
            "_d_trend_cooldown_rem": _int(getattr(self, "_d_trend_cooldown_rem", 0)),
            "_d_weak_streak": _int(getattr(self, "_d_weak_streak", 0)),
            "_d_strong_streak": _int(getattr(self, "_d_strong_streak", 0)),
            "d_first_strong_drop_done": bool(getattr(self, "d_first_strong_drop_done", False)),

            # drift guard
            "_spec_ema": _flt(getattr(self, "_spec_ema", None), None),
            "_bad_streak": _int(getattr(self, "_bad_streak", 0)),

            # short-window histories
            "psnr_hist": _lst(getattr(self, "psnr_hist", [])),
            "score_hist": _lst(getattr(self, "score_hist", [])),
            "h1_hist": _lst(getattr(self, "h1_hist", [])),
            "spec_hi_hist": _lst(getattr(self, "spec_hi_hist", [])),
            "edge_hist": _lst(getattr(self, "edge_hist", [])),
            "cov_hist": _lst(getattr(self, "cov_hist", [])),
            "psnr_p05_hist": _lst(getattr(self, "psnr_p05_hist", [])),
            "mae_p95_hist": _lst(getattr(self, "mae_p95_hist", [])),
            "_d_epoch_hist": _lst(getattr(self, "_d_epoch_hist", [])),
        }

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        """Restore steerer state from a prior `state_dict()` snapshot."""
        if not isinstance(st, dict):
            return

        def _set_float(attr: str, default: float = 0.0):
            if attr in st:
                try:
                    setattr(self, attr, float(st[attr]))
                except Exception:
                    setattr(self, attr, default)

        def _set_int(attr: str, default: int = 0):
            if attr in st:
                try:
                    setattr(self, attr, int(st[attr]))
                except Exception:
                    setattr(self, attr, default)

        def _set_bool(attr: str, default: bool = False):
            if attr in st:
                try:
                    setattr(self, attr, bool(st[attr]))
                except Exception:
                    setattr(self, attr, default)

        # core
        _set_int("epoch", 0)
        if "best" in st:
            try:
                self.best = np.inf if st["best"] is None else float(st["best"])
            except Exception:
                self.best = np.inf
        _set_int("wait", 0)
        _set_float("g_lr", self.g_lr)
        _set_float("d_lr", self.d_lr)
        _set_bool("hold_d", False)

        # adv gating
        _set_float("scheduled_adv_w", self.scheduled_adv_w)
        _set_float("adv_w_sched", self.adv_w_sched)
        _set_int("adv_wait", 0)
        _set_int("epochs_since_adv_change", 0)
        if "prev_psnr" in st:
            self.prev_psnr = None if st["prev_psnr"] is None else float(st["prev_psnr"])
        if "prev_pix" in st:
            self.prev_pix = None if st["prev_pix"] is None else float(st["prev_pix"])

        # penalties / streaks
        _set_int("epochs_since_penalty_change", 0)
        _set_int("_d_trend_cooldown_rem", 0)
        _set_int("_d_weak_streak", 0)
        _set_int("_d_strong_streak", 0)
        _set_bool("d_first_strong_drop_done", False)

        # drift guard
        if "_spec_ema" in st:
            self._spec_ema = None if st["_spec_ema"] is None else float(st["_spec_ema"])
        _set_int("_bad_streak", 0)

        # histories (rebuild deques with correct maxlen)
        def _restore_deque(name: str, values):
            if not hasattr(self, name):
                return
            dq = getattr(self, name)
            try:
                dq.clear()
                for v in (values or []):
                    dq.append(float(v))
            except Exception:
                pass

        _restore_deque("psnr_hist", st.get("psnr_hist"))
        _restore_deque("score_hist", st.get("score_hist"))
        _restore_deque("h1_hist", st.get("h1_hist"))
        _restore_deque("spec_hi_hist", st.get("spec_hi_hist"))
        _restore_deque("edge_hist", st.get("edge_hist"))
        _restore_deque("cov_hist", st.get("cov_hist"))
        _restore_deque("psnr_p05_hist", st.get("psnr_p05_hist"))
        _restore_deque("mae_p95_hist", st.get("mae_p95_hist"))
        _restore_deque("_d_epoch_hist", st.get("_d_epoch_hist"))

        # re-apply LR values to optimizers (important after resume)
        try:
            self._apply_g_lr(self.g_lr)
            self._apply_d_lr(self.d_lr)
        except Exception:
            pass

    def _apply_g_lr(self, lr):
        if hasattr(self.g_opt,'learning_rate'): self.g_opt.learning_rate=float(lr)
        elif hasattr(self.g_opt,'lr'): self.g_opt.lr=float(lr)
    def hold_d_lr(self, on: bool):
        # [PATCH][Steerer] Toggle LR decrease for D
        self.hold_d = bool(on)

    def _apply_d_lr(self, lr):
        if hasattr(self.d_opt,'learning_rate'): self.d_opt.learning_rate=float(lr)
        elif hasattr(self.d_opt,'lr'): self.d_opt.lr=float(lr)
    def set_schedule(self, scheduled_adv_w):
        # Base (scheduled) adversarial weight for the current epoch (ramp target).
        self.scheduled_adv_w = float(scheduled_adv_w)
        # adv_w_sched is interpreted by the training loop as a *cap* produced by guards
        # (e.g., drift guard). Keep it at ADV_MAX unless a guard actually reduces adv_w
        # in `step()`.
        self.adv_w_sched = float(ADV_MAX)
    def set_g_lr_up_enabled(self, on: bool, rel_delta: float = None):
        self.g_lr_up_enabled = bool(on)
        if rel_delta is not None:
            self.g_up_rel_delta = float(rel_delta)
    def set_d_regime(self, floor, ceil, d_min, d_max):
        self.d_floor = float(floor); self.d_ceil=float(ceil)
        self.d_min=float(d_min); self.d_max=float(d_max)
    def clamp_d_to_band(self):
        new_d = min(self.d_max, max(self.d_min, self.d_lr))
        if new_d != self.d_lr:
            self.d_lr = new_d; self._apply_d_lr(self.d_lr)
    # ---- epoch-aware caps/thresholds (was _adv_caps_for_epoch) ----
    def _refresh_epoch_caps(self, epoch: int):
        if epoch < 60:
            self.adv_cap, self.spec_hi_bad, self.bleed_hi, self.adv_decay = 1.0e-5, 0.850, 4.0e-4, 0.80
        elif epoch < 120:
            self.adv_cap, self.spec_hi_bad, self.bleed_hi, self.adv_decay = 7.0e-6, 0.790, 2.5e-4, 0.70
        else:
            self.adv_cap, self.spec_hi_bad, self.bleed_hi, self.adv_decay = 4.0e-6, 0.700, 1.5e-4, 0.60

    # ---- schedule planner for NEXT epoch (replaces _apply_adv_guard) ----
    def plan_next_adv_weight(self, base_next: float, epoch: int, vm: dict, extra_decay_mul: float = 1.0) -> float:
        """Return the next-epoch scheduled adv weight from a base value:
           - refresh per-epoch caps/thresholds
           - update EMA(spec_relerr_high)
           - 2-epoch patience with bleed+high-k
           - apply any extra decay from external guardrails (D too strong, collapse)
           - clamp to per-epoch cap then global clamps
        """
        self._refresh_epoch_caps(int(epoch))

        # EMA(spec_relerr_high)
        m = float(vm.get('spec_relerr_high', 0.0))
        if self._spec_ema is None:
            self._spec_ema = m
        self._spec_ema = 0.9 * self._spec_ema + 0.1 * m

        # patience on "bad" (high-k EMA above threshold AND bleed above threshold)
        bad = (self._spec_ema > self.spec_hi_bad) and (float(vm.get('edge_bleed', 0.0)) > self.bleed_hi)
        self._bad_streak = (self._bad_streak + 1) if bad else 0

        aw = float(base_next)
        if (epoch >= self.warmup_epochs) and (self._bad_streak >= 2):
            aw = max(ADV_MIN, self.adv_decay * aw)  # soften if persistently bad

        # apply any external decay (e.g., D too strong, collapse guard)
        aw *= float(extra_decay_mul)

        # local epoch cap then global clamps
        aw = min(self.adv_cap, aw)
        aw = min(ADV_MAX, max(ADV_MIN, aw))
        return aw

    def update(self, vm: Dict[str, Any], *, epoch: int, g_loss_epoch: Optional[float] = None, d_loss_epoch: Optional[float] = None) -> None:
        """Compatibility wrapper used by training.py.

        The original single-file script called a steerer method each epoch to adjust LRs and
        adversarial pressure. In this project, the underlying implementation lives in `step()`.

        We drive `step()` using a *validation-derived* objective:
          - if VAL_PRIMARY == 'score': maximize vm['val_score'] -> minimize (-val_score)
          - otherwise: fall back to maximizing PSNR -> minimize (-val_psnr)

        Parameters:
          epoch: current epoch index (used for internal schedules)
          g_loss_epoch: optional training G loss (used only if provided)
          d_loss_epoch: optional training D loss (used only if provided)
        """
        self.epoch = int(epoch)

        # Pick a scalar objective to minimize for `step()`.
        if self.primary == 'score':
            score = float(vm.get('val_score', vm.get('val_psnr', 0.0)))
            obj = -score
        else:
            obj = -float(vm.get('val_psnr', vm.get('psnr', 0.0)))

        # If the caller provided the training G loss, blend it very lightly (optional).
        if g_loss_epoch is not None:
            try:
                obj = 0.95 * float(obj) + 0.05 * float(g_loss_epoch)
            except Exception:
                pass

        # D loss used for regime detection; if not provided, assume stable.
        dloss = float(d_loss_epoch) if d_loss_epoch is not None else 1.0
        self.step(obj, vm, dloss)

    def step(self, g_loss, vm, d_loss_epoch):
        """Metric-aware steering. Still PSNR-first, but gates GAN pressure when
        high-k error / H1 error / edge-bleed are worsening."""
        # unpack
        val_psnr = float(vm.get("val_psnr", 0.0))
        val_pix  = float(vm.get("val_pix",  0.0))

        # book-keeping
        # NOTE: `update(..., epoch=...)` sets `self.epoch` from the training loop.
        # Do NOT increment here; otherwise the internal epoch counter drifts and,
        # more importantly, resumes can replay a deterministic warmup window.
        self.epochs_since_adv_change += 1
        self.psnr_hist.append(val_psnr)
        self.score_hist.append(float(vm.get("val_score", val_psnr)))

        # push new metrics into short deques
        self.h1_hist.append(float(vm.get("h1_l1", 0.0)))
        self.spec_hi_hist.append(float(vm.get("spec_relerr_high", 0.0)))
        self.edge_hist.append(float(vm.get("edge_bleed", 0.0)))
        self.cov_hist.append(float(vm.get("cov_offdiag", 0.0)))
        self.psnr_p05_hist.append(float(vm.get("psnr_p05", val_psnr)))
        self.mae_p95_hist.append(float(vm.get("mae_p95",  vm.get("val_mae", val_pix))))

        improved = (self.best - g_loss) > self.min_delta
        rel_improved = ((self.best - g_loss) / max(self.best, 1e-12)) > self.g_up_rel_delta

        # G LR: same behavior as before (with "meaningful" up-bump)
        if rel_improved:
            self.best = min(self.best, g_loss)
            self.wait = 0
            if self.g_lr_up_enabled and rel_improved and (d_loss_epoch < self.d_ceil) and (self.g_lr < self.g_max):
                new_g = min(self.g_max, self.g_lr * self.factor_up)
                if new_g > self.g_lr:
                    self.g_lr = new_g
                    self._apply_g_lr(self.g_lr)
                    print(f"[*] G improved (↑ meaningfully): ↑ G LR → {self.g_lr:.2e}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_g = max(self.g_min, self.g_lr * self.factor_down)
                if new_g < self.g_lr:
                    self.g_lr = new_g
                    self._apply_g_lr(self.g_lr)
                    print(f"[*] Plateau: ↓ G LR → {self.g_lr:.2e}")
                self.wait = 0

        # Track per-epoch D loss for trend detection
        self._d_epoch_hist.append(float(d_loss_epoch))
        if self._d_trend_cooldown_rem > 0:
            self._d_trend_cooldown_rem -= 1

        # D LR guardrails (streak-gated)
        if not self.hold_d:
            # Update streak counters
            if d_loss_epoch < self.d_floor:
                self._d_strong_streak += 1; self._d_weak_streak = 0
            elif d_loss_epoch > self.d_ceil:
                self._d_weak_streak += 1; self._d_strong_streak = 0
            else:
                self._d_strong_streak = 0; self._d_weak_streak = 0

            # Apply updates only when the streak requirement is met
            if (self._d_strong_streak >= self.d_streak_req) and (self.d_lr > self.d_min):
                newr1 = max(0, 0.25*2.0)
                if not self.d_first_strong_drop_done:
                    drop_factor = self.d_first_strong_factor
                    self.d_first_strong_drop_done = True
                    tag = " (first hit)"
                else:
                    drop_factor = self.factor_down
                    tag = ""
                new_d = max(self.d_min, self.d_lr * drop_factor)
                self.withloss_d.set_r1_gamma(newr1)
                try:
                    self.withloss_d.reg_every.assign(4)
                except Exception:
                    self.withloss_d.reg_every = 4
                if new_d < self.d_lr:
                    self.d_lr = new_d; self._apply_d_lr(self.d_lr)
                    print(f"[*] D too strong x{self._d_strong_streak}{tag} (loss {d_loss_epoch:.3f}) → ↓ D LR to {self.d_lr:.2e}")
                self._d_strong_streak = 0

            elif (self._d_weak_streak >= self.d_streak_req) and (self.d_lr < self.d_max):
                new_d = min(self.d_max, self.d_lr * self.factor_up)
                self.withloss_d.set_r1_gamma(max(0.5*0.25, 0))
                try:
                    self.withloss_d.reg_every.assign(8)
                except Exception:
                    self.withloss_d.reg_every = 8
                if new_d > self.d_lr:
                    self.d_lr = new_d; self._apply_d_lr(self.d_lr)
                    print(f"[*] D too weak  x{self._d_weak_streak} (loss {d_loss_epoch:.3f}) → ↑ D LR to {self.d_lr:.2e}")
                self._d_weak_streak = 0

        else:
            # Hold mode: only allow gentle increases when clearly weak, also streak-gated
            if d_loss_epoch > self.d_ceil and self.d_lr < self.d_max:
                self._d_weak_streak += 1
                if self._d_weak_streak >= self.d_streak_req:
                    new_d = min(self.d_max, self.d_lr * self.factor_up)
                    if new_d > self.d_lr:
                        self.d_lr = new_d; self._apply_d_lr(self.d_lr)
                        print(f"[*] D weak x{self._d_weak_streak}; ↑ D LR to {self.d_lr:.2e} (hold mode)")
                    self._d_weak_streak = 0
            else:
                self._d_weak_streak = 0

        # Temporary D LR bump if D loss is *rising* (above floor, but not yet above ceiling).
        # This is separate from the existing ceiling-based LR increase logic above.
        if (not self.hold_d) and (self._d_trend_cooldown_rem == 0):
            if (len(self._d_epoch_hist) == self._d_epoch_hist.maxlen) and (self.d_lr < self.d_max):
                if (d_loss_epoch > self.d_floor) and (d_loss_epoch <= self.d_ceil):
                    diffs = [self._d_epoch_hist[i + 1] - self._d_epoch_hist[i] for i in range(len(self._d_epoch_hist) - 1)]
                    avg_delta = sum(diffs) / max(1, len(diffs))
                    if (avg_delta >= self.d_trend_min_delta) and all(d > 0 for d in diffs):
                        new_d_lr = min(self.d_max, self.d_lr * self.d_trend_lr_increase_factor)
                        if new_d_lr > self.d_lr:
                            self.d_lr = float(new_d_lr)
                            self._apply_d_lr(self.d_lr)
                            self._d_trend_cooldown_rem = int(max(0, self.d_trend_cooldown))
                            self.withloss_d.set_r1_gamma(max(0.5*0.25, 0))
                        try:
                            self.withloss_d.reg_every.assign(8)
                        except Exception:
                            self.withloss_d.reg_every = 8
                            self.withloss_d.set_inst_noise(0); self.withloss_g.set_inst_noise(0)
                            print(f"[*] D loss trending up (avg Δ {avg_delta:.4f}); temporary ↑ D LR → {self.d_lr:.2e}")

        # --- Metric-aware adv weight gating (PSNR/score first) ---
        adv_w = self.scheduled_adv_w

        # patience/cooldown gate exactly as before
        if rel_improved: self.adv_wait = 0
        else:        self.adv_wait += 1
        can_adjust = (self.epochs_since_adv_change >= ADV_COOLDOWN and
                      self.adv_wait >= ADV_PATIENCE and
                      len(self.psnr_hist) == self.psnr_hist.maxlen)

        if can_adjust and (getattr(self, "prev_psnr", None) is not None and getattr(self, "prev_pix", None) is not None):
            # Choose primary trend source
            if self.primary == "score" and len(self.score_hist) == self.score_hist.maxlen:
                trend      = self.score_hist[-1] - self.score_hist[0]
                trend_up   = trend >  self.score_eps
                trend_down = trend < -self.score_eps
            else:
                trend      = self.psnr_hist[-1] - self.psnr_hist[0]
                trend_up   = trend >  PSNR_EPS
                trend_down = trend < -PSNR_EPS

            # short-window drifts
            h1_trend   = (self.h1_hist[-1]      - self.h1_hist[0])     if len(self.h1_hist)      == self.h1_hist.maxlen      else 0.0
            hi_trend   = (self.spec_hi_hist[-1] - self.spec_hi_hist[0]) if len(self.spec_hi_hist) == self.spec_hi_hist.maxlen else 0.0
            edge_trend = (self.edge_hist[-1]    - self.edge_hist[0])    if len(self.edge_hist)    == self.edge_hist.maxlen    else 0.0

            # If metric falling → reduce adversarial pressure
            if (trend_down) and (self.epoch >= 12):
                adv_w *= 0.80  # 20% cut
                print(f"[*] Trend down ↓, reducing adv_w")

            # If rising but structural/high-k/bleed getting worse → reduce a bit (protect)
            elif (trend_up) and (val_pix - (self.prev_pix or val_pix)) > -1e-6:
                bad_grad  = (h1_trend > self.h1_eps)
                bad_highk = (hi_trend > self.spec_hi_eps)
                bad_bleed = (edge_trend > self.edge_eps)
                if bad_grad or bad_highk or bad_bleed:
                    adv_w *= 0.90  # small back-off
                    print(f"[*] Gating adv_w (grad/high-k/bleed ↑: {bad_grad}/{bad_highk}/{bad_bleed})")
                else:
                    adv_w *= min(1.0 + STEER_LIMIT, 1.10)  # gentle increase
                    print(f"[*] Trend up ↑, increasing adv_w")

        # --- Hard drift guard (absolute levels): extra back-off when clearly drifting ---
        if (vm.get('edge_bleed', 0.0) > self.bleed_hi) or (vm.get('spec_relerr_high', 0.0) > self.spec_hi_bad):
            adv_w *= self.adv_decay  # e.g., 0.5
            print("[*] Drift guard: bleed/high-k high → decaying adv_w")

        # Enforce local cap (below global ADV_MAX), then global clamp
        adv_w = min(self.adv_cap, adv_w)
        adv_w = min(ADV_MAX, max(ADV_MIN, adv_w))

        self.withloss_g.set_adv_weight(adv_w)
        # Expose a *cap* for next epoch only if a guard reduced adv_w below the
        # scheduled ramp target for this epoch. Otherwise clear the cap so the
        # schedule can continue to ramp upward.
        sched = float(getattr(self, "scheduled_adv_w", adv_w))
        if adv_w < sched - 1e-12:
            self.adv_w_sched = float(adv_w)
        else:
            self.adv_w_sched = float(ADV_MAX)
        # === [NEW] Auto-nudge covariance penalty (COV_W) to reduce cross-talk ===
        self.epochs_since_penalty_change += 1
        # require short histories to be full and respect a small cooldown
        can_tune_cov = (
            len(self.edge_hist) == self.edge_hist.maxlen and
            len(self.cov_hist)  == self.cov_hist.maxlen  and
            self.epochs_since_penalty_change >= self.penalty_cooldown
        )
        if can_tune_cov:
            edge_trend = self.edge_hist[-1] - self.edge_hist[0]
            cov_trend  = self.cov_hist[-1]  - self.cov_hist[0]
            h1_trend   = self.h1_hist[-1]   - self.h1_hist[0]    if len(self.h1_hist)      == self.h1_hist.maxlen      else 0.0
            hi_trend   = self.spec_hi_hist[-1]-self.spec_hi_hist[0] if len(self.spec_hi_hist)== self.spec_hi_hist.maxlen else 0.0

            getting_worse = (edge_trend > self.edge_eps) or (cov_trend > self.cov_eps) or (h1_trend > self.h1_eps) or (hi_trend > self.spec_hi_eps)
            getting_better= (edge_trend < -self.edge_eps) and (cov_trend < -self.cov_eps) and (h1_trend < self.h1_eps) and (hi_trend < self.spec_hi_eps)

            cur_cov = float(self.withloss_g.get_cov_weight())
            base    = max(self.cov_base, 0.0)
            cap     = max(base * self.cov_cap, base + 1e-12)
            if getting_worse:
                # nudge up gently (bounded by steer_limit)
                bump = min(0.25, float(self.steer_limit))
                new_cov = min(cap, cur_cov * (1.0 + bump))
                if new_cov > cur_cov:
                    self.withloss_g.set_cov_weight(new_cov, None)
                    self.epochs_since_penalty_change = 0
                    print(f"[*] Cov decor ↑ to {new_cov:.2e} (edge/cov/H1/high-k trending up)")
            elif getting_better:
                # ease off very slightly but never below a floor
                new_cov = max(self.cov_floor, cur_cov * 0.90)
                if new_cov < cur_cov - 1e-12:
                    self.withloss_g.set_cov_weight(new_cov, None)
                    self.epochs_since_penalty_change = 0
                    print(f"[*] Cov decor ↓ to {new_cov:.2e} (edge/cov improving)")

        self.prev_psnr, self.prev_pix = val_psnr, val_pix
        # Ensure D stays within requested band every epoch
        self.clamp_d_to_band()
