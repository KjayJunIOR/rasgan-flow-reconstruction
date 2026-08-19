from __future__ import annotations

import os

from ..env import tf
from typing import Optional
from ..tf_layers import Module

from ..config import (
    PRETRAIN_W_LP_DELTA_P, PRETRAIN_P_UPWEIGHT, PRETRAIN_LP_KERNEL, R1_GAMMA_BASE, W_RES_DEFAULT,
    W_LOW_DEFAULT, W_SPEC_TARGET, W_ESPEC_TARGET, COV_W, DECORR_W, EDGE_TANH_GAIN, LP_W
)

from .content import (
    _finite, _grad_xy, _grad_mag, charbonnier_balanced_weighted, charbonnier,
    _edge_align_penalty, _offdiag_cov_penalty,
    lowpass_l1_on_delta,
    tv_loss,
)

from .physics import PhysicsParams, physics_terms

from .pod import PodParams, pod_sidecar_losses

from .gan import (
    _make_d_input,
    _resize_like_nchw,
    rasgan_d_loss, rasgan_g_loss,
    bce_d_loss, bce_g_loss,
    rpsgan_d_loss, rpsgan_g_loss,
    add_instance_noise, r1_penalty_safe,
)
from .spectrum import (
    log_amp_fft, safe_energy_spectrum, spectrum_loss_from_LA, energy_spectrum_loss_from_E,
    energy_spectrum2d, lowk_loss_from_E, _crop_border, _crop_logits
)

class WithLoss_init(Module):
    """ Content pretrain:  - channel-weighted balanced Charbonnier on (p, v, ωz)
      - tiny residual-align & low-k consistency (optional but helpful) """
    def __init__(self, G, low_mask, w_res_init=0.05, w_low_init=PRETRAIN_W_LP_DELTA_P,
                 p_up=PRETRAIN_P_UPWEIGHT, lp_kernel=PRETRAIN_LP_KERNEL):
        super().__init__()
        self.G = G
        self.low_mask = low_mask
        self.w_res_init = float(w_res_init)
        self.w_low_init = float(w_low_init)
        self.p_up = float(p_up)
        self.lp_kernel = int(lp_kernel)

        # Physics-informed losses (disabled by default)
        self.phys_params: Optional[PhysicsParams] = None
        self.hr_mean: Optional[tf.Tensor] = None
        self.hr_std: Optional[tf.Tensor] = None
        self.w_vomega = 0.0
        self.w_omcons = 0.0
        self.w_div    = 0.0
        self.w_mom    = 0.0
        self.w_ppois  = 0.0
        self.physics_steady = True

        # POD sidecar losses (disabled by default)
        self.pod_params: Optional[PodParams] = None
        self.lr_mean: Optional[tf.Tensor] = None
        self.lr_std: Optional[tf.Tensor] = None
        self.w_pod_vel = 0.0
        self.w_pod_p   = 0.0
        self.w_pod_w   = 0.0

    def set_physics(self, hr_mean: tf.Tensor, hr_std: tf.Tensor, params: PhysicsParams):
        self.hr_mean = hr_mean; self.hr_std = hr_std; self.phys_params = params

    def set_pod(self, lr_mean: tf.Tensor, lr_std: tf.Tensor, pod_params: PodParams):
        self.lr_mean = lr_mean; self.lr_std = lr_std; self.pod_params = pod_params

    def set_pod_vel_blend_alpha(self, a: float):
        self.pod_vel_blend_alpha = float(min(1.0, max(0.0, a)))

    def eval_terms(self, lr, hr, pod_coeffs=None):
        """Deterministic loss breakdown for validation.

        Returns a dict of *active* weighted loss terms plus `total`.
        This reuses the exact same primitives/definitions as training.
        """
        fake = self.G(lr, training=False)

        # Match LR conditioning resolution to fake/hr (needed for 2× SR training).
        sr_scale = int(getattr(self.G, "sr_scale", 1))
        lr_up = _resize_like_nchw(lr, fake, method="bilinear") if sr_scale == 2 else lr

        pix_w = (float(self.p_up), 1.0, 1.0)
        pix = self.content_w * charbonnier_balanced_weighted(fake, hr, pix_w)

        terms = {"pix": _finite(pix, "init_pix")}

        if self.w_res_init > 0.0:
            res = self.w_res_init * charbonnier_balanced_weighted(
                _crop_border(fake - lr_up),
                _crop_border(hr   - lr_up),
                (self.p_up, 1.0, 1.0),
            )
            terms["res"] = _finite(res, "init_res")

        if self.w_low_init > 0.0:
            delta_p = (fake - lr_up)[:, 0:1, :, :]
            lowp = self.w_low_init * lowpass_l1_on_delta(delta_p, k=self.lp_kernel)
            terms["lowp"] = _finite(lowp, "init_lowP")

        # Optional gradient alignment (if configured)
        grad_w = float(getattr(self, "grad_w", 0.0))
        if grad_w > 0.0:
            gradx, grady = _grad_xy(fake)
            hx, hy = _grad_xy(hr)
            grad_loss = charbonnier_balanced_weighted(gradx, hx, pix_w) + charbonnier_balanced_weighted(grady, hy, pix_w)
            terms["grad"] = _finite(grad_w * grad_loss, "init_grad")

        tv = float(getattr(self, "tv_w", 0.0))
        if tv > 0.0:
            tv_l = self.tv_w * tv_loss(_crop_border(fake))
            terms["tv"] = _finite(tv_l, "init_tv")

        # Physics terms (if configured)
        if self.phys_params is not None and self.hr_mean is not None and self.hr_std is not None:
            if (self.w_vomega + self.w_omcons + self.w_div + self.w_mom + self.w_ppois) > 0.0:
                ph = physics_terms(fake, self.hr_mean, self.hr_std, self.phys_params, steady=self.physics_steady)
                phys = (
                    self.w_vomega * ph["vomega"]
                    + self.w_omcons * ph["omcons"]
                    + self.w_div * ph["div"]
                    + self.w_mom * ph["mom"]
                    + self.w_ppois * ph["ppois"]
                )
                terms["phys"] = _finite(phys, "init_phys")

        # POD sidecar terms (if configured)
        if (
            self.pod_params is not None
            and self.lr_mean is not None
            and self.lr_std is not None
            and self.phys_params is not None
            and (self.w_pod_vel + self.w_pod_p + self.w_pod_w) > 0.0
        ):
            # POD losses are defined in LR space.
            # For true SR (scale=2), compare against the HR target restricted to LR.
            lr_for_pod = lr
            if sr_scale == 2 and self.hr_mean is not None and self.hr_std is not None:
                hr_phys = denorm(hr, self.hr_mean, self.hr_std)
                hr_lr_phys = resize_nchw(hr_phys, (self.pod_params.h_lr, self.pod_params.w_lr), method="bilinear")
                lr_for_pod = (tf.cast(hr_lr_phys, tf.float32) - tf.cast(self.lr_mean, tf.float32)) / (tf.cast(self.lr_std, tf.float32) + 1e-8)

            tim_u = tim_v = tim_p = None
            if pod_coeffs is not None and isinstance(pod_coeffs, (tuple, list)) and len(pod_coeffs) == 3:
                tim_u, tim_v, tim_p = pod_coeffs
            podt = pod_sidecar_losses(
                pred_hr_n=fake,
                lr_n=lr_for_pod,
                hr_mean=self.hr_mean,
                hr_std=self.hr_std,
                lr_mean=self.lr_mean,
                lr_std=self.lr_std,
                phys=self.phys_params,
                pod=self.pod_params,
                tim_u=tim_u,
                tim_v=tim_v,
                tim_p=tim_p,
                vel_blend_alpha=self.pod_vel_blend_alpha,
            )
            pod_loss = (
                self.w_pod_vel * (podt["vel_cycle"] + podt["vel_coeff"])
                + self.w_pod_p * (podt["p_cycle"] + podt["p_coeff"])
                + self.w_pod_w * podt["w_pod"]
            )
            terms["pod"] = _finite(pod_loss, "init_pod")

        total = None
        for v in terms.values():
            total = v if total is None else (total + v)
        terms["total"] = tf.cast(total if total is not None else 0.0, tf.float32)
        return terms

    def forward(self, lr, hr, pod_coeffs=None):
        fake = self.G(lr, pod_coeffs=pod_coeffs, training=True)

        # If lr is low-res and fake/hr are high-res (e.g. 2× SR),
        # upsample lr for residual-style losses.
        sr_scale = int(getattr(self.G, "sr_scale", 1))
        lr_up = _resize_like_nchw(lr, fake, method="bilinear") if sr_scale == 2 else lr

        self.pix_w = (self.p_up, 1.0, 1.0)  # weights for (P, v, ωz)
        # pixel (weighted)
        pix = self.content_w * charbonnier_balanced_weighted(fake, hr, self.pix_w)  # needs self.pix_w and helper already added
        # residual alignment (very small)
        res = 0.0
        if self.w_res_init > 0.0:
            res = _finite(self.w_res_init * charbonnier_balanced_weighted(
                _crop_border(fake - lr_up),
                _crop_border(hr   - lr_up), (1.0, 1.0, 1.0)), "init_res")
        # Delta in output space (difference from conditioning); works for both modes
        L_low = 0.0
        if self.w_low_init > 0.0:
            delta_p = (fake - lr_up)[:, 0:1, :, :]   # only pressure channel
            L_low  = _finite(self.w_low_init * lowpass_l1_on_delta(delta_p, k=self.lp_kernel), "init_lowP")   # penalize low-freq in delta

        grad_loss = 0.0
        if self.grad_w > 0.0:
            gradx, grady = _grad_xy(fake)
            hx, hy = _grad_xy(hr)
            grad_loss = self.grad_w * (charbonnier_balanced_weighted(gradx, hx, self.pix_w) + charbonnier_balanced_weighted(grady, hy, self.pix_w))

        tv = 0.0
        if self.tv_w > 0.0:
            tv = _finite(self.tv_w * tv_loss(_crop_border(fake)), "init_tv")

        phys = 0.0
        if self.phys_params is not None and self.hr_mean is not None and self.hr_std is not None:
            if (self.w_vomega + self.w_omcons + self.w_div + self.w_mom + self.w_ppois) > 0.0:
                terms = physics_terms(fake, self.hr_mean, self.hr_std, self.phys_params, steady=self.physics_steady)
                phys = (self.w_vomega * terms['vomega'] +
                        self.w_omcons * terms['omcons'] +
                        self.w_div    * terms['div'] +
                        self.w_mom    * terms['mom'] +
                        self.w_ppois  * terms['ppois'])

        pod_loss = 0.0
        if self.pod_params is not None and self.lr_mean is not None and self.lr_std is not None and self.phys_params is not None:
            if (self.w_pod_vel + self.w_pod_p + self.w_pod_w) > 0.0:
                lr_for_pod = lr
                if sr_scale == 2 and self.hr_mean is not None and self.hr_std is not None:
                    hr_phys = denorm(hr, self.hr_mean, self.hr_std)
                    hr_lr_phys = resize_nchw(hr_phys, (self.pod_params.h_lr, self.pod_params.w_lr), method="bilinear")
                    lr_for_pod = (tf.cast(hr_lr_phys, tf.float32) - tf.cast(self.lr_mean, tf.float32)) / (tf.cast(self.lr_std, tf.float32) + 1e-8)

                tim_u = tim_v = tim_p = None
                if pod_coeffs is not None and isinstance(pod_coeffs, (tuple, list)) and len(pod_coeffs) == 3:
                    tim_u, tim_v, tim_p = pod_coeffs
                terms_pod = pod_sidecar_losses(
                    pred_hr_n=fake,
                    lr_n=lr_for_pod,
                    hr_mean=self.hr_mean,
                    hr_std=self.hr_std,
                    lr_mean=self.lr_mean,
                    lr_std=self.lr_std,
                    phys=self.phys_params,
                    pod=self.pod_params,
                    tim_u=tim_u,
                    tim_v=tim_v,
                    tim_p=tim_p,
                    vel_blend_alpha=self.pod_vel_blend_alpha,
                )
                pod_loss = (
                    self.w_pod_vel * (terms_pod["vel_cycle"] + terms_pod["vel_coeff"]) +
                    self.w_pod_p   * (terms_pod["p_cycle"]   + terms_pod["p_coeff"]) +
                    self.w_pod_w   * terms_pod["w_pod"]
                )

        return tf.cast(pix + res + grad_loss + tv + L_low + phys + pod_loss, tf.float32)

class WithLoss_D(Module):
    def __init__(self, D, G, r1_gamma=R1_GAMMA_BASE, inst_noise_std=0.0, reg_every=8,
                 edge_lambda=1.0, cond_drop_p=0.0, ema_real=None, ema_fake=None):
        super().__init__(); self.D=D; self.G=G
        self.ema_real = ema_real; self.ema_fake = ema_fake
        self.r1_gamma      = tf.Variable(float(r1_gamma), trainable=False, dtype=tf.float32)
        self.inst_noise_std = tf.Variable(float(inst_noise_std), trainable=False, dtype=tf.float32)
        self.edge_lambda   = tf.Variable(float(edge_lambda), trainable=False, dtype=tf.float32)
        self.cond_drop_p   = tf.Variable(float(cond_drop_p), trainable=False, dtype=tf.float32)
        self.reg_every     = tf.Variable(int(reg_every), trainable=False, dtype=tf.int64)
        self.step          = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.gan_mode='rasgan'
        self._last_clf = tf.constant(0.0)
        self._last_r1  = tf.constant(0.0)
    def set_inst_noise(self, s): self.inst_noise_std.assign(float(s))
    def set_r1_gamma(self, g):
        self.r1_gamma.assign(float(g))
    def set_conditioning(self, edge_lambda=None, cond_drop_p=None):
        if edge_lambda is not None: self.edge_lambda.assign(float(edge_lambda))
        if cond_drop_p is not None: self.cond_drop_p.assign(float(cond_drop_p))
    def set_gan_mode(self, mode):
        assert mode in ('bce','rasgan','rpsgan'); self.gan_mode = mode
    def forward(self, lr, hr, pod_coeffs=None):
        self.step.assign_add(1)
        fake   = self.G(lr, pod_coeffs=pod_coeffs, training=False)
        hr_n   = add_instance_noise(hr,  self.inst_noise_std)
        fake_n = add_instance_noise(fake, self.inst_noise_std)

        # clip & finite guard BEFORE D sees them
        hr_c   = tf.where(tf.math.is_finite(tf.clip_by_value(hr_n,   -6.0, 6.0)), tf.clip_by_value(hr_n,   -6.0, 6.0), tf.zeros_like(hr_n))
        fake_c = tf.where(tf.math.is_finite(tf.clip_by_value(fake_n, -6.0, 6.0)), tf.clip_by_value(fake_n, -6.0, 6.0), tf.zeros_like(fake_n))
        lr_c   = tf.where(tf.math.is_finite(tf.clip_by_value(lr,     -6.0, 6.0)), tf.clip_by_value(lr,     -6.0, 6.0), tf.zeros_like(lr))

        logits_real = self.D(_make_d_input(hr_c,   lr_c, self.edge_lambda, self.cond_drop_p, training=True))
        logits_fake = self.D(_make_d_input(fake_c, lr_c, self.edge_lambda, self.cond_drop_p, training=True))
        logits_real = _crop_logits(logits_real)
        logits_fake = _crop_logits(logits_fake)
        if self.gan_mode == 'bce':
            clf = bce_d_loss(logits_real, logits_fake, scale=2.0)
        elif self.gan_mode == 'rasgan':
            clf = rasgan_d_loss(logits_real, logits_fake, self.ema_real, self.ema_fake)
        else:
            clf = rpsgan_d_loss(logits_real, logits_fake)
        self._last_clf = clf
        loss = clf

        # Detach LR to shrink the tape and avoid unnecessary grads/memory in R1
        do_r1 = (self.r1_gamma > 0.0) & tf.equal(tf.math.floormod(self.step, self.reg_every), 0)
        def r1_branch():
            real_for_D = _make_d_input(hr_c, tf.stop_gradient(lr_c), self.edge_lambda, 0.0, training=False)
            reg = r1_penalty_safe(self.D, real_for_D)
            # lazy R1: multiply by reg_every to keep expected strength
            r1 = 0.5 * self.r1_gamma * tf.cast(self.reg_every, tf.float32) * reg
            return r1
        r1 = tf.cond(do_r1, r1_branch, lambda: tf.constant(0.0, tf.float32))
        self._last_r1 = r1
        loss = loss + r1

        return loss

class WithLoss_G(Module):
    def __init__(self, D, G, adv_weight,
                 w_spec, w_espec, w_res, w_low,
                 M_STACK, WEIGHTS, DEN_BINS, LOW_MASK,
                 inst_noise_std=0.0, enable_fm=False, edge_lambda=1.0, cond_drop_p=0.0,
                 ema_real=None, ema_fake=None):
        super().__init__(); self.D=D; self.G=G
        self.ema_real = ema_real; self.ema_fake = ema_fake
        # === runtime knobs (TF variables; safe under tf.function) ===
        self._adv_weight = tf.Variable(float(adv_weight), trainable=False, dtype=tf.float32)
        self._w_spec  = tf.Variable(float(w_spec),  trainable=False, dtype=tf.float32)
        self._w_espec = tf.Variable(float(w_espec), trainable=False, dtype=tf.float32)
        self._w_res   = tf.Variable(float(w_res),   trainable=False, dtype=tf.float32)
        self._w_low   = tf.Variable(float(w_low),   trainable=False, dtype=tf.float32)
        self.gan_mode='rasgan'
        self._inst_noise_std = tf.Variable(float(inst_noise_std), trainable=False, dtype=tf.float32)
        self._edge_lambda    = tf.Variable(float(edge_lambda),    trainable=False, dtype=tf.float32)
        self._cond_drop_p    = tf.Variable(float(cond_drop_p),    trainable=False, dtype=tf.float32)

        # Mode + feature-matching toggle (also runtime-tunable)
        self._enable_fm = tf.Variable(bool(enable_fm), trainable=False, dtype=tf.bool)

        # These are assigned from training.py (so keep them mutable + graph-safe)
        self._tv_w      = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._grad_w    = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._content_w = tf.Variable(1.0, trainable=False, dtype=tf.float32)

        # FFT loss scaffolding
        self.M_STACK=M_STACK; self.WEIGHTS=WEIGHTS; self.DEN_BINS=DEN_BINS; self.LOW_MASK=LOW_MASK

        # === decorrelation weight (tunable at runtime) ===
        self.base_decorr_weight = float(globals().get("DECORR_W", 0.0))
        self._decorr_weight     = tf.Variable(float(self.base_decorr_weight), trainable=False, dtype=tf.float32)
        # optional cap from env (keeps things sane if we auto-bump)
        self.decorr_cap         = float(os.environ.get("DECORR_CAP", 5.0))
        # === covariance decorrelation (off-diagonal) ===
        self.base_cov_weight = float(globals().get("COV_W", 0.0))
        self._cov_weight     = tf.Variable(float(self.base_cov_weight), trainable=False, dtype=tf.float32)
        self._cov_on_grads   = tf.Variable(bool(globals().get("COV_ON_GRADS", True)), trainable=False, dtype=tf.bool)
        self._lambda_lp = tf.Variable(float(globals().get("LP_W", 0.0)), trainable=False, dtype=tf.float32)  # low-pass penalty on delta
        self.lp_kernel   = 4;

        # Physics-informed losses (disabled by default)
        self.phys_params: Optional[PhysicsParams] = None
        self.hr_mean: Optional[tf.Tensor] = None
        self.hr_std: Optional[tf.Tensor] = None
        self._w_vomega = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._w_omcons = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._w_div    = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._w_mom    = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._w_ppois  = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._physics_steady: bool = True
        # POD sidecar losses (disabled by default)
        self.pod_params: Optional[PodParams] = None
        self.lr_mean: Optional[tf.Tensor] = None
        self.lr_std: Optional[tf.Tensor] = None
        self._w_pod_vel = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._w_pod_p   = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._w_pod_w   = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._pod_vel_blend_alpha = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    # ---- properties to keep existing training.py attribute assignments working ----
    @property
    def adv_weight(self): return self._adv_weight
    @property
    def w_spec(self): return self._w_spec
    @property
    def w_espec(self): return self._w_espec
    @property
    def w_res(self): return self._w_res
    @property
    def w_low(self): return self._w_low
    @property
    def inst_noise_std(self): return self._inst_noise_std
    @property
    def edge_lambda(self): return self._edge_lambda
    @property
    def cond_drop_p(self): return self._cond_drop_p
    @property
    def enable_fm(self): return self._enable_fm

    @property
    def tv_w(self): return self._tv_w
    @tv_w.setter
    def tv_w(self, v): self._tv_w.assign(float(v))

    @property
    def grad_w(self): return self._grad_w
    @grad_w.setter
    def grad_w(self, v): self._grad_w.assign(float(v))

    @property
    def content_w(self): return self._content_w
    @content_w.setter
    def content_w(self, v): self._content_w.assign(float(v))

    # Physics weights (assigned directly in training.py)
    @property
    def w_vomega(self): return self._w_vomega
    @w_vomega.setter
    def w_vomega(self, v): self._w_vomega.assign(float(v))

    @property
    def w_omcons(self): return self._w_omcons
    @w_omcons.setter
    def w_omcons(self, v): self._w_omcons.assign(float(v))

    @property
    def w_div(self): return self._w_div
    @w_div.setter
    def w_div(self, v): self._w_div.assign(float(v))

    @property
    def w_mom(self): return self._w_mom
    @w_mom.setter
    def w_mom(self, v): self._w_mom.assign(float(v))

    @property
    def w_ppois(self): return self._w_ppois
    @w_ppois.setter
    def w_ppois(self, v): self._w_ppois.assign(float(v))

    @property
    def physics_steady(self) -> bool:
        return bool(self._physics_steady)
    @physics_steady.setter
    def physics_steady(self, v: Union[bool, int, float]):
        self._physics_steady = bool(v)

    # POD weights (assigned directly in training.py)
    @property
    def w_pod_vel(self): return self._w_pod_vel
    @w_pod_vel.setter
    def w_pod_vel(self, v): self._w_pod_vel.assign(float(v))

    @property
    def w_pod_p(self): return self._w_pod_p
    @w_pod_p.setter
    def w_pod_p(self, v): self._w_pod_p.assign(float(v))

    @property
    def w_pod_w(self): return self._w_pod_w
    @w_pod_w.setter
    def w_pod_w(self, v): self._w_pod_w.assign(float(v))

    @property
    def pod_vel_blend_alpha(self): return self._pod_vel_blend_alpha
    @pod_vel_blend_alpha.setter
    def pod_vel_blend_alpha(self, v):
        self._pod_vel_blend_alpha.assign(float(min(1.0, max(0.0, v))))

    def set_physics(self, hr_mean: tf.Tensor, hr_std: tf.Tensor, params: PhysicsParams):
        self.hr_mean = hr_mean; self.hr_std = hr_std; self.phys_params = params

    def set_pod(self, lr_mean: tf.Tensor, lr_std: tf.Tensor, pod_params: PodParams):
        self.lr_mean = lr_mean; self.lr_std = lr_std; self.pod_params = pod_params

    def set_adv_weight(self, w): self._adv_weight.assign(float(w))
    def set_decorr_weight(self, w: float):
        # clamp to [0, cap]
        w = float(w)
        if w < 0.0: w = 0.0
        if w > self.decorr_cap: w = self.decorr_cap
        self._decorr_weight.assign(w)
    def get_decorr_weight(self):
        return float(self._decorr_weight.numpy())
    def set_cov_weight(self, w: float, on_grads: bool = None):
        w = float(w)
        if w < 0.0: w = 0.0
        self.cov_weight = w
        if on_grads is not None:
            self._cov_on_grads.assign(bool(on_grads))
    def get_cov_weight(self):
        return float(self._cov_weight.numpy())
    def set_loss_weights(self, w_spec=None,w_espec=None,w_res=None,w_low=None):
        if w_spec is not None: self._w_spec.assign(float(w_spec))
        if w_espec is not None: self._w_espec.assign(float(w_espec))
        if w_res is not None: self._w_res.assign(float(w_res))
        if w_low is not None: self._w_low.assign(float(w_low))
    def set_inst_noise(self, s): self._inst_noise_std.assign(float(s))
    def set_gan_mode(self, mode):
        assert mode in ('bce','rasgan','rpsgan'); self.gan_mode = mode
    def set_feature_matching(self, on=True): self._enable_fm.assign(bool(on))
    def set_conditioning(self, edge_lambda=None, cond_drop_p=None):
        if edge_lambda is not None: self._edge_lambda.assign(float(edge_lambda))
        if cond_drop_p is not None: self._cond_drop_p.assign(float(cond_drop_p))
    def set_pod_vel_blend_alpha(self, a: float):
        self.pod_vel_blend_alpha = a
    def _compute_terms(self, lr, hr, pod_coeffs=None, *, deterministic: bool):
        """Compute weighted loss terms.

        When `deterministic=True`, this disables instance noise and conditioning
        dropout, and evaluates D in inference mode. This is intended for
        validation so the score matches training *definitions* but is stable.
        """
        g_training = False if deterministic else True
        fake = self.G(lr, pod_coeffs=pod_coeffs, training=g_training)

        # Conditioning upsample is only meaningful in true 2× SR mode.
        sr_scale = int(getattr(self.G, "sr_scale", 1))
        lr_up = _resize_like_nchw(lr, fake, method="bilinear") if sr_scale == 2 else lr

        inst_std = tf.constant(0.0, tf.float32) if deterministic else self.inst_noise_std
        p_drop   = tf.constant(0.0, tf.float32) if deterministic else self.cond_drop_p
        d_training = False if deterministic else True

        fake_n = add_instance_noise(fake, inst_std)
        hr_n   = add_instance_noise(hr,   inst_std)

        # clip & finite guard BEFORE D sees them
        hr_c   = tf.where(tf.math.is_finite(tf.clip_by_value(hr_n,   -6.0, 6.0)), tf.clip_by_value(hr_n,   -6.0, 6.0), tf.zeros_like(hr_n))
        fake_c = tf.where(tf.math.is_finite(tf.clip_by_value(fake_n, -6.0, 6.0)), tf.clip_by_value(fake_n, -6.0, 6.0), tf.zeros_like(fake_n))
        lr_c   = tf.where(tf.math.is_finite(tf.clip_by_value(lr_up,  -6.0, 6.0)), tf.clip_by_value(lr_up,  -6.0, 6.0), tf.zeros_like(lr_up))

        # Always request feats so enable_fm can be toggled safely under tf.function
        logits_fake, f_fake = self.D(_make_d_input(fake_c, lr_c, self.edge_lambda, p_drop, training=d_training), return_feats=True)
        logits_real, f_real = self.D(_make_d_input(hr_c,   lr_c, self.edge_lambda, p_drop, training=d_training), return_feats=True)
        logits_real = _crop_logits(logits_real, 8)
        logits_fake = _crop_logits(logits_fake, 8)

        # GAN loss
        if self.gan_mode == 'bce':
            g_gan = self.adv_weight * bce_g_loss(logits_fake, scale=2.0)
        elif self.gan_mode == 'rasgan':
            g_gan = self.adv_weight * rasgan_g_loss(logits_real, logits_fake, self.ema_real, self.ema_fake)
        else:
            g_gan = self.adv_weight * rpsgan_g_loss(logits_real, logits_fake)
        g_gan = _finite(g_gan, "g_gan")

        # Clip the signal a bit before FFT-based losses to prevent extreme outliers
        fake_clip = tf.where(tf.math.is_finite(tf.clip_by_value(fake, -6.0, 6.0)), tf.clip_by_value(fake, -6.0, 6.0), tf.zeros_like(fake))
        hr_clip   = tf.where(tf.math.is_finite(tf.clip_by_value(hr,   -6.0, 6.0)), tf.clip_by_value(hr, -6.0, 6.0), tf.zeros_like(hr))
        lr_clip   = tf.where(tf.math.is_finite(tf.clip_by_value(lr_up, -6.0, 6.0)), tf.clip_by_value(lr_up, -6.0, 6.0), tf.zeros_like(lr_up))

        # Use unclipped features for content loss, but keep finite guard
        fake_f = tf.where(tf.math.is_finite(fake), fake, tf.zeros_like(fake))
        hr_f   = tf.where(tf.math.is_finite(hr),   hr,   tf.zeros_like(hr))
        pix = _finite(self.content_w * charbonnier_balanced_weighted(fake_f, hr_f, self.pix_w), "pix")

        # Residuals
        r_fake = fake_clip - lr_clip
        r_hr   = hr_clip   - lr_clip

        # Spectrum losses (skip if both weights are 0)
        def _fft_terms():
            LA_fake = log_amp_fft(r_fake)
            LA_hr   = log_amp_fft(r_hr)
            spec  = _finite(self.w_spec  * spectrum_loss_from_LA(LA_fake, LA_hr, top_frac=0.30), "spec")
            E_fake = energy_spectrum2d(r_fake)
            E_hr   = energy_spectrum2d(r_hr)
            espec = _finite(self.w_espec * energy_spectrum_loss_from_E(E_fake, E_hr, self.M_STACK, self.WEIGHTS, self.DEN_BINS, top_frac=0.30), "espec")
            return spec, espec
        def _no_fft_terms():
            return (tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32))
        spec, espec = tf.cond((self.w_spec > 0.0) | (self.w_espec > 0.0), _fft_terms, _no_fft_terms)

        # residual alignment & low-k consistency
        res   = _finite(self.w_res * charbonnier_balanced_weighted(
            _crop_border(fake_clip - lr_clip),
            _crop_border(hr_clip - lr_clip),
            self.pix_w), "res")
        if self.LOW_MASK is not None:
            def _lowk():
                Ey = energy_spectrum2d(r_fake)
                Et = energy_spectrum2d(r_hr)
                return _finite(self.w_low * lowk_loss_from_E(Ey, Et, self.LOW_MASK), "lowk")
            lowk = tf.cond(self.w_low > 0.0, _lowk, lambda: tf.constant(0.0, tf.float32))
        else:
            lowk = tf.constant(0.0, tf.float32)

        tv = _finite(self.tv_w * tv_loss(_crop_border(fake_clip)), "tv")

        # Feature matching (last 3 blocks)
        def _fm():
            acc = tf.constant(0.0, tf.float32)
            for a, b in zip(f_fake[-3:], f_real[-3:]):
                acc = acc + charbonnier(a, b)
            return _finite(5e-4 * acc, "fm")
        fm = tf.cond(self.enable_fm, _fm, lambda: tf.constant(0.0, tf.float32))

        # Gradient alignment
        gradx, grady = _grad_xy(fake)
        hx, hy = _grad_xy(hr)
        grad_loss = charbonnier_balanced_weighted(gradx, hx, self.pix_w) + charbonnier_balanced_weighted(grady, hy, self.pix_w)
        grad = _finite(self.grad_w * grad_loss, "grad")

        # Cross-channel edge decorrelation
        gxT, gyT = gradx[:, 0:1], grady[:, 0:1]
        gxV, gyV = gradx[:, 1:2], grady[:, 1:2]
        gxW, gyW = gradx[:, 2:3], grady[:, 2:3]
        decor_Tv = _edge_align_penalty(gxT, gyT, gxV, gyV)
        decor_Tw = _edge_align_penalty(gxT, gyT, gxW, gyW)
        decor = _finite(self._decorr_weight * (decor_Tv + decor_Tw), "decor")

        # Off-diagonal covariance penalty
        def _cov():
            feat = tf.cond(self._cov_on_grads, lambda: _grad_mag(fake), lambda: fake)
            return _finite(self._cov_weight * _offdiag_cov_penalty(feat, as_correlation=True), "cov_pen")
        cov_pen = tf.cond(self._cov_weight > 0.0, _cov, lambda: tf.constant(0.0, tf.float32))

        # Optional low-pass penalty on delta
        lp = tf.cond(self._lambda_lp > 0.0,
                     lambda: _finite(self._lambda_lp * lowpass_l1_on_delta((fake - lr_up), k=self.lp_kernel), "lp"),
                     lambda: tf.constant(0.0, tf.float32))

        # Physics
        phys = tf.constant(0.0, tf.float32)
        if self.phys_params is not None and self.hr_mean is not None and self.hr_std is not None:
            wsum = self.w_vomega + self.w_omcons + self.w_div + self.w_mom + self.w_ppois
            def _phys():
                # NOTE: physics_steady MUST be a Python bool here.
                ph = physics_terms(fake, self.hr_mean, self.hr_std, self.phys_params, steady=bool(self.physics_steady))
                return _finite(
                    self.w_vomega * ph['vomega'] +
                    self.w_omcons * ph['omcons'] +
                    self.w_div    * ph['div'] +
                    self.w_mom    * ph['mom'] +
                    self.w_ppois  * ph['ppois'],
                    "phys",
                )
            # Keeping this tf.cond is fine now because _phys no longer traces a raising branch.
            phys = tf.cond(wsum > 0.0, _phys, lambda: tf.constant(0.0, tf.float32))

        # POD sidecar
        pod_loss = tf.constant(0.0, tf.float32)
        if self.pod_params is not None and self.lr_mean is not None and self.lr_std is not None and self.phys_params is not None:
            wsum = self.w_pod_vel + self.w_pod_p + self.w_pod_w
            def _pod():
                tim_u = tim_v = tim_p = None
                if pod_coeffs is not None and isinstance(pod_coeffs, (tuple, list)) and len(pod_coeffs) == 3:
                    tim_u, tim_v, tim_p = pod_coeffs
                lr_for_pod = lr
                if sr_scale == 2 and self.hr_mean is not None and self.hr_std is not None:
                    hr_phys = denorm(hr, self.hr_mean, self.hr_std)
                    hr_lr_phys = resize_nchw(hr_phys, (self.pod_params.h_lr, self.pod_params.w_lr), method="bilinear")
                    lr_for_pod = (tf.cast(hr_lr_phys, tf.float32) - tf.cast(self.lr_mean, tf.float32)) / (tf.cast(self.lr_std, tf.float32) + 1e-8)

                podt = pod_sidecar_losses(
                    pred_hr_n=fake,
                    lr_n=lr_for_pod,
                    hr_mean=self.hr_mean,
                    hr_std=self.hr_std,
                    lr_mean=self.lr_mean,
                    lr_std=self.lr_std,
                    phys=self.phys_params,
                    pod=self.pod_params,
                    tim_u=tim_u,
                    tim_v=tim_v,
                    tim_p=tim_p,
                    vel_blend_alpha=self.pod_vel_blend_alpha,
                )
                return _finite(
                    self.w_pod_vel * (podt["vel_cycle"] + podt["vel_coeff"]) +
                    self.w_pod_p   * (podt["p_cycle"]   + podt["p_coeff"]) +
                    self.w_pod_w   * podt["w_pod"],
                    "pod",
                )
            pod_loss = tf.cond(wsum > 0.0, _pod, lambda: tf.constant(0.0, tf.float32))

        terms = {
            "pix": pix,
            "spec": spec,
            "espec": espec,
            "res": res,
            "lowk": lowk,
            "tv": tv,
            "grad": grad,
            "decor": decor,
            "cov": cov_pen,
            "fm": fm,
            "gan": g_gan,
            "lp": lp,
            "phys": phys,
            "pod": pod_loss,
        }

        # Sum only finite terms (they're already _finite-wrapped)
        total = None
        for v in terms.values():
            total = v if total is None else (total + v)
        terms["total"] = tf.cast(total if total is not None else 0.0, tf.float32)
        return terms

    # NOTE ABOUT TF.COND + EAGER:
    # --------------------------
    # `_compute_terms()` relies on multiple `tf.cond(...)` branches where the
    # branch fns are created *inside* the call.
    #
    # If you call `_compute_terms()` (or an eager `eval_terms()`) inside a
    # Python loop (e.g. validation), TensorFlow will trace new branch graphs
    # repeatedly, which can cause *unbounded graph/kernels cache growth*.
    #
    # On GPUs that require PTX JIT compilation (e.g. SM90 w/ a TF build that
    # doesn't ship native cubins), this can become catastrophic and look like a
    # "random" hard-exit (SIGKILL / OOM) after a fixed amount of work.
    #
    # To prevent this, keep term-evaluation in a single cached graph.

    @tf.function(reduce_retracing=True)
    def eval_terms(self, lr, hr, pod_coeffs=None):
        """Deterministic per-term loss breakdown for validation.

        This is a `tf.function` specifically to avoid eager-mode `tf.cond`
        retracing/leaks when called repeatedly.
        """
        return self._compute_terms(lr, hr, pod_coeffs=pod_coeffs, deterministic=True)

    @tf.function(reduce_retracing=True)
    def train_terms(self, lr, hr, pod_coeffs=None):
        """Training-mode (stochastic) term breakdown.

        Used by optional debug printing without triggering eager retracing.
        """
        return self._compute_terms(lr, hr, pod_coeffs=pod_coeffs, deterministic=False)

    def forward(self, lr, hr, pod_coeffs=None):
        """Training-time objective (may include stochastic components)."""
        terms = self._compute_terms(lr, hr, pod_coeffs=pod_coeffs, deterministic=False)
        loss = terms["total"]
        loss = tf.where(tf.math.is_finite(loss), loss, tf.ones_like(loss))
        return tf.cast(loss, tf.float32)
