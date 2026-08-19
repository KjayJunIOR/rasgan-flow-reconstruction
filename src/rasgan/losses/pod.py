from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..env import tf

from .physics import denorm, split_p_v_w, PhysicsParams
from ..physics.poisson import streamfunction_from_omega
from ..physics.operators import curl2d
from .gan import resize_nchw


@dataclass
class PodParams:
    """POD parameters held as TF tensors.

    Modes are flattened spatial bases.
    """
    phiu: tf.Tensor  # [HW, K]
    phiv: tf.Tensor  # [HW, K]
    phip: tf.Tensor  # [HW, K]
    h_lr: int
    w_lr: int


def _flatten_hw(x: tf.Tensor) -> tf.Tensor:
    # x: [N,1,H,W] -> [N,HW]
    x = tf.cast(x, tf.float32)
    return tf.reshape(x, [tf.shape(x)[0], -1])


def _project_scalar(x: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
    # x: [N,1,H,W] (physical), phi: [HW,K] -> coeff: [N,K]
    x_f = _flatten_hw(x)
    return tf.linalg.matmul(x_f, tf.cast(phi, tf.float32))


def _project_vector(u: tf.Tensor, v: tf.Tensor, phiu: tf.Tensor, phiv: tf.Tensor) -> tf.Tensor:
    # coeff = U*phiu + V*phiv (matches user's MATLAB TimCoeU + TimCoeV)
    return _project_scalar(u, phiu) + _project_scalar(v, phiv)


def _reconstruct_scalar(coeff: tf.Tensor, phi: tf.Tensor, h: int, w: int) -> tf.Tensor:
    # coeff: [N,K], phi: [HW,K] -> [N,1,H,W]
    phi_t = tf.transpose(tf.cast(phi, tf.float32), [1, 0])  # [K,HW]
    x_f = tf.linalg.matmul(tf.cast(coeff, tf.float32), phi_t)  # [N,HW]
    x = tf.reshape(x_f, [tf.shape(coeff)[0], 1, h, w])
    return x


def _reconstruct_vector(coeff: tf.Tensor, phiu: tf.Tensor, phiv: tf.Tensor, h: int, w: int) -> Tuple[tf.Tensor, tf.Tensor]:
    return (
        _reconstruct_scalar(coeff, phiu, h, w),
        _reconstruct_scalar(coeff, phiv, h, w),
    )


def _downsample_nchw(pred_hr: tf.Tensor, h_lr: int, w_lr: int) -> tf.Tensor:
    """Downsample NCHW tensor to (h_lr, w_lr) with gradients.

    Important: Your LR/HR are the same physical simulation data sampled on
    different grids (interpolated), so the appropriate restriction operator is
    *interpolation-based* (not box-averaging / area pooling).

    We therefore use differentiable interpolation resize (default: bilinear).
    If you want a closer match to MATLAB's "cubic" interpolation, change
    method to "bicubic".
    """
    # If already on the desired grid, avoid emitting a resize op.
    if pred_hr.shape.rank == 4:
        if pred_hr.shape[2] is not None and pred_hr.shape[3] is not None:
            if int(pred_hr.shape[2]) == int(h_lr) and int(pred_hr.shape[3]) == int(w_lr):
                return pred_hr

    h = tf.shape(pred_hr)[2]
    w = tf.shape(pred_hr)[3]
    return tf.cond(
        tf.logical_and(tf.equal(h, int(h_lr)), tf.equal(w, int(w_lr))),
        lambda: pred_hr,
        lambda: resize_nchw(pred_hr, (h_lr, w_lr), method="bilinear"),
    )

def pod_sidecar_losses(
    *,
    pred_hr_n: tf.Tensor,
    lr_n: tf.Tensor,
    hr_mean: tf.Tensor,
    hr_std: tf.Tensor,
    lr_mean: tf.Tensor,
    lr_std: tf.Tensor,
    phys: PhysicsParams,
    pod: PodParams,
    tim_u: Optional[tf.Tensor] = None,
    tim_v: Optional[tf.Tensor] = None,
    tim_p: Optional[tf.Tensor] = None,
    vel_blend_alpha: float = 0.0,  # 0=full vector (original), 1=v-only
) -> Dict[str, tf.Tensor]:
    """Compute POD-based consistency losses.

    All losses operate in LR space (pod.h_lr, pod.w_lr).

    Returns dict with:
      - vel_cycle, vel_coeff
      - p_cycle,   p_coeff
      - w_pod

    Notes:
      * u is inferred from omega using the same streamfunction solver as the physics losses.
      * If tim_* tensors are provided, coefficient supervision terms are computed.
    """
    # Denormalize to physical units because POD operators/modes were computed
    # on physical-unit LR fields.
    # IMPORTANT: we will *return dimensionless losses* by normalizing errors
    # using dataset scales (lr_std) so the POD terms are comparable to the
    # other losses that operate in normalized space.
    pred_hr = denorm(pred_hr_n, hr_mean, hr_std)  # physical HR
    lr = denorm(lr_n, lr_mean, lr_std)            # physical LR (already POD-reconstructed)

    # Map prediction to LR grid
    pred_lr = _downsample_nchw(pred_hr, pod.h_lr, pod.w_lr)

    p_lr, v_lr, w_lr = split_p_v_w(lr)
    p_pr, v_pr, w_pr = split_p_v_w(pred_lr)

    # ---- normalization scales (physical -> dimensionless) ----
    # Do NOT trust dataset-provided stds here: many setups normalize channels
    # in different ways (z-score, min-max, etc.), and feeding SI-unit POD modes
    # through those scales can explode the loss.
    #
    # Instead, normalize each POD error by the *batch's* LR magnitude for that
    # channel (mean absolute value). This makes the POD terms roughly O(1)
    # regardless of units.
    # Use *prediction* magnitudes for scaling since cycle losses are defined
    # w.r.t. the prediction. Stop gradients so scale can't be "gamed".
    s_p = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(p_pr)), tf.constant(1e-6, tf.float32)))
    s_v = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(v_pr)), tf.constant(1e-6, tf.float32)))
    s_w = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(w_pr)), tf.constant(1e-6, tf.float32)))

    # ---- LR grid spacing ----
    # The user provides phys.dx/phys.dy in *HR* physical units (meters per cell
    # on the HR grid). For LR-domain operations (e.g. inferring u from LR omega),
    # scale spacings assuming LR and HR span the same physical domain.
    #
    # Compute LR grid spacings in **Python floats**.
    #
    # Our Poisson/derivative operators in rasgan/physics assume dx,dy are Python
    # scalars (they call float(dx)/float(dy)). Passing a Tensor here will break
    # under tf.function tracing. Since SR uses fixed spatial sizes, we can safely
    # derive the required sizes from static shapes.
    #
    # NOTE: avoid name collisions with the vorticity tensor `w_lr`.
    w_hr_size = int(pred_hr.shape[3])  # NCHW
    h_hr_size = int(pred_hr.shape[2])
    w_lr_size = int(pod.w_lr)
    h_lr_size = int(pod.h_lr)
    w_hr_cells = max(w_hr_size - 1, 1)
    h_hr_cells = max(h_hr_size - 1, 1)
    w_lr_cells = max(w_lr_size - 1, 1)
    h_lr_cells = max(h_lr_size - 1, 1)

    # phys.dx/phys.dy are HR grid spacings; scale to LR based on cell-count ratio.
    dx_lr = float(phys.dx) * (w_hr_cells / w_lr_cells)
    dy_lr = float(phys.dy) * (h_hr_cells / h_lr_cells)

    # infer u from omega for both LR and pred
    u_lr, v_from_w_lr, _ = streamfunction_from_omega(
        w_lr,
        dx=dx_lr,
        dy=dy_lr,
        method=phys.poisson_method,
        jacobi_iters=phys.poisson_iters,
    )
    u_pr, v_from_w_pr, _ = streamfunction_from_omega(
        w_pr,
        dx=dx_lr,
        dy=dy_lr,
        method=phys.poisson_method,
        jacobi_iters=phys.poisson_iters,
    )
    # NOTE:
    # u is not a model output; inferring u from omega early in training can be very noisy
    # and will corrupt the shared vector-POD coefficient estimate if used in projection.
    # We still keep dx_lr/dy_lr for curl(u,v) below.

    # --- velocity POD cycle ---
    # Vector-POD uses one shared coefficient vector c, but here we estimate c from v only
    # (the observable/predicted channel) to avoid injecting omega->u inversion noise.
    # This still reconstructs (u,v) with the same c and paired (phiu, phiv) modes.
    alpha = tf.cast(vel_blend_alpha, tf.float32)
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)

    c_v = _project_scalar(v_pr, pod.phiv)      # [N,K]
    c_u = _project_scalar(u_pr, pod.phiu)      # [N,K]
    c_vel_pr = c_v + (1.0 - alpha) * c_u       # [N,K]
    u4_pr, v4_pr = _reconstruct_vector(c_vel_pr, pod.phiu, pod.phiv, pod.h_lr, pod.w_lr)

    # NOTE: model I/O channels are (p, v, w) (no explicit u-channel).
    # We infer u from omega for POD projection/reconstruction, but enforcing
    # a direct u-cycle term can be brittle because u-from-omega depends on
    # boundary/mean assumptions. For robustness we enforce the POD cycle
    # only on the observed v-channel; vorticity consistency below still
    # couples u and v via curl(u,v).
    L_vel_cycle = tf.reduce_mean(tf.abs(v4_pr - v_lr)) / s_v

    # coefficient supervision (optional)
    if tim_u is not None and tim_v is not None:
        c_vel_gt = (1.0 - alpha) * tf.cast(tim_u, tf.float32) + tf.cast(tim_v, tf.float32)
        # Normalize coefficient errors to avoid exploding scales.
        c_vel_scale = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(c_vel_gt)), tf.constant(1e-6, tf.float32)))
        L_vel_coeff = tf.reduce_mean(tf.abs(c_vel_pr - c_vel_gt)) / c_vel_scale
    else:
        L_vel_coeff = tf.constant(0.0, dtype=tf.float32)

    # --- pressure POD cycle ---
    c_p_pr = _project_scalar(p_pr, pod.phip)
    p4_pr = _reconstruct_scalar(c_p_pr, pod.phip, pod.h_lr, pod.w_lr)
    L_p_cycle = tf.reduce_mean(tf.abs(p4_pr - p_lr)) / s_p

    if tim_p is not None:
        c_p_gt = tf.cast(tim_p, tf.float32)
        c_p_scale = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(c_p_gt)), tf.constant(1e-6, tf.float32)))
        L_p_coeff = tf.reduce_mean(tf.abs(c_p_pr - c_p_gt)) / c_p_scale
    else:
        L_p_coeff = tf.constant(0.0, dtype=tf.float32)

    # --- vorticity POD consistency ---
    # vorticity from POD-reconstructed velocity should match LR omega (which was created from POD velocity)
    w4_pr = curl2d(u4_pr, v4_pr, dx=dx_lr, dy=dy_lr, bc=phys.bc)
    L_w_pod = tf.reduce_mean(tf.abs(w4_pr - w_lr)) / s_w

    return {
        "vel_cycle": tf.cast(L_vel_cycle, tf.float32),
        "vel_coeff": tf.cast(L_vel_coeff, tf.float32),
        "p_cycle": tf.cast(L_p_cycle, tf.float32),
        "p_coeff": tf.cast(L_p_coeff, tf.float32),
        "w_pod": tf.cast(L_w_pod, tf.float32),
    }
