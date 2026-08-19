from __future__ import annotations

from ..env import tf
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from ..physics.operators import grad_central, laplacian, divergence, curl2d
from ..physics.poisson import streamfunction_from_omega


@dataclass
class PhysicsParams:
    dx: float = 1.0
    dy: float = 1.0
    nu: float = 0.0      # kinematic viscosity (set >0 if known)
    rho: float = 1.0     # density (often 1 in nondimensional units)
    bc: str = "replicate"  # 'replicate' or 'periodic'
    poisson_method: str = "fft"  # 'fft' or 'jacobi'
    poisson_iters: int = 200


def denorm(x_n: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    """Denormalize NCHW tensor with broadcastable mean/std."""
    x_n = tf.cast(x_n, tf.float32)
    return x_n * tf.cast(std, tf.float32) + tf.cast(mean, tf.float32)


def split_p_v_w(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split [N,3,H,W] into (p, v, w) each [N,1,H,W]."""
    return x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]


def physics_terms(
    pred: tf.Tensor,
    mean: tf.Tensor,
    std: tf.Tensor,
    params: PhysicsParams,
    *,
    steady: bool = True,
    u_t: Optional[tf.Tensor] = None,
    v_t: Optional[tf.Tensor] = None,
) -> Dict[str, tf.Tensor]:
    """Compute physics-informed loss terms for outputs (p, v, ω).

    Assumes 2D incompressible flow with streamfunction formulation.
    We infer u from ω by solving -∇²ψ = ω and using:
        u = ∂ψ/∂y, v_ω = -∂ψ/∂x

    Returns dict with:
        vomega, omcons, div, mom, ppois
    """
    pred_phys = denorm(pred, mean, std)
    p, v, w = split_p_v_w(pred_phys)

    # infer u and v_omega from omega
    u_w, v_w, psi = streamfunction_from_omega(
        w,
        dx=params.dx,
        dy=params.dy,
        method=params.poisson_method,
        jacobi_iters=params.poisson_iters,
    )

    # v-omega compatibility
    L_vomega = tf.reduce_mean(tf.abs(v - v_w))

    # ω consistency using inferred u and predicted v
    w_cons = curl2d(u_w, v, dx=params.dx, dy=params.dy, bc=params.bc)
    L_omcons = tf.reduce_mean(tf.abs(w - w_cons))

    # divergence-free: using inferred u and predicted v
    div_uv = divergence(u_w, v, dx=params.dx, dy=params.dy, bc=params.bc)
    L_div = tf.reduce_mean(tf.abs(div_uv))

    # Momentum residuals (optional; steady by default)
    # u, v are in physical units. p is physical too.
    du_dx, du_dy = grad_central(u_w, dx=params.dx, dy=params.dy, bc=params.bc)
    dv_dx, dv_dy = grad_central(v,   dx=params.dx, dy=params.dy, bc=params.bc)
    dp_dx, dp_dy = grad_central(p,   dx=params.dx, dy=params.dy, bc=params.bc)

    lap_u = laplacian(u_w, dx=params.dx, dy=params.dy, bc=params.bc)
    lap_v = laplacian(v,   dx=params.dx, dy=params.dy, bc=params.bc)

    if steady:
        u_t_term = 0.0
        v_t_term = 0.0
    else:
        if u_t is None or v_t is None:
            raise ValueError("For unsteady residuals, provide u_t and v_t.")
        # expect u_t/v_t already in physical units and shape [N,1,H,W]
        u_t_term = tf.cast(u_t, tf.float32)
        v_t_term = tf.cast(v_t, tf.float32)

    rho = float(params.rho)
    nu  = float(params.nu)

    # r_u = u_t + u*u_x + v*u_y + (1/rho) p_x - nu ∇²u
    r_u = u_t_term + (u_w * du_dx + v * du_dy) + (dp_dx / rho) - (nu * lap_u)
    r_v = v_t_term + (u_w * dv_dx + v * dv_dy) + (dp_dy / rho) - (nu * lap_v)

    L_mom = tf.reduce_mean(tf.abs(r_u)) + tf.reduce_mean(tf.abs(r_v))

    # Pressure Poisson residual (optional)
    # ∇²p ≈ -rho[(u_x)^2 + 2 u_y v_x + (v_y)^2]
    lap_p = laplacian(p, dx=params.dx, dy=params.dy, bc=params.bc)
    rhs_p = -rho * (tf.square(du_dx) + 2.0 * du_dy * dv_dx + tf.square(dv_dy))
    L_ppois = tf.reduce_mean(tf.abs(lap_p - rhs_p))

    return {
        "vomega": tf.cast(L_vomega, tf.float32),
        "omcons": tf.cast(L_omcons, tf.float32),
        "div":    tf.cast(L_div, tf.float32),
        "mom":    tf.cast(L_mom, tf.float32),
        "ppois":  tf.cast(L_ppois, tf.float32),
    }