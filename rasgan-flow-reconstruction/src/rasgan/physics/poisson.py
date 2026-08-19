from __future__ import annotations

from ..env import tf
from typing import Tuple
from .operators import _to_f32


_PI = tf.constant(3.141592653589793, dtype=tf.float32)


def _fftfreq(n: int, d: float) -> tf.Tensor:
    """Return FFT frequencies (like numpy.fft.fftfreq) as float32 tensor."""
    n = int(n)
    d = float(d)
    val = 1.0 / (n * d)
    if n % 2 == 0:
        k_pos = tf.range(0, n // 2 + 1, dtype=tf.float32)
        k_neg = tf.range(-(n // 2) + 1, 0, dtype=tf.float32)
        k = tf.concat([k_pos, k_neg], axis=0)
    else:
        k_pos = tf.range(0, (n - 1) // 2 + 1, dtype=tf.float32)
        k_neg = tf.range(-((n - 1) // 2), 0, dtype=tf.float32)
        k = tf.concat([k_pos, k_neg], axis=0)
    return k * val


def poisson_solve_fft(rhs: tf.Tensor, dx: float = 1.0, dy: float = 1.0, zero_mean: bool = True) -> tf.Tensor:
    """Solve -∇² ψ = rhs on a periodic domain using FFT.

    Notes:
        - Assumes periodic boundaries.
        - Requires H,W to be statically known (typical for fixed-size training).
    Args:
        rhs: [N,1,H,W] float tensor
    Returns:
        psi: [N,1,H,W] float32
    """
    rhs = _to_f32(rhs)
    dx = float(dx); dy = float(dy)

    H = rhs.shape[2]
    W = rhs.shape[3]
    if H is None or W is None:
        raise ValueError("poisson_solve_fft requires static H,W. Use method='jacobi' if shape is dynamic.")

    kx = 2.0 * _PI * _fftfreq(int(W), dx)  # [W]
    ky = 2.0 * _PI * _fftfreq(int(H), dy)  # [H]
    kx2 = tf.reshape(kx * kx, [1, 1, 1, int(W)])
    ky2 = tf.reshape(ky * ky, [1, 1, int(H), 1])
    k2 = kx2 + ky2

    rhs_hat = tf.signal.fft2d(tf.cast(rhs, tf.complex64))
    k2_safe = tf.where(k2 == 0.0, tf.ones_like(k2), k2)
    psi_hat = rhs_hat / tf.cast(k2_safe, tf.complex64)

    if zero_mean:
        # Zero the DC mode for every batch item and channel. `k2` has shape
        # [1,1,H,W], so this mask broadcasts over the leading dimensions.
        dc_mask = tf.cast(tf.not_equal(k2, 0.0), tf.complex64)
        psi_hat = psi_hat * dc_mask

    psi = tf.math.real(tf.signal.ifft2d(psi_hat))
    return tf.cast(psi, tf.float32)


def poisson_solve_jacobi(rhs: tf.Tensor, dx: float = 1.0, dy: float = 1.0, iters: int = 200) -> tf.Tensor:
    """Solve -∇² ψ = rhs with unrolled Jacobi iterations (differentiable).

    Boundary: symmetric padding (approx Neumann). Works on non-periodic domains but is slower.
    """
    rhs = _to_f32(rhs)
    dx = float(dx); dy = float(dy)
    iters = int(iters)

    psi = tf.zeros_like(rhs, dtype=tf.float32)

    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for _ in range(iters):
        p = tf.pad(psi, [[0, 0], [0, 0], [1, 1], [1, 1]], mode="SYMMETRIC")
        psi_xp = p[:, :, 1:-1, 2:]
        psi_xm = p[:, :, 1:-1, :-2]
        psi_yp = p[:, :, 2:, 1:-1]
        psi_ym = p[:, :, :-2, 1:-1]
        psi = ((psi_xp + psi_xm) * dy2 + (psi_yp + psi_ym) * dx2 + rhs * dx2 * dy2) / denom

    return psi


def streamfunction_from_omega(
    omega: tf.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    method: str = "fft",
    jacobi_iters: int = 200,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute (u,v,psi) from vorticity ωz for 2D incompressible flow.

    Uses ψ such that u = ∂ψ/∂y, v = -∂ψ/∂x and ω = -∇²ψ.
    Args:
        omega: [N,1,H,W]
        method: 'fft' (periodic) or 'jacobi' (non-periodic)
    Returns:
        u, v, psi: each [N,1,H,W]
    """
    omega = _to_f32(omega)
    method = str(method).lower()

    if method == "fft":
        psi = poisson_solve_fft(omega, dx=dx, dy=dy, zero_mean=True)

        psi_hat = tf.signal.fft2d(tf.cast(psi, tf.complex64))
        H = psi.shape[2]
        W = psi.shape[3]
        if H is None or W is None:
            raise ValueError("streamfunction_from_omega(method='fft') requires static H,W.")

        kx = 2.0 * _PI * _fftfreq(int(W), dx)
        ky = 2.0 * _PI * _fftfreq(int(H), dy)
        kx = tf.reshape(kx, [1, 1, 1, int(W)])
        ky = tf.reshape(ky, [1, 1, int(H), 1])

        j = tf.complex(tf.constant(0.0, tf.float32), tf.constant(1.0, tf.float32))
        u_hat = j * tf.cast(ky, tf.complex64) * psi_hat
        v_hat = -j * tf.cast(kx, tf.complex64) * psi_hat

        u = tf.cast(tf.math.real(tf.signal.ifft2d(u_hat)), tf.float32)
        v = tf.cast(tf.math.real(tf.signal.ifft2d(v_hat)), tf.float32)
        return u, v, psi

    if method == "jacobi":
        psi = poisson_solve_jacobi(omega, dx=dx, dy=dy, iters=jacobi_iters)
        from .operators import grad_central
        dpsi_dx, dpsi_dy = grad_central(psi, dx=dx, dy=dy, bc="replicate")
        u = dpsi_dy
        v = -dpsi_dx
        return u, v, psi

    raise ValueError(f"Unknown method {method}; expected 'fft' or 'jacobi'")