from __future__ import annotations

from ..env import tf
from typing import Tuple
def _to_f32(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(x, tf.float32) if x.dtype != tf.float32 else x


def grad_central(f: tf.Tensor, dx: float = 1.0, dy: float = 1.0, bc: str = "replicate") -> Tuple[tf.Tensor, tf.Tensor]:
    """Central-difference gradients for NCHW tensors.

    Args:
        f: Tensor [N,C,H,W]
        dx, dy: grid spacing
        bc: 'replicate' (default) or 'periodic'
    Returns:
        (df_dx, df_dy) each [N,C,H,W]
    """
    f = _to_f32(f)
    dx = float(dx); dy = float(dy)
    if bc not in ("replicate", "periodic"):
        raise ValueError(f"bc must be 'replicate' or 'periodic', got {bc}")

    if bc == "periodic":
        f_xp = tf.roll(f, shift=-1, axis=3)
        f_xm = tf.roll(f, shift=+1, axis=3)
        f_yp = tf.roll(f, shift=-1, axis=2)
        f_ym = tf.roll(f, shift=+1, axis=2)
        df_dx = (f_xp - f_xm) / (2.0 * dx)
        df_dy = (f_yp - f_ym) / (2.0 * dy)
        return df_dx, df_dy

    # replicate / Neumann-ish boundaries via edge replication
    # pad: [[N],[C],[H],[W]]
    f_pad_x = tf.pad(f, [[0,0],[0,0],[0,0],[1,1]], mode="SYMMETRIC")
    f_pad_y = tf.pad(f, [[0,0],[0,0],[1,1],[0,0]], mode="SYMMETRIC")

    df_dx = (f_pad_x[:,:,:,2:] - f_pad_x[:,:,:,:-2]) / (2.0 * dx)
    df_dy = (f_pad_y[:,:,2:,:] - f_pad_y[:,:,:-2,:]) / (2.0 * dy)
    return df_dx, df_dy


def laplacian(f: tf.Tensor, dx: float = 1.0, dy: float = 1.0, bc: str = "replicate") -> tf.Tensor:
    """5-point Laplacian for NCHW tensors."""
    f = _to_f32(f)
    dx = float(dx); dy = float(dy)
    if bc not in ("replicate", "periodic"):
        raise ValueError(f"bc must be 'replicate' or 'periodic', got {bc}")

    if bc == "periodic":
        f_xp = tf.roll(f, shift=-1, axis=3)
        f_xm = tf.roll(f, shift=+1, axis=3)
        f_yp = tf.roll(f, shift=-1, axis=2)
        f_ym = tf.roll(f, shift=+1, axis=2)
        return (f_xp - 2.0*f + f_xm) / (dx*dx) + (f_yp - 2.0*f + f_ym) / (dy*dy)

    f_pad = tf.pad(f, [[0,0],[0,0],[1,1],[1,1]], mode="SYMMETRIC")
    center = f_pad[:,:,1:-1,1:-1]
    f_xp = f_pad[:,:,1:-1,2:]
    f_xm = f_pad[:,:,1:-1,:-2]
    f_yp = f_pad[:,:,2:,1:-1]
    f_ym = f_pad[:,:,:-2,1:-1]
    return (f_xp - 2.0*center + f_xm) / (dx*dx) + (f_yp - 2.0*center + f_ym) / (dy*dy)


def divergence(u: tf.Tensor, v: tf.Tensor, dx: float = 1.0, dy: float = 1.0, bc: str = "replicate") -> tf.Tensor:
    """Compute div(u,v) = du/dx + dv/dy for NCHW tensors."""
    du_dx, _ = grad_central(u, dx=dx, dy=dy, bc=bc)
    _, dv_dy = grad_central(v, dx=dx, dy=dy, bc=bc)
    return du_dx + dv_dy


def curl2d(u: tf.Tensor, v: tf.Tensor, dx: float = 1.0, dy: float = 1.0, bc: str = "replicate") -> tf.Tensor:
    """Compute ωz = dv/dx - du/dy for NCHW tensors."""
    dv_dx, _ = grad_central(v, dx=dx, dy=dy, bc=bc)
    _, du_dy = grad_central(u, dx=dx, dy=dy, bc=bc)
    return dv_dx - du_dy