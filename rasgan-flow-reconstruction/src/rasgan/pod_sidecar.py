from __future__ import annotations

"""POD sidecar loader.

This project can optionally use POD-consistency losses driven by a MATLAB .mat
sidecar file. The POD is assumed to be computed on LR-space fields (e.g. 96x96)
using a method-of-snapshots formulation similar to the user's MATLAB code.

Expected variables (case-insensitive):
  Required:
    - phiu, phiv : velocity POD spatial modes (flattened), shape [HW, M]
    - phip       : pressure POD spatial modes (flattened), shape [HW, M]

  Optional (for coefficient supervision; should align with the TRAIN split order
  in the H5 file):
    - TimCoeU, TimCoeV : velocity time coefficients, shape [N_train, M]
    - TimCoeP          : pressure time coefficients, shape [N_train, M]

We only use the first K modes (TrainConfig.pod_k).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np


@dataclass(frozen=True)
class PodSidecar:
    # Spatial modes (flattened): [HW, K]
    phiu: np.ndarray
    phiv: np.ndarray
    phip: np.ndarray

    # Optional per-sample coeffs (aligned to train set): [N, K]
    tim_u: Optional[np.ndarray] = None
    tim_v: Optional[np.ndarray] = None
    tim_p: Optional[np.ndarray] = None


def _norm_key(k: str) -> str:
    return k.strip().lower().replace(" ", "").replace("_", "")


def _pick_var(mat: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
    """Pick the first matching variable name from a loaded .mat dict."""
    # Build normalized key map
    norm_map = {_norm_key(k): k for k in mat.keys()}
    for n in names:
        kk = norm_map.get(_norm_key(n))
        if kk is not None:
            v = mat[kk]
            # squeeze MATLAB 2D/1D quirks
            v = np.asarray(v)
            return v
    return None


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # MATLAB may store as (HW, M) or (M, HW); we don't guess transpose here.
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    return x.astype(np.float32, copy=False)


def _maybe_transpose_modes(x: np.ndarray) -> np.ndarray:
    """Accept (HW, M) or MATLAB-style (M, HW) and return (HW, M).

    Heuristic: if one dimension is "small" (<=256) and the other is large,
    treat the small dim as mode-count.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        return x
    a, b = x.shape
    # Common for POD: M is small (e.g., 4..O(100)), HW is large (e.g., 96*96=9216)
    if a <= 256 and b > 256:
        return x.T
    return x


def _maybe_transpose_coeffs(x: np.ndarray) -> np.ndarray:
    """Accept (N, M) or MATLAB-style (M, N) and return (N, M)."""
    x = np.asarray(x)
    if x.ndim != 2:
        return x
    a, b = x.shape
    if a <= 256 and b > 256:
        return x.T
    return x


def _load_mat_any(mat_path: str) -> Dict[str, Any]:
    """Load either MATLAB v7/v7.2 (scipy.io.loadmat) or v7.3 (HDF5 via h5py)."""
    # First try scipy (works for <= v7.2)
    try:
        import scipy.io as sio
        try:
            return sio.loadmat(mat_path)
        except NotImplementedError:
            # v7.3 is HDF5; scipy raises NotImplementedError
            pass
        except ValueError:
            # Some scipy versions raise ValueError for v7.3
            pass
    except Exception:
        # We'll fall back to h5py.
        pass

    # Fallback: MATLAB v7.3 HDF5
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "Failed to load .mat via scipy.io (likely v7.3) and h5py is not available. "
            "Install h5py or save the .mat in v7.2 format."
        ) from e

    out: Dict[str, Any] = {}
    with h5py.File(mat_path, "r") as f:
        # MATLAB v7.3 stores variables as datasets or groups at the root.
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                out[k] = np.array(obj)
    return out


def load_pod_sidecar(mat_path: str, *, k: int = 4) -> PodSidecar:
    """Load POD modes (+ optional coeffs) from a MATLAB .mat file."""
    mat = _load_mat_any(mat_path)

    phiu = _pick_var(mat, "phiu", "phi_u", "phix", "phixlr")
    phiv = _pick_var(mat, "phiv", "phi_v", "phiy", "phiylr")
    phip = _pick_var(mat, "phip", "phi_p", "phipressure", "phip_lr")

    if phiu is None or phiv is None or phip is None:
        present = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            "POD sidecar .mat must contain phiu, phiv, phip (case-insensitive). "
            f"Present variables: {present}"
        )

    phiu = _maybe_transpose_modes(_as_2d_float(phiu))
    phiv = _maybe_transpose_modes(_as_2d_float(phiv))
    phip = _maybe_transpose_modes(_as_2d_float(phip))

    # Slice first K modes
    k = int(k)
    if phiu.shape[1] < k or phiv.shape[1] < k or phip.shape[1] < k:
        raise ValueError(
            f"Requested K={k} modes but got shapes phiu={phiu.shape}, phiv={phiv.shape}, phip={phip.shape}"
        )
    phiu = phiu[:, :k]
    phiv = phiv[:, :k]
    phip = phip[:, :k]

    # Optional coeffs
    tim_u = _pick_var(mat, "timcoeu", "timcoe_u", "timcoeu_train")
    tim_v = _pick_var(mat, "timcoev", "timcoe_v", "timcoev_train")
    tim_p = _pick_var(mat, "timcoep", "timcoe_p", "timcoep_train")

    if tim_u is not None:
        tim_u = _maybe_transpose_coeffs(_as_2d_float(tim_u))[:, :k]
    if tim_v is not None:
        tim_v = _maybe_transpose_coeffs(_as_2d_float(tim_v))[:, :k]
    if tim_p is not None:
        tim_p = _maybe_transpose_coeffs(_as_2d_float(tim_p))[:, :k]

    return PodSidecar(phiu=phiu, phiv=phiv, phip=phip, tim_u=tim_u, tim_v=tim_v, tim_p=tim_p)
