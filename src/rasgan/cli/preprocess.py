from __future__ import annotations

"""Prepare paired low-/high-resolution flow fields for RASGAN.

The public data contract is NCHW in the output HDF5 file. Input arrays can be
MATLAB v7.3/HDF5 or classic MATLAB files and can use C,N,H,W or N,C,H,W.
Normalization statistics are computed from the high-resolution training split
only and then applied to both LR and HR arrays, matching the original workflow.
"""

import argparse
from pathlib import Path
from typing import Literal

import h5py
import numpy as np


def _load_array(path: str, key: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    try:
        with h5py.File(p, "r") as f:
            if key in f:
                return np.asarray(f[key])
            lowered = {name.lower(): name for name in f.keys()}
            match = lowered.get(key.lower())
            if match is not None:
                return np.asarray(f[match])
    except OSError:
        pass

    try:
        import scipy.io as sio
    except Exception as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("SciPy is required to read classic MATLAB files.") from exc

    mat = sio.loadmat(p)
    if key in mat:
        return np.asarray(mat[key])
    lowered = {name.lower(): name for name in mat if not name.startswith("__")}
    match = lowered.get(key.lower())
    if match is None:
        present = sorted(name for name in mat if not name.startswith("__"))
        raise KeyError(f"Could not find {key!r} in {path}. Available variables: {present}")
    return np.asarray(mat[match])


def _to_nchw(x: np.ndarray, layout: Literal["cnhw", "nchw"]) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"Expected a rank-4 array, got shape {x.shape}")
    if layout == "cnhw":
        return np.transpose(x, (1, 0, 2, 3))
    return x


def prepare_dataset(
    *,
    hr_path: str,
    lr_path: str,
    output_path: str,
    hr_key: str = "HR",
    lr_key: str = "LR",
    input_layout: Literal["cnhw", "nchw"] = "cnhw",
    train_count: int | None = None,
    train_fraction: float = 0.9,
    seed: int | None = 123,
    compression: int = 4,
) -> Path:
    hr = _to_nchw(_load_array(hr_path, hr_key), input_layout).astype(np.float32, copy=False)
    lr = _to_nchw(_load_array(lr_path, lr_key), input_layout).astype(np.float32, copy=False)

    if hr.shape[0] != lr.shape[0]:
        raise ValueError(f"HR and LR must contain the same number of samples: {hr.shape} vs {lr.shape}")
    if hr.shape[1] != lr.shape[1]:
        raise ValueError(f"HR and LR must contain the same channels: {hr.shape} vs {lr.shape}")
    if hr.shape[1] != 3:
        raise ValueError(
            f"The current RASGAN generator and losses expect three channels; got C={hr.shape[1]}."
        )

    n = int(hr.shape[0])
    if n < 2:
        raise ValueError("At least two paired samples are required.")
    if train_count is None:
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        train_count = int(round(n * float(train_fraction)))
    train_count = int(train_count)
    if not 1 <= train_count < n:
        raise ValueError(f"train_count must be in [1, {n - 1}], got {train_count}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n) if seed is not None else np.random.default_rng().permutation(n)
    hr = hr[perm]
    lr = lr[perm]

    eps = np.float32(1e-8)
    means = np.mean(hr[:train_count], axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    stds = np.std(hr[:train_count], axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    stds = np.maximum(stds, eps)

    hr_norm = (hr - means[None, :, None, None]) / stds[None, :, None, None]
    lr_norm = (lr - means[None, :, None, None]) / stds[None, :, None, None]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("meanshr", data=means)
        meta.create_dataset("stdshr", data=stds)
        meta.create_dataset("meanslr", data=means)
        meta.create_dataset("stdslr", data=stds)
        meta.create_dataset("perm", data=perm.astype(np.int64))
        meta.create_dataset("train_idx", data=perm[:train_count].astype(np.int64))
        meta.create_dataset("test_idx", data=perm[train_count:].astype(np.int64))
        meta.attrs["normalization"] = "HR training mean/std applied to HR and LR"
        meta.attrs["input_layout"] = input_layout
        meta.attrs["source_hr"] = str(Path(hr_path))
        meta.attrs["source_lr"] = str(Path(lr_path))

        train = f.create_group("train")
        test = f.create_group("test")
        kwargs = dict(compression="gzip", compression_opts=int(compression), chunks=True)
        train.create_dataset("lr", data=lr_norm[:train_count], **kwargs)
        train.create_dataset("hr", data=hr_norm[:train_count], **kwargs)
        test.create_dataset("lr", data=lr_norm[train_count:], **kwargs)
        test.create_dataset("hr", data=hr_norm[train_count:], **kwargs)

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare paired flow-field arrays for RASGAN training.")
    p.add_argument("--hr", required=True, help="High-resolution/reference MAT or HDF5 file")
    p.add_argument("--lr", required=True, help="Low-resolution/POD MAT or HDF5 file")
    p.add_argument("--out", required=True, help="Output HDF5 path")
    p.add_argument("--hr-key", default="HR", help="HR variable/dataset name")
    p.add_argument("--lr-key", default="LR", help="LR variable/dataset name")
    p.add_argument("--input-layout", choices=["cnhw", "nchw"], default="cnhw")
    split = p.add_mutually_exclusive_group()
    split.add_argument("--train-count", type=int, default=None)
    split.add_argument("--train-fraction", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=123, help="Shuffle seed; use a negative value for a random seed")
    p.add_argument("--compression", type=int, default=4, choices=range(0, 10))
    return p


def main() -> None:
    args = build_parser().parse_args()
    seed = None if int(args.seed) < 0 else int(args.seed)
    out = prepare_dataset(
        hr_path=args.hr,
        lr_path=args.lr,
        output_path=args.out,
        hr_key=args.hr_key,
        lr_key=args.lr_key,
        input_layout=args.input_layout,
        train_count=args.train_count,
        train_fraction=args.train_fraction,
        seed=seed,
        compression=args.compression,
    )
    print(f"Saved normalized paired dataset to {out}")


if __name__ == "__main__":
    main()
