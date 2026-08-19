
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
import h5py

if TYPE_CHECKING:
    import tensorflow as tf


@dataclass(frozen=True)
class H5Data:
    lr_train: np.ndarray
    hr_train: np.ndarray
    lr_test: np.ndarray
    hr_test: np.ndarray
    means_hr: np.ndarray
    stds_hr: np.ndarray
    means_lr: np.ndarray
    stds_lr: np.ndarray
    # Optional: permutation indices used when shuffling the original snapshots
    # before splitting into train/test. If present, this is a 1D int array of
    # length N_total where train samples are perm[:N_train].
    perm: np.ndarray | None = None

def _h5_dataset_paths(f: h5py.File) -> list[str]:
    """Return all dataset paths (not groups) within an H5 file."""
    paths: list[str] = []

    def _visitor(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    f.visititems(_visitor)
    return paths


def _h5_get_any(f: h5py.File, candidates: list[str], *, label: str) -> np.ndarray:
    """
    Fetch the first existing dataset among candidates.

    Supports:
      - exact matches (e.g. "lr_train")
      - group paths (e.g. "train/lr")
      - case-insensitive matches
    """
    # Fast exact/path lookup
    for key in candidates:
        if key in f:
            return np.asarray(f[key])

    # Case-insensitive fallback across all datasets
    ds_paths = _h5_dataset_paths(f)
    lower_map = {p.lower(): p for p in ds_paths}
    for key in candidates:
        p = lower_map.get(key.lower())
        if p is not None:
            return np.asarray(f[p])

    raise KeyError(
        f"Could not find required dataset for '{label}'. Tried: {candidates}. "
        f"Available datasets: {ds_paths}"
    )


def _to_nchw(x: np.ndarray) -> np.ndarray:
    """Convert NHWC -> NCHW if it looks like channels_last."""
    if x.ndim == 4:
        # Heuristic: channels_last if last dim is small (1/3/4) and second dim is not.
        if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
            return np.transpose(x, (0, 3, 1, 2))
    return x


def load_h5(path: str) -> H5Data:
    """
    Load a dataset H5 file into memory.

    The original project expected datasets named:
      lr_train, hr_train, lr_test, hr_test, means_hr, stds_hr, means_lr, stds_lr

    This TF-only version is more flexible and will also accept common alternatives,
    like group-based layouts:
      train/lr, train/hr, test/lr, test/hr
    """
    with h5py.File(path, "r") as f:
        lr_train = _h5_get_any(
            f,
            candidates=[
                "lr_train", "x_train", "train/lr", "lr/train", "train_lr", "lrTrain",
                "train_lr_images", "train_lr_data",
            ],
            label="lr_train",
        )
        hr_train = _h5_get_any(
            f,
            candidates=[
                "hr_train", "y_train", "train/hr", "hr/train", "train_hr", "hrTrain",
                "train_hr_images", "train_hr_data",
            ],
            label="hr_train",
        )
        lr_test = _h5_get_any(
            f,
            candidates=[
                "lr_test", "x_test", "test/lr", "lr/test", "valid/lr", "val/lr",
                "lr_val", "lr_valid", "lr_validation",
            ],
            label="lr_test",
        )
        hr_test = _h5_get_any(
            f,
            candidates=[
                "hr_test", "y_test", "test/hr", "hr/test", "valid/hr", "val/hr",
                "hr_val", "hr_valid", "hr_validation",
            ],
            label="hr_test",
        )

        # Normalization stats are optional in some exports; default to zeros/ones if missing.
        try:
            # Common naming conventions:
            # - flat: means_hr / stds_hr
            # - grouped: stats/means_hr
            # - this project's exporter: meta/meanshr, meta/stdshr, meta/meanslr, meta/stdslr
            means_hr = _h5_get_any(
                f,
                [
                    "means_hr", "mean_hr", "hr_means",
                    "stats/means_hr", "norm/means_hr",
                    "meta/meanshr", "meta/means_hr", "meta/meanhr",
                ],
                label="means_hr",
            )
            stds_hr = _h5_get_any(
                f,
                [
                    "stds_hr", "std_hr", "hr_stds",
                    "stats/stds_hr", "norm/stds_hr",
                    "meta/stdshr", "meta/stds_hr", "meta/std_hr", "meta/stdhr",
                ],
                label="stds_hr",
            )
            means_lr = _h5_get_any(
                f,
                [
                    "means_lr", "mean_lr", "lr_means",
                    "stats/means_lr", "norm/means_lr",
                    "meta/meanslr", "meta/means_lr", "meta/meanlr",
                ],
                label="means_lr",
            )
            stds_lr = _h5_get_any(
                f,
                [
                    "stds_lr", "std_lr", "lr_stds",
                    "stats/stds_lr", "norm/stds_lr",
                    "meta/stdslr", "meta/stds_lr", "meta/std_lr", "meta/stdlr",
                ],
                label="stds_lr",
            )
        except KeyError:
            # If any are missing, fall back to safe defaults.
            # Infer channel dim if possible; otherwise scalar.
            c_hr = hr_train.shape[1] if hr_train.ndim == 4 else 1
            c_lr = lr_train.shape[1] if lr_train.ndim == 4 else 1
            means_hr = np.zeros((c_hr,), dtype=np.float32)
            stds_hr  = np.ones((c_hr,), dtype=np.float32)
            means_lr = np.zeros((c_lr,), dtype=np.float32)
            stds_lr  = np.ones((c_lr,), dtype=np.float32)

        # Optional shuffle permutation indices (used to align external sidecars
        # like POD coefficients with the H5 training set order).
        perm = None
        try:
            perm = _h5_get_any(
                f,
                candidates=[
                    "perm", "shuffle_perm", "permutation",
                    "meta/perm", "meta/permutation", "meta/shuffle_perm",
                ],
                label="perm",
            )
        except KeyError:
            perm = None

    # Ensure float32 + NCHW
    lr_train = _to_nchw(lr_train).astype(np.float32, copy=False)
    hr_train = _to_nchw(hr_train).astype(np.float32, copy=False)
    lr_test  = _to_nchw(lr_test).astype(np.float32, copy=False)
    hr_test  = _to_nchw(hr_test).astype(np.float32, copy=False)

    return H5Data(
        lr_train=lr_train,
        hr_train=hr_train,
        lr_test=lr_test,
        hr_test=hr_test,
        means_hr=np.asarray(means_hr, dtype=np.float32),
        stds_hr=np.asarray(stds_hr, dtype=np.float32),
        means_lr=np.asarray(means_lr, dtype=np.float32),
        stds_lr=np.asarray(stds_lr, dtype=np.float32),
        perm=(None if perm is None else np.asarray(perm, dtype=np.int64).reshape(-1)),
    )



def make_loaders(
    h5: H5Data,
    batch_size: int,
    shuffle: bool = True,
    *,
    pod_tim_u: np.ndarray | None = None,
    pod_tim_v: np.ndarray | None = None,
    pod_tim_p: np.ndarray | None = None,
) -> Tuple[Any, Any]:
    """Create TensorFlow datasets that yield `(lr, hr)` batches in NCHW format.

    This TF-only SR version assumes your H5 already stores LR/HR pairs at the
    desired training resolutions (e.g. LR=96×96, HR=192×192 for scale=2).
    No random cropping/patch extraction is performed.
    """
    from .env import tf

    # Optionally attach POD coefficients as a "sidecar" for training/validation.
    # Supported shapes:
    #   - (N_train, k)  -> attach only to training batches
    #   - (N_total, k)  -> attach to both training and validation (split train/test)
    n_train = int(h5.lr_train.shape[0])
    n_test = int(h5.lr_test.shape[0])
    n_total = n_train + n_test

    attach_train = attach_test = False
    tim_u_tr = tim_v_tr = tim_p_tr = None
    tim_u_te = tim_v_te = tim_p_te = None

    if pod_tim_u is not None or pod_tim_v is not None or pod_tim_p is not None:
        if pod_tim_u is None or pod_tim_v is None or pod_tim_p is None:
            raise ValueError(
                "If any POD tim coefficients are provided, you must provide all of: pod_tim_u, pod_tim_v, pod_tim_p"
            )

        nu = int(pod_tim_u.shape[0])
        nv = int(pod_tim_v.shape[0])
        np_ = int(pod_tim_p.shape[0])
        if not (nu == nv == np_):
            raise ValueError(
                f"POD tim arrays must have same first dim; got tim_u={nu}, tim_v={nv}, tim_p={np_}"
            )

        if nu == n_train:
            # Train-only POD coeff stream.
            attach_train = True
            tim_u_tr, tim_v_tr, tim_p_tr = pod_tim_u, pod_tim_v, pod_tim_p
        elif nu == n_total:
            # Full POD coeff stream covering train+test in the SAME ordering as the H5 split.
            attach_train = True
            attach_test = True
            tim_u_tr, tim_v_tr, tim_p_tr = pod_tim_u[:n_train], pod_tim_v[:n_train], pod_tim_p[:n_train]
            tim_u_te, tim_v_te, tim_p_te = pod_tim_u[n_train:], pod_tim_v[n_train:], pod_tim_p[n_train:]
        else:
            raise ValueError(
                f"POD tim first dim must match N_train ({n_train}) or N_total ({n_total}); got {nu}"
            )

    if attach_train:
        train_ds = tf.data.Dataset.from_tensor_slices((h5.lr_train, h5.hr_train, tim_u_tr, tim_v_tr, tim_p_tr))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((h5.lr_train, h5.hr_train))

    if attach_test:
        test_ds = tf.data.Dataset.from_tensor_slices((h5.lr_test, h5.hr_test, tim_u_te, tim_v_te, tim_p_te))
    else:
        test_ds = tf.data.Dataset.from_tensor_slices((h5.lr_test, h5.hr_test))

    if shuffle:
        # conservative buffer; avoids huge memory spikes
        buf = min(len(h5.lr_train), batch_size * 32)
        train_ds = train_ds.shuffle(buf, reshuffle_each_iteration=True)

    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()
    return train_ds, test_ds
