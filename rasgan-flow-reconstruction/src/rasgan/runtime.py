from __future__ import annotations

from .env import tf

from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class RuntimeStats:
    # Numpy broadcastable (1,C,1,1)
    hr_mean_np: np.ndarray
    hr_std_np: np.ndarray
    lr_mean_np: np.ndarray
    lr_std_np: np.ndarray

    # TensorFlow tensors broadcastable (1,C,1,1)
    hr_mean: tf.Tensor
    hr_std: tf.Tensor
    lr_mean: tf.Tensor
    lr_std: tf.Tensor
    hr_inv: tf.Tensor


def build_stats(means_hr: np.ndarray, stds_hr: np.ndarray, means_lr: np.ndarray, stds_lr: np.ndarray) -> RuntimeStats:
    c = int(means_hr.shape[0])
    hr_mean_np = np.asarray(means_hr, dtype=np.float32).reshape(1, c, 1, 1)
    hr_std_np = np.asarray(stds_hr, dtype=np.float32).reshape(1, c, 1, 1)
    lr_mean_np = np.asarray(means_lr, dtype=np.float32).reshape(1, c, 1, 1)
    lr_std_np = np.asarray(stds_lr, dtype=np.float32).reshape(1, c, 1, 1)

    # avoid tiny stds
    hr_std_np = np.maximum(hr_std_np, 1e-6).astype(np.float32)
    lr_std_np = np.maximum(lr_std_np, 1e-6).astype(np.float32)

    hr_mean = tf.convert_to_tensor(hr_mean_np, dtype=tf.float32)
    hr_std = tf.convert_to_tensor(hr_std_np, dtype=tf.float32)
    lr_mean = tf.convert_to_tensor(lr_mean_np, dtype=tf.float32)
    lr_std = tf.convert_to_tensor(lr_std_np, dtype=tf.float32)
    hr_inv = tf.convert_to_tensor(1.0 / hr_std_np, dtype=tf.float32)

    return RuntimeStats(
        hr_mean_np=hr_mean_np,
        hr_std_np=hr_std_np,
        lr_mean_np=lr_mean_np,
        lr_std_np=lr_std_np,
        hr_mean=hr_mean,
        hr_std=hr_std,
        lr_mean=lr_mean,
        lr_std=lr_std,
        hr_inv=hr_inv,
    )