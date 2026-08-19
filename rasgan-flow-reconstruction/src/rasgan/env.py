
"""Runtime environment setup for TensorFlow.

This project used to rely on TensorLayerX (TLX) with a TensorFlow backend.
To support newer TensorFlow versions without TLX, we keep a small TF-only
runtime configuration helper here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"   # 0=all, 1=info, 2=warning, 3=error
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf

@dataclass(frozen=True)
class RuntimeFlags:
    mixed: bool = False
    xla: bool = False
    # Backward compatibility with the original TLX-based version.
    # The TF-only port doesn't use TLX, but configs may still pass this flag.
    tlx_verbose: bool = False


def configure_runtime(flags: RuntimeFlags) -> None:
    """Apply optional runtime settings (mixed precision / XLA)."""
    # Avoid TF grabbing all GPU memory up-front (helps small GPUs).
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass
    # Rough equivalent of the old TLX verbosity switch.
    try:
        tf.get_logger().setLevel("INFO" if flags.tlx_verbose else "WARNING")
    except Exception:
        pass

    if flags.mixed:
        # Safe default: mixed_float16
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")

    if flags.xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            # XLA may be unavailable in some builds; ignore if so.
            pass
