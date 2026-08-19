from __future__ import annotations

from .env import tf

from typing import Iterable, List, Optional, Sequence, Union
import numpy as np


class EMA:
    """Exponential Moving Average over TensorFlow variables.

    Works with either a module (with .trainable_weights) or a list/sequence of variables.
    Stores shadow weights as numpy arrays and can swap them into the live variables.
    """

    def __init__(self, module_or_vars: Union[object, Sequence], decay: float = 0.999, clip_abs: Optional[float] = None):
        self.decay = float(decay)
        self.clip_abs = clip_abs  # e.g., 1e6 or None

        if hasattr(module_or_vars, "trainable_weights"):
            self.vars = list(getattr(module_or_vars, "trainable_weights"))
        else:
            self.vars = list(module_or_vars)

        self.shadow: List[np.ndarray] = []
        self.backup: Optional[List[np.ndarray]] = None
        # IMPORTANT:
        # The original reference script treats EMA as "not ready" until at least
        # one `update()` has happened (after a successful G step). That prevents
        # saving/using stale EMA weights during init pretrain.
        self.steps: int = 0

        # Initialize shadow from current live vars, but keep `steps=0` so
        # `ready()` stays False until an actual update occurs.
        self.register()

    def _clean(self, arr: np.ndarray) -> np.ndarray:
        """Sanitize an array in-place when possible.

        Important for long runs:
          - Using `nan_to_num(copy=False)` and `np.clip(..., out=...)` avoids
            large temporary allocations.
          - Keeps dtype unchanged (caller may cast explicitly).
        """
        arr = np.asarray(arr)
        # Replace NaN/Inf defensively (in-place)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_abs is not None:
            np.clip(arr, -self.clip_abs, self.clip_abs, out=arr)
        return arr

    def register(self) -> None:
        """Initialize shadow weights from current live vars."""
        self.shadow = [self._clean(v.numpy().copy()) for v in self.vars]
        # Mirror rasgansol3.2.2.py: register() does NOT make EMA "ready".
        # Only update() increments steps.
        self.steps = 0

    def update(self) -> None:
        """Update shadow weights from current live vars."""
        if not self.shadow:
            self.register()
            return
        # Keep computations in float32 to avoid float64 promotion (python float
        # ufuncs default to float64), and update shadows IN PLACE to avoid
        # unbounded CPU heap churn on long runs.
        decay = np.float32(self.decay)
        one_minus = np.float32(1.0) - decay
        for i, v in enumerate(self.vars):
            w = v.numpy()
            # Ensure float32 math
            if w.dtype != np.float32:
                w = w.astype(np.float32, copy=False)
            self._clean(w)

            s = self.shadow[i]
            if s.dtype != np.float32:
                s = s.astype(np.float32, copy=False)
                self.shadow[i] = s

            # s = decay*s + (1-decay)*w  (in-place)
            s *= decay
            s += one_minus * w
            self._clean(s)
        self.steps += 1

    def apply_shadow(self) -> None:
        """Swap live weights to EMA shadow (stores backup to restore later)."""
        self.backup = []
        for v, s in zip(self.vars, self.shadow):
            self.backup.append(v.numpy().copy())
            v.assign(tf.convert_to_tensor(self._clean(s), dtype=v.dtype))

    def restore(self) -> None:
        """Restore live weights from the last backup created by apply_shadow()."""
        if self.backup is None:
            return
        for v, b in zip(self.vars, self.backup):
            v.assign(tf.convert_to_tensor(self._clean(b), dtype=v.dtype))
        self.backup = None

    def ready(self) -> bool:
        return self.steps >= 1
