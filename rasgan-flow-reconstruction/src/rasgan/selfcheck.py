"""Smoke tests for the modular RA-SGAN refactor.

Run after installing deps (tensorflow):

    python -m rasgan.selfcheck
"""

from __future__ import annotations


import sys
import compileall
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print(f"[selfcheck] project root: {root}")
    print("[selfcheck] compiling all python files...")
    ok = compileall.compile_dir(str(root), quiet=1)
    if not ok:
        print("[selfcheck] FAIL: syntax/compile error detected")
        return 2

    print("[selfcheck] importing tensorflow (forcing TF backend)...")
    try:
        from .env import tf  # noqa: F401
    except Exception as e:
        print("[selfcheck] FAIL: could not import tensorflow:", e)
        return 3

    print("[selfcheck] importing models...")
    from .models.generator import SRGAN_g
    from .models.discriminator import CondPatchD

    # Minimal forward pass sanity.
    print("[selfcheck] running a tiny forward pass...")
    # Your code uses NCHW tensors.
    x = tf.zeros([1, 3, 32, 32], dtype=tf.float32)
    g = SRGAN_g()
    g.init_build(x)
    y = g(x)
    assert y.shape[0] == 1, f"Unexpected batch: {y.shape}"

    # Discriminator receives candidate, condition, residual, and three edge maps:
    # 3 + 3 + 3 + 3 = 12 channels.
    d_in = tf.zeros([1, 12, int(y.shape[2]), int(y.shape[3])], dtype=tf.float32)
    d = CondPatchD()
    d.init_build(d_in)
    logits, feats = d(d_in, return_feats=True)
    print(
        "[selfcheck] OK. Generator output:",
        y.shape,
        " D logits:",
        getattr(logits, "shape", None),
        " feats:",
        len(feats),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
