from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np

from rasgan.env import tf
from rasgan.data import load_h5
from rasgan.pod_sidecar import load_pod_sidecar
from rasgan.utils.checkpoint import _init_G_from_ckpt


def _best_weight_file(ckpt_dir: str) -> Optional[str]:
    """Choose generator weights for a checkpoint directory.

    Priority:
      1) If meta.json tags best_family (or extra.val.chosen), use that family if present.
      2) Otherwise, prefer EMA if it exists, else RAW.

    Returns the path to a .weights.h5 file, or None if none found.
    """
    meta_path = os.path.join(ckpt_dir, "meta.json")
    chosen: str = ""
    stage: str = ""
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            stage = str(m.get("stage", "")).strip().lower()
            chosen = (
                str(m.get("extra", {}).get("val", {}).get("chosen", "")).strip().lower()
                or str(m.get("best_family", "")).strip().lower()
            )
        except Exception:
            chosen = ""
            stage = ""

    if chosen in ("ema", "raw"):
        cand = os.path.join(ckpt_dir, f"generator_{chosen}.weights.h5")
        if os.path.exists(cand):
            return cand

    # fallback preference:
    # - For INIT stage checkpoints, prefer RAW because EMA may be un-updated in some older runs.
    # - For ADV stage checkpoints, prefer EMA (if present).
    if stage == "init":
        cand = os.path.join(ckpt_dir, "generator_raw.weights.h5")
        if os.path.exists(cand):
            return cand
        cand = os.path.join(ckpt_dir, "generator_ema.weights.h5")
        if os.path.exists(cand):
            return cand
    else:
        cand = os.path.join(ckpt_dir, "generator_ema.weights.h5")
        if os.path.exists(cand):
            return cand
        cand = os.path.join(ckpt_dir, "generator_raw.weights.h5")
        if os.path.exists(cand):
            return cand
    return None


def _resolve_weights(weights: Optional[str], ckpt_dir: str) -> Tuple[str, Optional[str]]:
    """Resolve which generator weights to load.

    Returns: (weight_path, meta_dir)
      - meta_dir is a directory containing meta.json, or None.
    """
    weight_path: Optional[str] = None
    meta_dir: Optional[str] = None

    if weights:
        if os.path.isdir(weights):
            meta_dir = weights if os.path.exists(os.path.join(weights, "meta.json")) else None
            weight_path = _best_weight_file(weights)
            if not weight_path:
                raise FileNotFoundError(f"No generator weights found in dir: {weights}")
        else:
            if not os.path.exists(weights):
                raise FileNotFoundError(f"Checkpoint file not found: {weights}")
            weight_path = weights
            maybe_meta_dir = os.path.dirname(weights)
            if os.path.exists(os.path.join(maybe_meta_dir, "meta.json")):
                meta_dir = maybe_meta_dir
    else:
        # No explicit weights path: pick newest checkpoint dir (by meta.json mtime)
        latest_dir: Optional[str] = None
        try:
            subdirs = [
                os.path.join(ckpt_dir, d)
                for d in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, d))
            ]
            subdirs = [d for d in subdirs if os.path.exists(os.path.join(d, "meta.json"))]
            if subdirs:
                latest_dir = sorted(
                    subdirs,
                    key=lambda d: os.path.getmtime(os.path.join(d, "meta.json")),
                    reverse=True,
                )[0]
        except Exception:
            latest_dir = None

        if latest_dir is not None:
            meta_dir = latest_dir
            weight_path = _best_weight_file(latest_dir)
            if not weight_path:
                raise FileNotFoundError(f"No generator weights found in dir: {latest_dir}")
            print(f"[*] Using checkpoint dir: {latest_dir}")
        else:
            # Legacy fallbacks directly inside ckpt_dir
            for name in (
                "generator_ema.weights.h5",
                "generator_raw.weights.h5",
                "g_best_ema.weights.h5",
                "g_best_raw.weights.h5",
            ):
                p = os.path.join(ckpt_dir, name)
                if os.path.exists(p):
                    weight_path = p
                    maybe_meta_dir = os.path.dirname(p)
                    if os.path.exists(os.path.join(maybe_meta_dir, "meta.json")):
                        meta_dir = maybe_meta_dir
                    break
            if not weight_path:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    assert weight_path is not None
    return weight_path, meta_dir


def _build_generator_from_meta_or_cli(lr_shape, sr_scale: int, upsample_mode: str, meta_dir: Optional[str]):
    """Prefer building the generator from checkpoint meta.json when available.

    This allows loading non-RRDB generators (e.g. transformer) without changing
    inference CLI flags.
    """
    if meta_dir and os.path.exists(os.path.join(meta_dir, "meta.json")):
        try:
            with open(os.path.join(meta_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            # meta contains full input_shape; but we still provide a batch_size=1.
            return _init_G_from_ckpt(meta, batch_size=1)
        except Exception:
            pass

    # Fallback: CLI-controlled RRDB generator
    from rasgan.models.generator import SRGAN_g
    upsample_mode = (upsample_mode or "pixelshuffle").lower()
    if upsample_mode in ("resizebilinear", "bilinear", "resize_bilinear"):
        upsample_mode = "resizeconv"
    G = SRGAN_g(sr_scale=int(sr_scale), upsample_mode=upsample_mode)
    c, h, w = lr_shape
    _ = G(tf.zeros([1, int(c), int(h), int(w)], dtype=tf.float32))
    return G

def main() -> None:
    ap = argparse.ArgumentParser(description="Inference on H5 test split -> MAT file")
    ap.add_argument("--data", required=True, help="Path to sol_data.h5")
    ap.add_argument(
        "--weights",
        default=None,
        help="Checkpoint dir (contains meta.json) or generator_*.weights.h5 file. If omitted, picks newest in --ckpt-dir.",
    )
    ap.add_argument(
        "--ckpt-dir",
        default="checkpoints",
        help="Checkpoint root directory (used when --weights is not provided)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output .mat file path. Default: <ckpt-dir>/rasgansol.mat or next to --weights if file/dir provided.",
    )
    ap.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Optionally limit the number of test samples to run (useful for quick smoke tests).",
    )
    ap.add_argument("--sr-scale", type=int, default=2, help="Super-resolution scale factor (default: 2)")
    ap.add_argument("--upsample-mode", type=str, default="pixelshuffle",
                    choices=["pixelshuffle", "resizeconv", "resizebilinear"],
                    help="Generator upsampling path. resizebilinear is an alias for resizeconv.")
    ap.add_argument(
        "--pod-mat",
        default=None,
        help="Optional: path to POD sidecar .mat (v7.3 HDF5) containing TimCoeU/V/P. If provided, test POD coefficients are fed to the generator as pod_coeffs.",
    )
    ap.add_argument(
        "--pod-k",
        type=int,
        default=5,
        help="Number of POD modes (K) to read from TimCoe arrays (default: 5). Coeff vector passed to generator is [TimCoeU,TimCoeV,TimCoeP] concatenated.",
    )
    args = ap.parse_args()

    try:
        import scipy.io as sio
    except Exception as e:
        raise RuntimeError(
            "scipy is required for saving .mat files. Install it via `pip install scipy`."
        ) from e

    # Load test split
    h5 = load_h5(args.data)
    lr_test = h5.lr_test
    hr_test = h5.hr_test
    means = h5.means_hr
    stds = h5.stds_hr
    # Optional POD coeffs (aligned to H5 shuffle + split)
    pod_u_test = pod_v_test = pod_p_test = None
    if args.pod_mat:
        side = load_pod_sidecar(args.pod_mat, k=int(args.pod_k))
        if side.tim_u is None or side.tim_v is None or side.tim_p is None:
            raise ValueError("--pod-mat must contain TimCoeU/TimCoeV/TimCoeP arrays for inference")
        tim_u_all = np.asarray(side.tim_u, dtype=np.float32)
        tim_v_all = np.asarray(side.tim_v, dtype=np.float32)
        tim_p_all = np.asarray(side.tim_p, dtype=np.float32)

        n_train = int(h5.lr_train.shape[0])
        n_test  = int(h5.lr_test.shape[0])
        n_total = n_train + n_test
        if int(tim_u_all.shape[0]) != n_total or int(tim_v_all.shape[0]) != n_total or int(tim_p_all.shape[0]) != n_total:
            raise ValueError(
                f"POD coeff rows must match total snapshots ({n_total} = train {n_train} + test {n_test}). "
                "Your pod sidecar should include both train+test coefficients."
            )

        if h5.perm is not None:
            perm = np.asarray(h5.perm, dtype=np.int64).reshape(-1)
            if int(perm.shape[0]) != n_total:
                raise ValueError(f"H5 perm length {perm.shape[0]} must equal total snapshots {n_total}")
            tim_u_all = tim_u_all[perm]
            tim_v_all = tim_v_all[perm]
            tim_p_all = tim_p_all[perm]
        # After shuffle, H5 is split as: [0:n_train] train, [n_train:n_total] test
        pod_u_test = tim_u_all[n_train:n_total]
        pod_v_test = tim_v_all[n_train:n_total]
        pod_p_test = tim_p_all[n_train:n_total]
        print(f"[*] POD coeffs loaded: test shapes u={pod_u_test.shape} v={pod_v_test.shape} p={pod_p_test.shape}")

    print(f"[*] mean: {means}")
    print(f"[*] std: {stds}")

    # Resolve weights
    weight_path, meta_dir = _resolve_weights(args.weights, args.ckpt_dir)

    # Default output path
    if args.out is None:
        if args.weights:
            base_dir = args.weights if os.path.isdir(args.weights) else os.path.dirname(args.weights)
        else:
            base_dir = meta_dir or args.ckpt_dir
        args.out = os.path.join(base_dir, "compsol.mat")

    # Build + load generator
    lr_shape = tuple(lr_test.shape[1:])  # (C,H,W)
    G = _build_generator_from_meta_or_cli(lr_shape, args.sr_scale, args.upsample_mode, meta_dir)

    print(f"[*] Loading generator weights: {weight_path}")

    G.load_weights(weight_path)
    try:
        G.set_eval()
    except Exception:
        pass

    # Run inference
    N = int(lr_test.shape[0])
    if args.max_n is not None:
        N = min(N, int(args.max_n))

    gen = []
    lrs = []
    hrs = []

    for i in range(N):
        lr_s = lr_test[i]  # (C,H,W)
        hr_s = hr_test[i]

        inp = tf.convert_to_tensor(lr_s[None, ...], dtype=tf.float32)
        if pod_u_test is not None:
            u = tf.convert_to_tensor(pod_u_test[i][None, :], dtype=tf.float32)
            v = tf.convert_to_tensor(pod_v_test[i][None, :], dtype=tf.float32)
            p = tf.convert_to_tensor(pod_p_test[i][None, :], dtype=tf.float32)
            try:
                out = G(inp, pod_coeffs=(u, v, p), training=False)
            except TypeError:
                out = G(inp, training=False)
        else:
            out = G(inp, training=False)
        out_np = out.numpy()
        out_np = np.squeeze(out_np, axis=0)

        gen.append(out_np)
        lrs.append(lr_s)
        hrs.append(hr_s)
        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"Gen {i+1}/{N}", end="\r")

    gen = np.asarray(gen)
    lrs = np.asarray(lrs)
    hrs = np.asarray(hrs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sio.savemat(
        args.out,
        {
            "valid_gen": gen,
            "valid_lr": lrs,
            "valid_hr": hrs,
            **({
                "valid_pod_u": pod_u_test[:N],
                "valid_pod_v": pod_v_test[:N],
                "valid_pod_p": pod_p_test[:N],
            } if pod_u_test is not None else {}),
            "weights_path": np.array([weight_path], dtype=object),
        },
    )
    print(f"\n[*] Saved {os.path.basename(args.out)} → {args.out}")


if __name__ == "__main__":
    main()
