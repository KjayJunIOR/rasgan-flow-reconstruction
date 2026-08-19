from __future__ import annotations

from ..env import tf

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ..models.generator import SRGAN_g
from ..models.transformer_generator import TransformerSR_g
from ..models.composite_generator import CompositeSR_g
from ..models.discriminator import CondPatchD
from ..config import (
    d_in_channels,
    EDGE_MODE, EDGE_INIT, EDGE_MIN, EDGE_DECAY_EPOCHS,
    D_CLF_TARGET_LOW, D_CLF_TARGET_HIGH,
    ADA_NOISE_MAX,
    COND_DROP_P_INIT, COND_DROP_P_MIN, COND_DROP_P_MAX, COND_DROP_P,
    EDGE_TANH_GAIN, D_SOLO_CHAN_P,
)

CKPT_FORMAT_VERSION = 1

def _save_checkpoint(dirpath, stage, epoch, G, D, ema, input_shape, *, g_lr=None, d_lr=None,
    state=None,      # <- pass steerer + net state here (see call sites below)
    extra=None       # <- kept for backward-compat
):
    os.makedirs(dirpath, exist_ok=True)
    d_in_ch = d_in_channels(use_coords=bool(getattr(G, "use_coords", False)))

    # Detect whether EMA weights are available *before* writing meta.json so
    # inference scripts can reliably pick the right family.
    had_shadow = False
    try:
        had_shadow = bool(ema.ready())
    except Exception:
        had_shadow = False
    # --- Build meta.json ------------------------------------------------------
    meta = {
        "format_version": CKPT_FORMAT_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "epoch": int(epoch),
        "arch": {
            "input_shape": list(input_shape),
            "g_sr_scale": int(getattr(G, "sr_scale", 1)),
            "g_arch": str(getattr(G, "g_arch", "rrdb")),
            "g_upsample": str(getattr(G, "upsample_mode", "resizeconv")),
            "g_out_ch": int(getattr(G, "out_ch", 3)),
            # Optional transformer hyperparams (present only if the generator exposes them)
            "g_patch": int(getattr(G, "patch_size", getattr(G, "g_patch", 0)) or 0),
            "g_embed_dim": int(getattr(G, "embed_dim", getattr(G, "g_embed_dim", 0)) or 0),
            "g_depth": int(getattr(G, "depth", getattr(G, "g_depth", 0)) or 0),
            "g_heads": int(getattr(G, "num_heads", getattr(G, "g_heads", 0)) or 0),
            "g_mlp_ratio": float(getattr(G, "mlp_ratio", getattr(G, "g_mlp_ratio", 4.0)) or 4.0),
            "g_dropout": float(getattr(G, "dropout", getattr(G, "g_dropout", 0.0)) or 0.0),
            "g_use_film": bool(getattr(G, "use_film", getattr(G, "g_use_film", False))),
            "g_coeff_dim": int(getattr(G, "coeff_dim", getattr(G, "g_coeff_dim", 0)) or 0),
            "d_in_channels": int(d_in_ch),
        },
    }

    # Training state (optim LRs + anything else we want to restore)
    train_state = {}
    if g_lr is not None:
        try: train_state["g_lr"] = float(g_lr)
        except Exception: pass
    if d_lr is not None:
        try: train_state["d_lr"] = float(d_lr)
        except Exception: pass
    if state:
        # Ensure pure-Python scalars so json.dump never chokes on numpy types.
        def _py(v):
            try:
                if isinstance(v, (np.generic,)):
                    return v.item()
            except Exception:
                pass
            return v
        train_state.update({k: _py(v) if not isinstance(v, dict)
                           else {kk: _py(vv) for kk, vv in v.items()}
                           for k, v in state.items()})
    if train_state:
        meta["train_state"] = train_state

    if extra:
        meta["extra"] = extra
    # If provided, surface the best family (ema/raw) at the top level for easy consumers
    try:
        v = extra.get("val", {}) if isinstance(extra, dict) else {}
        chosen = str(v.get("chosen", "")).strip().lower()
        if chosen in ("ema","raw"):
            meta["best_family"] = chosen
    except Exception:
        pass

    # If the run didn't explicitly tag a best family, default to EMA when available.
    # This is important because many training runs don't write extra.val.chosen.
    if "best_family" not in meta:
        meta["best_family"] = "ema" if had_shadow else "raw"

    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Weights --------------------------------------------------------------
    G.save_weights(os.path.join(dirpath, "generator_raw.weights.h5"))

    if had_shadow:
        ema.apply_shadow()
        G.save_weights(os.path.join(dirpath, "generator_ema.weights.h5"))
        ema.restore()

    if D is not None:
        try:
            D.save_weights(os.path.join(dirpath, "discriminator.weights.h5"))
        except Exception as e:
            print(f"[ckpt] Warn: failed to save D: {e}")

    print(f"[*] Checkpoint saved → {dirpath}")

def _load_checkpoint(path: str) -> dict:
    """
    Load and normalize a checkpoint's meta.json.

    Accepts either a checkpoint DIR (containing meta.json) or a direct path to meta.json.
    Always returns a dict with keys: stage, epoch, arch, train_state, extra.
    Back-compat safe for older checkpoints that don't have train_state.
    """
    # Resolve to meta.json
    meta_path = path
    if os.path.isdir(path):
        meta_path = os.path.join(path, "meta.json")

    # Defaults for old/missing metas
    default_meta = {
        "format_version": 1,
        "stage": "adv",
        "epoch": 0,
        "arch": {},
        "train_state": {},
        "extra": {},
    }

    if not os.path.exists(meta_path):
        print(f"[ckpt] warn: meta.json not found at {meta_path}; using defaults")
        # Best-effort: infer epoch from folder name like "...-e0040"
        try:
            base = os.path.basename(path if os.path.isdir(path) else os.path.dirname(path))
            m = re.search(r"e(\d{1,6})", base)
            if m:
                default_meta["epoch"] = int(m.group(1))
        except Exception:
            pass
        return default_meta

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[ckpt] warn: failed to read meta.json ({e}); using defaults")
        return default_meta

    # -------- Normalize / back-compat --------
    # Old runs may have "state" instead of "train_state"
    if "train_state" not in meta and "state" in meta:
        meta["train_state"] = meta.pop("state")

    # Ensure required keys
    meta.setdefault("format_version", default_meta["format_version"])
    meta.setdefault("stage",          default_meta["stage"])
    meta.setdefault("epoch",          default_meta["epoch"])
    meta.setdefault("arch",           {})
    meta.setdefault("train_state",    {})
    meta.setdefault("extra",          {})

    # If epoch still missing/zero, try to infer from dir name
    try:
        base = os.path.basename(path if os.path.isdir(path) else os.path.dirname(path))
        m = re.search(r"e(\d{1,6})", base)
        if m and int(meta.get("epoch", 0)) == 0:
            meta["epoch"] = int(m.group(1))
    except Exception:
        pass

    # -------- Type normalization for train_state --------
    ts = meta["train_state"]

    # LRs
    for k in ("g_lr", "d_lr"):
        if k in ts:
            try: ts[k] = float(ts[k])
            except Exception: ts.pop(k, None)

    # Steerer fields
    s = ts.get("steerer", {})
    if isinstance(s, dict):
        for k in ("d_min", "d_max"):
            if k in s:
                try: s[k] = float(s[k])
                except Exception: s.pop(k, None)
        if "hold_d_lr" in s:
            s["hold_d_lr"] = bool(s["hold_d_lr"])
        ts["steerer"] = s

    # ADA bits
    a = ts.get("ada", {})
    if isinstance(a, dict):
        for k in ("cond_drop_p", "dyn_noise"):
            if k in a:
                try: a[k] = float(a[k])
                except Exception: a.pop(k, None)
        ts["ada"] = a

    # D regularization
    dr = ts.get("d_reg", {})
    if isinstance(dr, dict):
        if "r1_gamma" in dr:
            try: dr["r1_gamma"] = float(dr["r1_gamma"])
            except Exception: dr.pop("r1_gamma", None)
        if "reg_every" in dr:
            try: dr["reg_every"] = int(dr["reg_every"])
            except Exception: dr.pop("reg_every", None)
        ts["d_reg"] = dr

    # Loss weights
    lw = ts.get("loss_weights", {})
    if isinstance(lw, dict):
        for k in ("w_spec", "w_espec", "w_res", "w_low"):
            if k in lw:
                try: lw[k] = float(lw[k])
                except Exception: lw.pop(k, None)
        ts["loss_weights"] = lw

    # Penalties (decorrelation + covariance)
    pen = ts.get("penalties", {})
    if isinstance(pen, dict):
        if "decorr_w" in pen:
            try: pen["decorr_w"] = float(pen["decorr_w"])
            except Exception: pen.pop("decorr_w", None)
        if "cov_w" in pen:
            try: pen["cov_w"] = float(pen["cov_w"])
            except Exception: pen.pop("cov_w", None)
        if "cov_on_grads" in pen:
            pen["cov_on_grads"] = bool(pen["cov_on_grads"])
        ts["penalties"] = pen

    # Stability (edge scale)
    stab = ts.get("stability", {})
    if isinstance(stab, dict):
        if "edge_scale" in stab:
            try: stab["edge_scale"] = float(stab["edge_scale"])
            except Exception: stab.pop("edge_scale", None)
        ts["stability"] = stab

    # Normalize top-level best family hint (if present)
    bf = meta.get("best_family", None)
    if isinstance(bf, str):
        bf = bf.strip().lower()
        if bf in ("ema","raw"):
            meta["best_family"] = bf
        else:
            meta.pop("best_family", None)

    # n_critic
    if "n_critic" in ts:
        try: ts["n_critic"] = int(ts["n_critic"])
        except Exception: ts.pop("n_critic", None)

    # adv weights
    adv = ts.get("adv", {})
    if isinstance(adv, dict):
        for k in ("eff", "sched"):
            if k in adv:
                try: adv[k] = float(adv[k])
                except Exception: adv.pop(k, None)
        ts["adv"] = adv

    return meta

def _set_lr(opt, lr):
    if lr is None:
        return
    try:
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = float(lr)
        elif hasattr(opt, "lr"):
            opt.lr = float(lr)
    except Exception:
        pass

def _restore_train_state(meta, g_opt, d_opt, steerer, net_g, net_d):
    """Best-effort restore of training dynamics from meta['train_state']."""
    ts = meta.get("train_state", {}) if isinstance(meta, dict) else {}

    # LRs
    g_lr = ts.get("g_lr"); d_lr = ts.get("d_lr")
    _set_lr(g_opt, g_lr); _set_lr(d_opt, d_lr)
    try:
        if g_lr is not None: steerer.g_lr = float(g_lr)
        if d_lr is not None: steerer.d_lr = float(d_lr)
    except Exception:
        pass

    # Steerer band/holds & factors (if present)
    s = ts.get("steerer", {})
    try:
        if "d_min" in s: steerer.d_min = float(s["d_min"])
        if "d_max" in s: steerer.d_max = float(s["d_max"])
        if "hold_d_lr" in s: steerer.hold_d_lr(bool(s["hold_d_lr"]))
        # Optional knobs you might be saving:
        if "factor_up" in s:   steerer.factor_up   = float(s["factor_up"])
        if "factor_down" in s: steerer.factor_down = float(s["factor_down"])
        if "allow_g_lr_up" in s:
            # support both attr or setter
            if hasattr(steerer, "allow_g_lr_up"):
                setattr(steerer, "allow_g_lr_up", bool(s["allow_g_lr_up"]))
            elif hasattr(steerer, "set_g_lr_up_enabled"):
                steerer.set_g_lr_up_enabled(bool(s["allow_g_lr_up"]))
    except Exception:
        pass

    # Full steerer state (histories/cooldowns/EMAs). Without this, resumes restart
    # the steerer warmups and periodic adjustments, which can lead to repeatable
    # instabilities after a fixed number of epochs.
    try:
        st_full = ts.get("steerer_state", None)
        if st_full is not None and hasattr(steerer, "load_state_dict"):
            steerer.load_state_dict(st_full)
    except Exception:
        pass

    # ADA bits (cond drop + dyn inst noise)
    ada = ts.get("ada", {})
    try:
        if "cond_drop_p" in ada:
            # global var in your train() scope
            globals()["cond_drop_p_state"] = float(ada["cond_drop_p"])
        if "dyn_noise" in ada:
            globals()["dyn_noise"] = float(ada["dyn_noise"])
    except Exception:
        pass

    # D regularization
    dreg = ts.get("d_reg", {})
    try:
        if "r1_gamma" in dreg:
            net_d.set_r1_gamma(float(dreg["r1_gamma"]))
        if "reg_every" in dreg:
            # IMPORTANT: don't overwrite a tf.Variable with a Python int.
            # If reg_every becomes a Python constant, any tf.function that
            # captured it will NOT see later changes (and you can get
            # resume-only, repeatable instabilities).
            re = int(dreg["reg_every"])
            if hasattr(getattr(net_d, "reg_every", None), "assign"):
                net_d.reg_every.assign(re)
            else:
                net_d.reg_every = re
    except Exception:
        pass

    # Loss weights
    lw = ts.get("loss_weights", {})
    # Only override what exists in the checkpoint; otherwise keep current values.
    try:
        if isinstance(lw, dict) and len(lw) > 0:
            def _f(x, default=0.0):
                try:
                    if hasattr(x, "numpy"):
                        return float(x.numpy())
                    return float(x)
                except Exception:
                    return float(default)

            net_g.set_loss_weights(
                w_spec = float(lw.get("w_spec",  _f(getattr(net_g, "w_spec", 0.0)))),
                w_espec= float(lw.get("w_espec", _f(getattr(net_g, "w_espec",0.0)))),
                w_res  = float(lw.get("w_res",   _f(getattr(net_g, "w_res",  0.0)))),
                w_low  = float(lw.get("w_low",   _f(getattr(net_g, "w_low",  0.0)))),
            )
    except Exception:
        pass

    # NEW: penalties / decorrelation knobs
    pen = ts.get("penalties", {})
    try:
        if "decorr_w" in pen and hasattr(net_g, "set_decorr_weight"):
            net_g.set_decorr_weight(float(pen["decorr_w"]))
        if ("cov_w" in pen) and hasattr(net_g, "set_cov_weight"):
            # respect stored cov_on_grads if provided; otherwise leave current flag unchanged
            cog = pen.get("cov_on_grads", getattr(net_g, "cov_on_grads", True))
            net_g.set_cov_weight(float(pen["cov_w"]), bool(cog))
        elif "cov_on_grads" in pen and hasattr(net_g, "set_cov_weight"):
            # toggle only the flag if weight missing
            net_g.set_cov_weight(net_g.get_cov_weight(), bool(pen["cov_on_grads"]))
    except Exception:
        pass

    # Adv weight (effective vs scheduled)
    adv = ts.get("adv", {})
    try:
        if "eff" in adv:
            net_g.set_adv_weight(float(adv["eff"]))     # what was really used
        # If your steerer keeps a planned/scheduled field, reflect it:
        if "sched" in adv and hasattr(steerer, "scheduled_adv_w"):
            steerer.scheduled_adv_w = float(adv["sched"])
    except Exception:
        pass

    # Stability knobs (resume-friendly locals)
    stab = ts.get("stability", {})
    try:
        if "edge_scale" in stab:
            globals()["edge_scale"] = float(stab["edge_scale"])
    except Exception:
        pass

    # n_critic (optional: stash to apply on first epoch)
    try:
        ncrit = ts.get("n_critic", None)
        if ncrit is not None:
            globals()["_ncritic_resume"] = int(ncrit)
    except Exception:
        pass

def _init_G_from_ckpt(meta, batch_size):
    H, W = meta["arch"]["input_shape"][2], meta["arch"]["input_shape"][3]
    sr_scale = int(meta.get('arch', {}).get('g_sr_scale', 1))
    g_arch = str(meta.get('arch', {}).get('g_arch', 'rrdb')).strip().lower()
    if g_arch in ('transformer', 'vit', 'trans'):
        arch = meta.get('arch', {})
        G = TransformerSR_g(
            sr_scale=sr_scale,
            in_ch=3,
            out_ch=int(arch.get('g_out_ch') or 3),
            patch_size=int(arch.get('g_patch') or 4),
            embed_dim=int(arch.get('g_embed_dim') or 192),
            depth=int(arch.get('g_depth') or 8),
            num_heads=int(arch.get('g_heads') or 6),
            mlp_ratio=float(arch.get('g_mlp_ratio') or 4.0),
            dropout=float(arch.get('g_dropout') or 0.1),
            coeff_dim=int(arch.get('g_coeff_dim') or 12),
            use_film=bool(arch.get('g_use_film', True)),
        )
    elif g_arch in ('composite', 'comp'):
        arch = meta.get('arch', {})
        G = CompositeSR_g(
            sr_scale=sr_scale,
            in_ch=3,
            out_ch=int(arch.get('g_out_ch') or 3),
            patch_size=int(arch.get('g_patch') or 4),
            embed_dim=int(arch.get('g_embed_dim') or 192),
            depth=int(arch.get('g_depth') or 8),
            num_heads=int(arch.get('g_heads') or 6),
            mlp_ratio=float(arch.get('g_mlp_ratio') or 4.0),
            dropout=float(arch.get('g_dropout') or 0.1),
            coeff_dim=int(arch.get('g_coeff_dim') or 12),
            use_film=bool(arch.get('g_use_film', True)),
        )
    else:
        arch = meta.get('arch', {})
        G = SRGAN_g(sr_scale=sr_scale, upsample_mode=str(arch.get('g_upsample') or 'resizeconv'))

    G.init_build(tf.zeros((batch_size, 3, H, W), dtype=tf.float32))
    return G

def _init_D_from_ckpt(meta, batch_size):
    H, W = meta["arch"]["input_shape"][2], meta["arch"]["input_shape"][3]
    scale = int(meta.get("arch", {}).get("g_sr_scale", 1))
    d_in = int(meta.get("arch", {}).get("d_in_channels", 12))
    D = CondPatchD()
    D.init_build(tf.zeros((batch_size, d_in, H * scale, W * scale), dtype=tf.float32))
    return D
