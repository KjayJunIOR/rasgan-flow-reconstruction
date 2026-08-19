from __future__ import annotations

"""Model factory helpers.

This project historically hard-coded the RRDB-style generator. As we add
alternative generators (e.g. a Transformer-based SR model), centralizing
construction here keeps training/checkpoint/inference paths consistent.
"""

from typing import Tuple

from ..config import TrainConfig

from .generator import SRGAN_g
from .transformer_generator import TransformerSR_g
from .composite_generator import CompositeSR_g

def build_generator(cfg: TrainConfig, *, sr_scale: int, lr_shape: Tuple[int, int, int]):
    """Instantiate the configured generator.

    Args:
      cfg: TrainConfig
      sr_scale: inferred scale from data shapes (1 or 2)
      lr_shape: (C,H,W) of LR tensors
    """
    g_arch = str(getattr(cfg, "g_arch", "rrdb")).strip().lower()
    c_lr, h_lr, w_lr = (int(lr_shape[0]), int(lr_shape[1]), int(lr_shape[2]))

    if g_arch in ("rrdb", "srgan", "baseline"):
        return SRGAN_g(sr_scale=int(sr_scale), upsample_mode=str(getattr(cfg, "g_upsample", "resizeconv")))

    if g_arch in ("transformer", "vit", "trans"):
        # POD coeff conditioning dimension: concatenate (tim_u, tim_v, tim_p)
        _cd = getattr(cfg, "g_coeff_dim", None)
        coeff_dim = int(_cd) if (_cd is not None) else 3 * int(getattr(cfg, "pod_k", 4))
        return TransformerSR_g(
            sr_scale=int(sr_scale),
            in_ch=c_lr,
            out_ch=int(getattr(cfg, "g_out_ch", 3)),
            patch_size=int(getattr(cfg, "g_patch", 4)),
            embed_dim=int(getattr(cfg, "g_embed_dim", 192)),
            depth=int(getattr(cfg, "g_depth", 8)),
            num_heads=int(getattr(cfg, "g_heads", 6)),
            mlp_ratio=float(getattr(cfg, "g_mlp_ratio", 4.0)),
            dropout=float(getattr(cfg, "g_dropout", 0.1)),
            coeff_dim=coeff_dim,
            use_film=bool(getattr(cfg, "g_use_film", True)),
        )

    if g_arch in ("composite", "comp"):
        # POD coeff conditioning dimension: concatenate (tim_u, tim_v, tim_p)
        _cd = getattr(cfg, "g_coeff_dim", None)
        coeff_dim = int(_cd) if (_cd is not None) else 3 * int(getattr(cfg, "pod_k", 4))
        return CompositeSR_g(
            sr_scale=int(sr_scale),
            in_ch=c_lr,
            out_ch=int(getattr(cfg, "g_out_ch", 3)),
            patch_size=int(getattr(cfg, "g_patch", 4)),
            embed_dim=int(getattr(cfg, "g_embed_dim", 192)),
            depth=int(getattr(cfg, "g_depth", 8)),
            num_heads=int(getattr(cfg, "g_heads", 6)),
            mlp_ratio=float(getattr(cfg, "g_mlp_ratio", 4.0)),
            dropout=float(getattr(cfg, "g_dropout", 0.1)),
            coeff_dim=coeff_dim,
            use_film=bool(getattr(cfg, "g_use_film", True)),
        )

    raise ValueError(f"Unknown generator architecture g_arch={g_arch!r}. Expected 'rrdb', 'transformer', or 'composite'.")
