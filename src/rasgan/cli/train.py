from __future__ import annotations

import argparse
from dataclasses import replace

from rasgan.config import TrainConfig, LossWeights
from rasgan.training import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RASGAN for paired flow-field reconstruction.")
    p.add_argument("--data", dest="data_path", default=TrainConfig().data_path, help="Path to the paired NCHW HDF5 dataset")
    p.add_argument("--ckpt-dir", default=TrainConfig().ckpt_dir, help="Checkpoint directory")
    p.add_argument("--save-dir", default=TrainConfig().save_dir, help="Sample/output directory")
    p.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    p.add_argument("--init-epochs", type=int, default=TrainConfig().init_epochs)
    p.add_argument("--adv-epochs", type=int, default=TrainConfig().adv_epochs)
    p.add_argument("--mixed", action="store_true", help="Enable mixed precision")
    p.add_argument("--xla", action="store_true", help="Enable TF XLA JIT")
    p.add_argument("--resume-from", default=None, help="Checkpoint directory or meta.json path")
    p.add_argument("--resume-stage", default=None, choices=["init", "adv"], help="Stage to resume (optional)")
    p.add_argument("--g-lr-i", type=float, default=TrainConfig().g_lr_i, help="Initial G LR (init)")
    p.add_argument("--g-lr-adv", type=float, default=TrainConfig().g_lr_adv, help="Initial G LR (adv)")
    p.add_argument("--d-lr", type=float, default=TrainConfig().d_lr_adv, help="Initial D LR")

    # Gradient-magnitude loss reweighting (adv stage, generator only)
    p.add_argument(
        "--grad-reweight",
        action="store_true",
        help="Enable experimental per-term gradient-magnitude reweighting for G in the adversarial stage",
    )
    p.add_argument(
        "--grad-reweight-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated list of G loss-term keys to reweight (from WithLoss_G._compute_terms). "
            "If omitted, uses the config default. For memory-constrained runs, select a small subset explicitly."
        ),
    )
    p.add_argument("--grad-reweight-every", type=int, default=TrainConfig().grad_reweight_every, help="Recompute grad-norm weights every N G steps (saves compute).")
    p.add_argument("--grad-reweight-ema", type=float, default=TrainConfig().grad_reweight_ema, help="EMA decay for grad norms")
    p.add_argument("--grad-reweight-power", type=float, default=TrainConfig().grad_reweight_power, help="Exponent p in (target/norm)^p")
    p.add_argument("--grad-reweight-clip", type=float, default=TrainConfig().grad_reweight_clip, help="Clamp for multipliers (symmetrical)")
    p.add_argument("--grad-reweight-eps", type=float, default=TrainConfig().grad_reweight_eps, help="Epsilon for numerical stability")
    p.add_argument(
        "--grad-reweight-deterministic-norms",
        action="store_true",
        help="Estimate grad norms without instance noise or conditioning dropout",
    )

    # Optimizer (AdamW)
    p.add_argument("--g-weight-decay", type=float, default=TrainConfig().g_weight_decay,
                   help="Generator AdamW weight_decay (decoupled).")
    p.add_argument("--d-weight-decay", type=float, default=TrainConfig().d_weight_decay,
                   help="Discriminator AdamW weight_decay (decoupled).")

    # Super-resolution
    p.add_argument("--scale", type=int, default=TrainConfig().scale, help="Expected SR scale (1 or 2); paired array shapes are authoritative")
    p.add_argument("--g-upsample", "--g_upsample", dest="g_upsample", type=str, default=TrainConfig().g_upsample, help="Generator upsampling: resizeconv (bilinear+conv) or pixelshuffle (subpixel conv).")

    # Generator architecture
    p.add_argument("--g-arch", dest="g_arch", type=str, default=TrainConfig().g_arch,
                   choices=["rrdb", "transformer", "composite"],
                   help="Generator architecture. rrdb = SRGAN_g (RRDB). transformer = TransformerSR_g.  composite = CompositeSR_g")
    # Transformer hyperparams (only used when --g-arch=transformer)
    p.add_argument("--g-patch", type=int, default=TrainConfig().g_patch, help="Transformer patch size on LR grid")
    p.add_argument("--g-embed-dim", type=int, default=TrainConfig().g_embed_dim, help="Transformer embed dim")
    p.add_argument("--g-depth", type=int, default=TrainConfig().g_depth, help="Transformer depth")
    p.add_argument("--g-heads", type=int, default=TrainConfig().g_heads, help="Transformer attention heads")
    p.add_argument("--g-dropout", type=float, default=TrainConfig().g_dropout, help="Transformer dropout")
    p.add_argument("--g-use-film", action="store_true", default=TrainConfig().g_use_film, help="Enable FiLM conditioning")
    p.add_argument("--g-no-film", action="store_false", dest="g_use_film", help="Disable FiLM conditioning")
    p.add_argument("--g-coeff-dim", type=int, default=None, help="Override coeff dim for transformer conditioning (default: 3*pod_k)")

    # POD sidecar (optional)
    p.add_argument("--pod-mat", dest="pod_mat_path", default=TrainConfig().pod_mat_path, help="Path to POD .mat sidecar")
    p.add_argument("--pod-k", type=int, default=TrainConfig().pod_k, help="Number of POD modes to use")

    # checkpoint cadence
    p.add_argument(
        "--save-every",
        type=int,
        default=TrainConfig().save_every,
        help="Save a checkpoint every N epochs (also saves last epoch of each stage).",
    )

    # loss weights quick overrides
    p.add_argument("--pix-w", type=float, nargs=3, default=None, metavar=("P", "V", "WZ"), help="Pixel weights (3 values)")
    p.add_argument("--grad-w", type=float, default=None, help="Gradient loss weight")
    p.add_argument("--tv-w", type=float, default=None, help="TV loss weight")
    p.add_argument("--w-low", type=float, default=None, help="Low-pass (LP delta-P) weight (adv stage)")
    p.add_argument("--w-res", type=float, default=None, help="Residual/edge-aware weight (adv stage)")
    p.add_argument("--init-w-low", type=float, default=None, help="Init-stage low-pass (LP delta-P) weight")
    p.add_argument("--init-w-res", type=float, default=None, help="Init-stage residual/edge-aware weight")

    # Physics / grid params (used by physics-informed losses)
    p.add_argument("--dx", type=float, default=TrainConfig().dx)
    p.add_argument("--dy", type=float, default=TrainConfig().dy)
    p.add_argument("--nu", type=float, default=TrainConfig().nu, help="Kinematic viscosity (0 disables viscous term)")
    p.add_argument("--rho", type=float, default=TrainConfig().rho, help="Density")
    p.add_argument("--bc", default=TrainConfig().bc, choices=["replicate", "periodic"], help="Boundary mode for finite differences")
    p.add_argument("--poisson-method", default=TrainConfig().poisson_method, choices=["fft", "jacobi"], help="Poisson solver for streamfunction")
    p.add_argument("--poisson-iters", type=int, default=TrainConfig().poisson_iters, help="Jacobi iterations if poisson-method=jacobi")

    # Physics loss weights (adv stage unless prefixed with init-)
    p.add_argument("--w-vomega", type=float, default=None, help="Weight for v-omega compatibility")
    p.add_argument("--w-omcons", type=float, default=None, help="Weight for omega consistency")
    p.add_argument("--w-div", type=float, default=None, help="Weight for divergence-free penalty")
    p.add_argument("--w-mom", type=float, default=None, help="Weight for momentum residual")
    p.add_argument("--w-ppois", type=float, default=None, help="Weight for pressure Poisson residual")
    p.add_argument("--init-w-vomega", type=float, default=None, help="Init-stage v-omega weight")
    p.add_argument("--init-w-omcons", type=float, default=None, help="Init-stage omega consistency weight")
    p.add_argument("--init-w-div", type=float, default=None, help="Init-stage divergence weight")
    p.add_argument("--init-w-mom", type=float, default=None, help="Init-stage momentum residual weight")
    p.add_argument("--init-w-ppois", type=float, default=None, help="Init-stage pressure Poisson residual weight")

    # POD sidecar loss weights (adv stage unless prefixed with init-)
    p.add_argument("--w-pod-vel", type=float, default=None, help="Weight for velocity POD cycle/coeff loss")
    p.add_argument("--w-pod-p", type=float, default=None, help="Weight for pressure POD cycle/coeff loss")
    p.add_argument("--w-pod-w", type=float, default=None, help="Weight for vorticity POD consistency")
    p.add_argument("--init-w-pod-vel", type=float, default=None, help="Init-stage velocity POD weight")
    p.add_argument("--init-w-pod-p", type=float, default=None, help="Init-stage pressure POD weight")
    p.add_argument("--init-w-pod-w", type=float, default=None, help="Init-stage vorticity POD weight")
    return p


def main():
    args = build_parser().parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        ckpt_dir=args.ckpt_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        init_epochs=args.init_epochs,
        adv_epochs=args.adv_epochs,
        mixed=args.mixed,
        xla=args.xla,
        resume_from=args.resume_from,
        resume_stage=args.resume_stage,
    )
    # Initial LRs
    cfg = replace(
        cfg,
        g_lr_i=float(getattr(args, "g_lr_i", cfg.g_lr_i)),
        g_lr_adv=float(getattr(args, "g_lr_adv", cfg.g_lr_adv)),
        d_lr_adv=float(getattr(args, "d_lr", cfg.d_lr_adv))
    )

    # Gradient-magnitude reweighting knobs
    cfg = replace(
        cfg,
        grad_reweight=bool(getattr(args, "grad_reweight", False)),
        grad_reweight_every=int(getattr(args, "grad_reweight_every", cfg.grad_reweight_every)),
        grad_reweight_ema=float(getattr(args, "grad_reweight_ema", cfg.grad_reweight_ema)),
        grad_reweight_power=float(getattr(args, "grad_reweight_power", cfg.grad_reweight_power)),
        grad_reweight_clip=float(getattr(args, "grad_reweight_clip", cfg.grad_reweight_clip)),
        grad_reweight_eps=float(getattr(args, "grad_reweight_eps", cfg.grad_reweight_eps)),
        grad_reweight_deterministic_norms=bool(getattr(args, "grad_reweight_deterministic_norms", False)) or cfg.grad_reweight_deterministic_norms,
    )
    if getattr(args, "grad_reweight_keys", None):
        keys = tuple(k.strip() for k in str(args.grad_reweight_keys).split(",") if k.strip())
        if keys:
            cfg = replace(cfg, grad_reweight_keys=keys)

    # Optimizer (AdamW)
    cfg = replace(cfg, g_weight_decay=float(getattr(args, "g_weight_decay", cfg.g_weight_decay)))
    cfg = replace(cfg, d_weight_decay=float(getattr(args, "d_weight_decay", cfg.d_weight_decay)))

    # Super-resolution scale factor
    cfg = replace(cfg, scale=int(args.scale))
    cfg = replace(cfg, g_upsample=str(args.g_upsample))

    # Generator architecture
    cfg = replace(cfg, g_arch=str(getattr(args, "g_arch", cfg.g_arch)))
    cfg = replace(cfg, g_patch=int(getattr(args, "g_patch", cfg.g_patch)))
    cfg = replace(cfg, g_embed_dim=int(getattr(args, "g_embed_dim", cfg.g_embed_dim)))
    cfg = replace(cfg, g_depth=int(getattr(args, "g_depth", cfg.g_depth)))
    cfg = replace(cfg, g_heads=int(getattr(args, "g_heads", cfg.g_heads)))
    cfg = replace(cfg, g_dropout=float(getattr(args, "g_dropout", cfg.g_dropout)))
    cfg = replace(cfg, g_use_film=bool(getattr(args, "g_use_film", cfg.g_use_film)))
    if getattr(args, "g_coeff_dim", None) is not None:
        cfg = replace(cfg, g_coeff_dim=int(getattr(args, "g_coeff_dim")))

    # Grid/physics params
    cfg = replace(cfg, dx=float(args.dx), dy=float(args.dy), nu=float(args.nu), rho=float(args.rho),
                  bc=str(args.bc), poisson_method=str(args.poisson_method), poisson_iters=int(args.poisson_iters))

    # POD sidecar
    cfg = replace(cfg, pod_mat_path=getattr(args, "pod_mat_path", None), pod_k=int(getattr(args, "pod_k", cfg.pod_k)))


    # Optional overrides into adv weights
    # Optional overrides into init weights
    init_w = cfg.init_weights
    if args.init_w_vomega is not None: init_w = replace(init_w, w_vomega=float(args.init_w_vomega))
    if args.init_w_omcons is not None: init_w = replace(init_w, w_omcons=float(args.init_w_omcons))
    if args.init_w_div is not None:    init_w = replace(init_w, w_div=float(args.init_w_div))
    if args.init_w_mom is not None:    init_w = replace(init_w, w_mom=float(args.init_w_mom))
    if args.init_w_ppois is not None:  init_w = replace(init_w, w_ppois=float(args.init_w_ppois))
    if args.init_w_pod_vel is not None: init_w = replace(init_w, w_pod_vel=float(args.init_w_pod_vel))
    if args.init_w_pod_p is not None:   init_w = replace(init_w, w_pod_p=float(args.init_w_pod_p))
    if args.init_w_pod_w is not None:   init_w = replace(init_w, w_pod_w=float(args.init_w_pod_w))
    if args.init_w_low is not None:
        init_w = replace(init_w, w_low=float(args.init_w_low))
    elif args.w_low is not None:
        init_w = replace(init_w, w_low=float(args.w_low))
    if args.init_w_res is not None:
        init_w = replace(init_w, w_res=float(args.init_w_res))
    elif args.w_res is not None:
        init_w = replace(init_w, w_res=float(args.w_res))
    cfg = replace(cfg, init_weights=init_w)

    adv_w = cfg.adv_weights
    if args.pix_w is not None:
        adv_w = replace(adv_w, pix_w=tuple(float(x) for x in args.pix_w))
    if args.grad_w is not None:
        adv_w = replace(adv_w, grad_w=float(args.grad_w))
    if args.tv_w is not None:
        adv_w = replace(adv_w, tv_w=float(args.tv_w))
    if args.w_vomega is not None: adv_w = replace(adv_w, w_vomega=float(args.w_vomega))
    if args.w_omcons is not None: adv_w = replace(adv_w, w_omcons=float(args.w_omcons))
    if args.w_div is not None:    adv_w = replace(adv_w, w_div=float(args.w_div))
    if args.w_mom is not None:    adv_w = replace(adv_w, w_mom=float(args.w_mom))
    if args.w_ppois is not None:  adv_w = replace(adv_w, w_ppois=float(args.w_ppois))
    if args.w_pod_vel is not None: adv_w = replace(adv_w, w_pod_vel=float(args.w_pod_vel))
    if args.w_pod_p is not None:   adv_w = replace(adv_w, w_pod_p=float(args.w_pod_p))
    if args.w_pod_w is not None:   adv_w = replace(adv_w, w_pod_w=float(args.w_pod_w))
    if args.w_low is not None:     adv_w = replace(adv_w, w_low=float(args.w_low))
    if args.w_res is not None:     adv_w = replace(adv_w, w_res=float(args.w_res))
    cfg = replace(cfg, adv_weights=adv_w)
    # Backwards-compatible: older runs may not include this arg.
    cfg = replace(cfg, save_every=int(getattr(args, "save_every", cfg.save_every)))

    train(cfg)


if __name__ == "__main__":
    main()
