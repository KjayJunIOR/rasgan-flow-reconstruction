from __future__ import annotations

"""Generate a tiny, redistributable three-channel flow-field example.

This is an analytic smoke-test dataset, not CFD and not a reproduction of the
research data. It mimics the public data contract: a coarse/POD-like three-
channel field is paired with a higher-resolution field containing smaller-scale
jet and vortex structure.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

CHANNEL_NAMES = ("pressure_like", "crossflow_velocity_like", "vorticity_like")


def _gaussian(x, y, x0, y0, sx, sy):
    return np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))


def _field_sample(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, sample_id: int) -> np.ndarray:
    phase = rng.uniform(0.0, 2.0 * np.pi)
    strength = rng.uniform(0.85, 1.15)
    bend = rng.uniform(0.12, 0.28)
    jitter = rng.normal(0.0, 0.025)

    center = -0.15 + bend * y + 0.08 * np.sin(1.5 * y + phase) + jitter
    width = 0.15 + 0.045 * y
    plume = np.exp(-0.5 * ((x - center) / width) ** 2) * np.exp(-0.12 * y)

    # Pressure-like channel: inlet high-pressure region, wake deficit, and weak fine-scale texture.
    pressure = (
        1.02e5
        + 1.6e4 * strength * _gaussian(x, y, -0.25, 0.55, 0.38, 0.48)
        - 4.0e3 * _gaussian(x, y, 0.45 + 0.05 * np.sin(phase), 1.45, 0.65, 0.70)
        + 7.5e2 * np.sin(4.0 * x - 2.1 * y + phase) * plume
    )

    # Crossflow-normal velocity-like channel: curved jet core plus smaller shear-layer waves.
    velocity = 520.0 * strength * plume
    velocity += 48.0 * np.sin(5.2 * y + phase) * np.exp(-((x - center) / (1.8 * width)) ** 2)
    velocity -= 35.0 * _gaussian(x, y, -0.65, 1.2, 0.32, 0.75)

    # Vorticity-like channel: alternating compact cores laid along the curved jet trajectory.
    omega = np.zeros_like(x)
    for k, y0 in enumerate(np.linspace(0.55, 3.45, 7)):
        x0 = -0.15 + bend * y0 + 0.08 * np.sin(1.5 * y0 + phase)
        sign = -1.0 if k % 2 else 1.0
        amp = sign * strength * (3000.0 - 180.0 * k)
        omega += amp * _gaussian(x, y, x0 + 0.10 * sign, y0, 0.12 + 0.01 * k, 0.16 + 0.012 * k)
    omega += 650.0 * np.sin(7.0 * y - 2.5 * x + phase) * plume

    # Small deterministic sample-to-sample perturbation keeps the tiny dataset nontrivial.
    ripple = np.sin((sample_id % 5 + 2) * np.pi * (x + 1.0) / 4.0 + phase)
    pressure += 180.0 * ripple * np.exp(-0.5 * ((y - 2.3) / 1.1) ** 2)
    velocity += 8.0 * ripple
    omega += 90.0 * ripple

    return np.stack([pressure, velocity, omega], axis=0).astype(np.float32)


def _coarsen(hr: np.ndarray, scale: int) -> np.ndarray:
    if scale != 2:
        raise ValueError("The bundled example currently supports scale=2 only.")
    blurred = np.stack(
        [gaussian_filter(hr[c], sigma=1.15, mode="nearest") for c in range(hr.shape[0])],
        axis=0,
    )
    c, h, w = blurred.shape
    if h % scale or w % scale:
        raise ValueError("HR dimensions must be divisible by scale.")
    return blurred.reshape(c, h // scale, scale, w // scale, scale).mean(axis=(2, 4)).astype(np.float32)


def _write_h5(
    out_path: Path,
    lr: np.ndarray,
    hr: np.ndarray,
    *,
    train_count: int,
    seed: int,
    x_hr: np.ndarray,
    y_hr: np.ndarray,
) -> None:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(hr.shape[0])
    lr = lr[perm]
    hr = hr[perm]

    means = np.mean(hr[:train_count], axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    stds = np.std(hr[:train_count], axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    stds = np.maximum(stds, np.float32(1e-8))

    lr_n = (lr - means[None, :, None, None]) / stds[None, :, None, None]
    hr_n = (hr - means[None, :, None, None]) / stds[None, :, None, None]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        meta = f.create_group("meta")
        meta.create_dataset("meanshr", data=means)
        meta.create_dataset("stdshr", data=stds)
        meta.create_dataset("meanslr", data=means)
        meta.create_dataset("stdslr", data=stds)
        meta.create_dataset("perm", data=perm.astype(np.int64))
        meta.create_dataset("train_idx", data=perm[:train_count].astype(np.int64))
        meta.create_dataset("test_idx", data=perm[train_count:].astype(np.int64))
        meta.create_dataset("channel_names", data=np.asarray(CHANNEL_NAMES, dtype=object), dtype=string_dtype)
        meta.create_dataset("x_hr", data=x_hr.astype(np.float32))
        meta.create_dataset("y_hr", data=y_hr.astype(np.float32))
        meta.attrs["note"] = "Synthetic analytic smoke-test data; not CFD or research data."
        meta.attrs["normalization"] = "HR training mean/std applied to both LR and HR"
        meta.attrs["scale"] = 2

        kwargs = dict(compression="gzip", compression_opts=4, chunks=True)
        train = f.create_group("train")
        test = f.create_group("test")
        train.create_dataset("lr", data=lr_n[:train_count], **kwargs)
        train.create_dataset("hr", data=hr_n[:train_count], **kwargs)
        test.create_dataset("lr", data=lr_n[train_count:], **kwargs)
        test.create_dataset("hr", data=hr_n[train_count:], **kwargs)


def _write_preview(path: Path, lr: np.ndarray, hr: np.ndarray) -> None:
    lr_up = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
    residual = hr - lr_up
    titles = ("POD-like / coarse input", "analytic target", "missing fine-scale correction")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, name in enumerate(CHANNEL_NAMES):
        fields = (lr_up[row], hr[row], residual[row])
        limits = []
        for col, field in enumerate(fields):
            if col < 2:
                vmin = min(float(lr_up[row].min()), float(hr[row].min()))
                vmax = max(float(lr_up[row].max()), float(hr[row].max()))
            else:
                vmax_abs = float(np.max(np.abs(field)))
                vmin, vmax = -vmax_abs, vmax_abs
            im = axes[row, col].imshow(field, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col])
            if col == 0:
                axes[row, col].set_ylabel(name.replace("_", " "))
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.03)
    fig.suptitle("Bundled synthetic RASGAN smoke-test pair", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate(
    output: Path,
    preview: Path,
    *,
    samples: int = 40,
    train_count: int = 32,
    hr_size: int = 32,
    seed: int = 123,
) -> None:
    if not 1 <= train_count < samples:
        raise ValueError("train_count must be smaller than samples.")
    if hr_size % 2:
        raise ValueError("hr_size must be even.")

    x1 = np.linspace(-1.0, 3.0, hr_size, dtype=np.float32)
    y1 = np.linspace(0.0, 4.0, hr_size, dtype=np.float32)
    x, y = np.meshgrid(x1, y1, indexing="xy")

    rng = np.random.default_rng(seed)
    hr = np.stack([_field_sample(x, y, rng, i) for i in range(samples)], axis=0)
    lr = np.stack([_coarsen(sample, 2) for sample in hr], axis=0)

    _write_h5(output, lr, hr, train_count=train_count, seed=seed, x_hr=x1, y_hr=y1)
    _write_preview(preview, lr[0], hr[0])
    print(f"Wrote {output} with train={train_count}, test={samples-train_count}")
    print(f"LR shape: {lr.shape}; HR shape: {hr.shape}")
    print(f"Wrote preview: {preview}")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Generate the bundled synthetic RASGAN example.")
    p.add_argument("--out", type=Path, default=here / "demo_flow.h5")
    p.add_argument("--preview", type=Path, default=here / "preview.png")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--train-count", type=int, default=32)
    p.add_argument("--hr-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    generate(
        args.out,
        args.preview,
        samples=args.samples,
        train_count=args.train_count,
        hr_size=args.hr_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
