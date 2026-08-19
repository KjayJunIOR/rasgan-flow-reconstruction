# Development validation snapshot

This file records development-time runtime checks for the public repository. It
is **not** a scientific benchmark and does not establish CFD accuracy or
performance on the private research dataset.

## Validated execution paths

The bundled synthetic-data path has been exercised through package tests,
model construction, training, checkpointing, restart/inference, and MATLAB
export.

Two GPU environments were used during repository validation:

- a 4 GB WSL2 GPU environment with TensorFlow 2.21 for compatibility smoke
  testing; and
- an NVIDIA GeForce RTX 5070 Ti (16 GB, Blackwell) Docker environment with the
  project's already-working TensorFlow 2.17 stack for the realistic-size
  synthetic validation described below.

The Blackwell container was intentionally left unchanged during validation.
The repository was installed editable with `--no-deps` so the known-working
TensorFlow/CUDA stack was not upgraded or rebuilt.

## Realistic synthetic Blackwell check

A deterministic analytic dataset was generated at 2x scale with
`3 x 96 x 96` LR fields and `3 x 192 x 192` HR targets. With `float32`, batch
size 4, mixed precision disabled, explicit `--xla` disabled, and gradient
reweighting disabled, the following completed successfully:

- 3 content-pretraining epochs;
- 5 adversarial epochs;
- best/epoch checkpoint saves and best-init restore;
- EMA/checkpoint inference in a fresh process; and
- MATLAB export of all 8 validation samples.

Observed GPU memory reservation stayed near 13.2 GiB on the 16 GB card without
an out-of-memory failure in the baseline stability run. TensorFlow may reserve
substantially more memory than the instantaneous working set, so this number is
an environment observation rather than a portable memory requirement.

## Gradient-reweighting experiments

Gradient-magnitude reweighting is **off by default** and remains experimental.
On the same 96 -> 192, batch-4, float32 Blackwell setup:

- reweighting the broad default term set every step caused host/pinned-memory
  pressure and did not complete;
- `res,tv` reweighting every step completed but was comparatively expensive;
- `res,tv` with `--grad-reweight-every 4` and deterministic norms disabled
  completed a 3+5 epoch stability run, with modest steady-state overhead; and
- enabling `--grad-reweight-deterministic-norms` for that configuration caused
  a GPU `ResourceExhaustedError` in the tested 16 GB environment.

For that reason, the repository does not enable gradient reweighting by
default. If experimenting on a memory-constrained GPU, select a small term set
and cadence explicitly rather than assuming the broad config defaults are
portable. One development-tested command was:

```bash
rasgan-train \
  --data DATA.h5 \
  --batch-size 4 \
  --grad-reweight \
  --grad-reweight-keys res,tv \
  --grad-reweight-every 4
```

This is a development observation, not a recommended scientific
hyperparameter setting.

## Mixed precision and XLA

The validated baseline uses `float32`. Mixed precision remains opt-in and was
not promoted to the validated baseline because the custom training loop should
be reviewed for explicit loss-scaling behavior before relying on it for
production runs.

The CLI `--xla` flag also remains opt-in. In the Blackwell Docker environment,
TensorFlow itself reported compiling at least one XLA cluster even when the
project's explicit XLA flag was not enabled. The repository does not attempt to
suppress or override that framework/runtime behavior.

## Runtime log noise

The Blackwell Docker environment emits substantially more TensorFlow/CUDA/XLA
startup and kernel-timing messages than the H100 NVL HPC environment used in
earlier development. Examples observed during otherwise successful runs
include duplicate cuFFT/cuDNN/cuBLAS factory registration messages, NUMA-node
messages, repeated GPU timer warnings, and PTX-target warnings.

These messages are produced by the TensorFlow/CUDA runtime rather than RASGAN's
application logger. They are intentionally not globally suppressed in the
project because aggressive filtering could also hide actionable allocator or
CUDA failures.

## Regression checks retained in the repository

The test suite includes a graph-mode adversarial regression test that traces the
real discriminator-loss wrapper under `tf.function`. It exists because an eager
smoke test did not catch a Python-boolean-versus-symbolic-tensor control-flow
bug in discriminator conditioning.

The public baseline defaults are also covered by a small configuration test so
cleanup work does not silently enable mixed precision, explicit XLA, or
gradient reweighting.
