# Supplied-source manifest

This manifest records the repository-placement audit for the Python files in
`rasgan.zip`. It verifies that every current source module needed by the supplied
training and inference entrypoints has a corresponding file under the installable
`src/rasgan/` package. It is a presence and import-structure audit, not a claim of
byte-for-byte identity: repository construction included documented packaging fixes
and targeted correctness corrections.

## Retained source files

| Supplied archive path | Repository path |
| --- | --- |
| `__init__.py` | `src/rasgan/__init__.py` |
| `config.py` | `src/rasgan/config.py` |
| `data.py` | `src/rasgan/data.py` |
| `ema.py` | `src/rasgan/ema.py` |
| `env.py` | `src/rasgan/env.py` |
| `losses/__init__.py` | `src/rasgan/losses/__init__.py` |
| `losses/content.py` | `src/rasgan/losses/content.py` |
| `losses/gan.py` | `src/rasgan/losses/gan.py` |
| `losses/physics.py` | `src/rasgan/losses/physics.py` |
| `losses/pod.py` | `src/rasgan/losses/pod.py` |
| `losses/spectrum.py` | `src/rasgan/losses/spectrum.py` |
| `losses/wrappers.py` | `src/rasgan/losses/wrappers.py` |
| `models/__init__.py` | `src/rasgan/models/__init__.py` |
| `models/composite_generator.py` | `src/rasgan/models/composite_generator.py` |
| `models/discriminator.py` | `src/rasgan/models/discriminator.py` |
| `models/factory.py` | `src/rasgan/models/factory.py` |
| `models/generator.py` | `src/rasgan/models/generator.py` |
| `models/transformer_generator.py` | `src/rasgan/models/transformer_generator.py` |
| `physics/__init__.py` | `src/rasgan/physics/__init__.py` |
| `physics/operators.py` | `src/rasgan/physics/operators.py` |
| `physics/poisson.py` | `src/rasgan/physics/poisson.py` |
| `pod_sidecar.py` | `src/rasgan/pod_sidecar.py` |
| `runtime.py` | `src/rasgan/runtime.py` |
| `schedules.py` | `src/rasgan/schedules.py` |
| `selfcheck.py` | `src/rasgan/selfcheck.py` |
| `steerer.py` | `src/rasgan/steerer.py` |
| `tf_layers.py` | `src/rasgan/tf_layers.py` |
| `training.py` | `src/rasgan/training.py` |
| `utils/__init__.py` | `src/rasgan/utils/__init__.py` |
| `utils/checkpoint.py` | `src/rasgan/utils/checkpoint.py` |
| `validate.py` | `src/rasgan/validate.py` |

**Result:** 31 of 31 required Python source files are represented.

## Standalone entrypoints supplied separately

| Supplied file | Repository form |
| --- | --- |
| `train.py` | `src/rasgan/cli/train.py` and `scripts/train.py` |
| `infer.py` | `src/rasgan/cli/infer.py` and `scripts/infer.py` |
| `solpreproc.py` | `src/rasgan/cli/preprocess.py` and `scripts/prepare_data.py` |

The training CLI retains every effective supplied option. The legacy
`--g_upsample` spelling remains as an alias to `--g-upsample`; the unused
`--adv-weight` parser option was removed because the supplied entrypoint never
read or applied it.

## Deliberately excluded archive artifacts

- `__pycache__/` and `.pyc` files: generated interpreter caches.
- `tf_layersbackup.py`: a duplicate backup, not imported by the project.
- `stages/__init__.py`: an empty package with no importing source or implementation.

## Added repository files

- `src/rasgan/cli/train.py`: packaged form of the supplied training entrypoint.
- `src/rasgan/cli/infer.py`: packaged form of the supplied inference entrypoint.
- `src/rasgan/cli/preprocess.py`: reusable form of the supplied preprocessing script.
- `src/rasgan/cli/inspect_data.py`: HDF5 contract inspection utility.
- `scripts/`: thin convenience wrappers.
- packaging, tests, documentation, CI, and the synthetic demonstration.

All package-relative imports are checked by `tests/test_source_integrity.py`.
