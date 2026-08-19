# Project-file audit

## Core completeness assessment

The supplied files are sufficient to construct a coherent repository and to
represent the complete current training/inference implementation.

Included and resolved:

- training CLI;
- inference CLI;
- HDF5 preprocessing logic;
- configuration dataclasses;
- HDF5 data loader and TensorFlow datasets;
- RRDB/ECA, Transformer, and composite generators;
- conditional spectral PatchGAN discriminator;
- content, spectrum, GAN, physics, and POD losses;
- schedules, EMA, validation steering, and checkpointing;
- physics operators and Poisson solvers; and
- custom TensorFlow/Keras layer compatibility wrappers.

All 31 required Python source files from the supplied archive have a corresponding
module under `src/rasgan/`; see [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md). All
internal imports resolve after packaging, and all Python files pass bytecode
compilation. No TensorLayer or TensorLayerX runtime import remains.

## Upstream provenance review

The present repository was compared against the linked TensorLayer/SRGAN
`master` before public packaging. The review found clear historical lineage in
the broad CNN generator organization, historical naming, and the local
TensorLayerX-like compatibility surface, while the current residual/ECA blocks,
per-field heads, conditional discriminator, scientific loss system, physics/POD
terms, training/checkpoint infrastructure, Transformer/composite paths, and data
pipeline are materially different.

The detailed component-level summary is in
[`UPSTREAM_COMPARISON.md`](UPSTREAM_COMPARISON.md). The exact historical upstream
commit used when development first began was not supplied, so the audit does not
claim that revision has been identified.

## Repository construction fixes

The supplied ZIP contained package modules at its archive root, while the
entrypoints import `rasgan.*`. The public repository therefore places the
modules in an installable `src/rasgan` package.

Other repository-level corrections:

- removed `__pycache__` artifacts;
- excluded `tf_layersbackup.py` and the empty `stages` package;
- replaced personal absolute data/checkpoint paths with repository-relative
  defaults;
- refactored the hard-coded preprocessing script into a CLI;
- retained the legacy `--g_upsample` spelling as an alias while adding the
  conventional `--g-upsample` form;
- removed the supplied `--adv-weight` parser flag because it was never read or
  applied by the entrypoint; the scheduled adversarial weight remains intact;
- corrected inference output-directory handling and the POD-mode help text;
- added packaging, tests, CI, documentation, a synthetic example, and notices;
- documented actual kernel/dilation behavior; and
- validated scale relationships from the paired data shapes.

## Items not supplied

These are **not blockers** for the public methods repository, but they would
strengthen it if release is permitted:

1. **Exact production environment** — Python, TensorFlow, CUDA, cuDNN, GPU,
   and package versions. The source indicates TensorFlow 2.17-era development,
   but no lock file or environment export was included.
2. **Final production command/configuration** — the CLI exposes the settings,
   but the exact command used for the preliminary result is not archived here.
3. **A trained checkpoint** — even a small generator-only EMA checkpoint would
   allow immediate inference. It is optional and may be too large or restricted.
4. **Quantitative evaluation output** — the notes provide qualitative fields
   and line plots, but no final CSV/JSON table of baseline-versus-model metrics
   was included.
5. **Training-history export** — a compact loss/validation history would support
   portfolio plots without publishing the dataset.

The repository is usable without these items. If provided later, place only
redistributable artifacts under Git LFS or a release asset rather than in the
normal Git history.

## Documentation/code discrepancy requiring author confirmation

The model notes and stale source comments describe "symmetric dilations."
However, both the RRDB and composite generator source set `dilation=1` for all
five convolutions and vary the kernel size as `3,5,7,5,3` instead. This draft
preserves executable behavior and describes it as a symmetric multi-kernel
block.

Before publishing, confirm whether:

- the intended released model is the current `dilation=1` implementation; or
- the production checkpoint used dilation rates `1,2,3,2,1` and a corresponding
  source version should replace these files.

A checkpoint trained with one architecture cannot be assumed compatible with a
silent architectural change.

## Validation performed

- Eight non-TensorFlow tests pass, covering the HDF5 contract, preprocessing,
  source compilation/import structure, and targeted source regressions.
- The built wheel installs without dependencies and the packaged data-inspection
  CLI reads the committed synthetic dataset successfully.
- The synthetic dataset regenerates deterministically and contains finite
  paired `float32` tensors with the documented 2x scale relationship.

TensorFlow was not available in the repository-construction environment, so the
model forward pass and full training loop could not be executed here. The draft
includes an optional TensorFlow smoke test and `rasgan-selfcheck`. Run both in
the actual TensorFlow environment before publication.
