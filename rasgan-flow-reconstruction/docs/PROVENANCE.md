# Provenance and release boundary

## Upstream starting point

Early development of RASGAN used the TensorLayer SRGAN example as a baseline:

- https://github.com/tensorlayer/SRGAN/tree/master

The current code is not a drop-in TensorLayer application. It uses native
TensorFlow/Keras and adds a scientific three-channel data contract, dense
multi-kernel residual blocks with ECA, per-field residual heads, conditional
residual/edge discriminator inputs, a spectral PatchGAN discriminator,
relativistic-average losses, flow-specific content and spectral terms, optional
physics/POD losses, EMA, validation steering, experimental Transformer/composite
generators, and a new checkpoint/inference system.

## Upstream comparison status

A source-structure comparison was performed against the linked TensorLayer/SRGAN
`master` before this repository was prepared for public release. The comparison
is documented in `UPSTREAM_COMPARISON.md`.

The clearest retained lineage is:

- the historical `SRGAN_g` / residual-block naming in parts of the CNN path;
- the broad generator pattern of input stem -> residual trunk -> long skip ->
  output/upsampling path; and
- `tf_layers.py`, which intentionally provides a small TensorLayerX-like calling
  surface implemented with TensorFlow/Keras.

The current discriminator, dense residual/ECA internals, per-channel output
heads, scientific losses, relativistic training system, physics/POD machinery,
EMA/validation/checkpoint logic, Transformer/composite paths, and scientific data
pipeline are materially different from the linked upstream example.


## Use-term decision

The linked TensorLayer/SRGAN README states that its repository is for academic
and non-commercial use only. The public RASGAN repository therefore uses
matching academic/non-commercial source-available terms in `../LICENSE.md`
rather than claiming a permissive commercial open-source license.

These terms apply to the author's original contributions subject to any
third-party rights. They do not grant broader rights in upstream or other
third-party material than the applicable rights holder can grant.

## Runtime dependency boundary

This repository imports TensorFlow, NumPy, SciPy, and h5py. It does not import
TensorLayer or TensorLayerX at runtime. `tf_layers.py` is a TensorFlow/Keras
compatibility layer that preserves the calling style used by the evolved model.

## Research-data boundary

The public repository excludes the original CFD snapshots, POD reconstructions,
production checkpoints, and exact training results. The committed HDF5 example
is generated analytically and is clearly labeled as non-CFD demonstration data.
The qualitative images are extracted from the author's supplied unpublished
model notes.
