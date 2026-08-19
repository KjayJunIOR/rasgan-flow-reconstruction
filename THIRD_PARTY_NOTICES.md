# Third-party notices and attribution

## Architectural starting point

Early development of this project used the following repository as a baseline:

- TensorLayer/SRGAN: https://github.com/tensorlayer/SRGAN/tree/master

The linked upstream README states that the repository is for **academic and
non-commercial use only** and directs commercial users to the TensorLayer team.
The public RASGAN repository therefore uses matching academic/non-commercial
terms rather than a permissive commercial license.

A comparison against the linked upstream `master` found that the present project
retains recognizable lineage primarily in the high-level generator organization,
some historical naming, and a TensorLayerX-like compatibility surface. The
current dense residual/ECA blocks, per-field heads, conditional spectral
PatchGAN discriminator, scientific loss stack, physics/POD constraints,
training controller, checkpoint/inference system, and Transformer/composite
paths are project-specific TensorFlow/Keras implementations. See
`docs/UPSTREAM_COMPARISON.md` for the component-level summary.

This repository does not import TensorLayer or TensorLayerX at runtime and does
not vendor the upstream TensorLayer/SRGAN repository as a standalone dependency.
Attribution here records project lineage; it does not imply endorsement by
TensorLayer.

## Runtime dependencies

Third-party packages are installed separately and are not vendored here:

- TensorFlow — Apache License 2.0
- NumPy — BSD 3-Clause License
- SciPy — BSD 3-Clause License
- h5py — BSD 3-Clause License
- Matplotlib — PSF-based license (demo extra)

Dependency licenses do not automatically determine the use terms of this
project.

## Scholarly method attribution

The implementation draws on SRGAN/ESRGAN, conditional PatchGAN, relativistic
GAN losses, spectral normalization, Squeeze-and-Excitation, Efficient Channel
Attention, R1 regularization, feature matching, and spectral/physics-aware
flow-field reconstruction methods. See `docs/REFERENCES.md` and the supplied
model notes for the complete bibliography.
