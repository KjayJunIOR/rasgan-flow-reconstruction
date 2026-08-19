# TensorLayer/SRGAN upstream comparison

## Scope

This document records the source-provenance comparison used to prepare the
public RASGAN repository. The comparison target was the linked current
TensorLayer/SRGAN `master` repository:

- https://github.com/tensorlayer/SRGAN/tree/master

This is not a claim that the exact historical commit used at the beginning of
development has been identified.

## Summary

RASGAN has clear historical lineage from the TensorLayer/SRGAN example, but the
present repository is not a framework port of that codebase. The strongest
remaining similarities are the broad CNN generator skeleton, historical naming,
and a small compatibility API. The scientific model, discriminator, losses,
training controller, checkpointing, data pipeline, and alternate generator
families have been substantially redesigned in TensorFlow/Keras.

| Component | Relationship to linked TensorLayer/SRGAN baseline |
| --- | --- |
| Project concept/history | Clear lineage; upstream SRGAN was the initial baseline |
| `SRGAN_g` name | Historical name retained in the CNN path |
| Residual-block naming | Historical naming retained |
| Stem/trunk/long-skip generator pattern | Recognizable high-level lineage |
| Current dense multi-kernel residual/ECA block | Substantially redesigned |
| Per-channel pressure/velocity/vorticity heads | Project-specific |
| Residual field prediction | Project-specific |
| Conditional 12-channel discriminator | Project-specific architecture |
| Spectral normalization and SE discriminator path | Added relative to baseline |
| Scientific reconstruction/spectral loss stack | Project-specific |
| Relativistic-average softplus GAN training | Added relative to baseline |
| Physics and POD-sidecar losses | Project-specific |
| EMA, validation steering, restart metadata | Project-specific infrastructure |
| Transformer and composite generators | Project-specific/experimental additions |
| CFD HDF5 preprocessing and data contract | Project-specific |
| TensorFlow/Keras implementation | Current implementation layer |
| `tf_layers.py` compatibility API | Retains TensorLayerX-like calling conventions, implemented locally |

## Generator

The linked TensorLayer/SRGAN generator follows the familiar SRGAN organization:
an initial convolution, a residual stack, a post-trunk convolution and long
skip, an upsampling path, and an output convolution. RASGAN retains that broad
family resemblance.

The current residual block is materially different: it uses a dense
multi-kernel sequence, Swish activations, reflection padding, Efficient Channel
Attention, and residual scaling rather than the baseline two-convolution
BatchNorm residual block. The shared trunk also fans out into separate physical
channel heads.

## Discriminator

The present discriminator is not the baseline scalar SRGAN discriminator. It is
a conditional PatchGAN-style model that evaluates the candidate/reference field
together with the POD/coarse condition, residual fields, and learned edge maps.
It includes spectral normalization, Squeeze-and-Excitation, a patch head, and a
global head.

## Training and losses

The current training code adds a domain-specific loss system and controller,
including robust pixel, gradient, residual, low-wavenumber, spectral, energy,
feature-matching, total-variation, decorrelation, covariance, relativistic GAN,
R1, optional physics/POD terms, EMA, adaptive steering, and checkpoint metadata.
These components are central to the current repository and are not simply a
translation of the TensorLayer/SRGAN training script.

## Compatibility layer

`src/rasgan/tf_layers.py` intentionally preserves a small TensorLayerX-like API
surface so evolved model code can use familiar names while executing on native
TensorFlow/Keras. This compatibility intent should remain documented even though
TensorLayer/TensorLayerX is not a runtime dependency.

## Release implication

The linked upstream repository itself states that it is for academic and
non-commercial use only. RASGAN therefore uses matching academic/non-commercial
source-available terms and keeps explicit upstream attribution. The repository
should not be described as OSI-approved open-source software while commercial
use is restricted.
