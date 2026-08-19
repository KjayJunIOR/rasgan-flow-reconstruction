# RASGAN model card

## Model summary

RASGAN is a conditional relativistic generative adversarial network for paired
scientific-field reconstruction. It maps a three-channel coarse or
POD-reconstructed field to a same-grid or 2x reference field. The research
channels are pressure, crossflow-normal velocity, and vorticity.

The primary generator is a TensorFlow/Keras RRDB-style convolutional model with
dense multi-kernel residual blocks, Efficient Channel Attention, a shared
trunk, and one output head per physical channel. Experimental Transformer and
composite CNN/Transformer generators are also included.

The discriminator is conditional: it evaluates the candidate/reference field,
the coarse/POD input, their residual, and three learned edge maps. Its patch and
global logits use spectral normalization and Squeeze-and-Excitation features.

## Intended use

- research and educational inspection of a scientific GAN implementation;
- development of paired coarse-to-reference flow-field reconstruction models;
- same-grid correction or 2x super-resolution experiments;
- portfolio demonstration of TensorFlow model, loss, training, checkpoint, and
  data-pipeline engineering.

The repository is not a pretrained general-purpose flow solver. No claim is
made that a model trained on one geometry, mesh, parameter range, or field
normalization will generalize to another.

## Inputs and outputs

Inputs and targets are `float32` NCHW arrays with three aligned physical
channels. HDF5 preprocessing computes per-channel mean and standard deviation
from the HR training split and applies those statistics to both LR and HR.

The main input/output relationships are:

```text
same grid: [N, 3, H, W] -> [N, 3, H, W]
2x scale : [N, 3, H, W] -> [N, 3, 2H, 2W]
```

Inference exports normalized generated, input, and target fields to a MATLAB
file. The saved HDF5 statistics are required to return predictions to physical
units.

## Training approach

Training has two stages:

1. generator-only content pretraining; and
2. alternating generator/discriminator adversarial refinement.

The implementation combines robust reconstruction, gradient, residual,
low-wavenumber, spectrum, total-variation, feature-matching, decorrelation,
covariance, relativistic-average adversarial, and optional physics/POD losses.
EMA generator weights, validation-aware steering, checkpoint metadata, and
resume support are included.

## Public evaluation status

The repository includes:

- a deterministic analytic smoke-test dataset for exercising the data contract;
- preliminary qualitative figures from the author's development notes; and
- source-level and data-pipeline tests.

The synthetic example is not CFD and is not a quality benchmark. The private
research dataset, production checkpoint, and final quantitative metric table are
not distributed. Development runtime checks on the bundled synthetic data are
recorded in `docs/VALIDATION.md`; those checks establish execution compatibility,
not scientific accuracy. Consequently, this release should be evaluated as a
methods implementation rather than a reproducible performance claim.

## Known limitations

- Training is computationally and memory intensive.
- GAN reconstructions can generate plausible but incorrect fine-scale content.
- Qualitative improvement does not guarantee conservation or equation
  consistency.
- The optional physics losses depend on grid spacing, boundary treatment,
  normalization, and variable semantics supplied by the user.

## Ethical and scientific use

Do not treat generated fields as ground truth without independent validation.
Report the baseline input, reference field, normalization, split construction,
error metrics, spectral behavior, and failure cases. Avoid using the model for
safety-critical decisions without domain-specific verification and uncertainty
analysis.

## License and provenance

Early development used the TensorLayer SRGAN example as a baseline, after
which the project was substantially reworked into a TensorFlow/Keras
scientific-field model. A comparison against the linked upstream `master` is
summarized in `docs/UPSTREAM_COMPARISON.md`.

The repository is distributed under academic and non-commercial source-available
terms in `LICENSE.md`, consistent with the restriction stated by the linked
TensorLayer/SRGAN repository. It should not be described as OSI-approved open
source because commercial use is restricted.
