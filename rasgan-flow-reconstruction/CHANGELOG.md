# Changelog

## Unreleased

### Fixed

- Made discriminator conditioning graph-safe under `tf.function` by keeping the
  training flag as a Python tracing-time boolean while using a TensorFlow
  predicate for runtime conditioning dropout.
- Added an adversarial graph regression test covering the real discriminator
  loss wrapper.
- Made the compatibility `Module.call()` signature expose its first input
  explicitly, removing Keras auto-build/introspection warnings seen during
  model construction.

### Maintenance

- Cleaned local formatting and comments in `rasgan.losses.gan` without changing
  the GAN objective or conditioning math.
- Simplified gradient-reweight setup to read directly from `TrainConfig`,
  removing inconsistent fallback values from the training path.
- Clarified that gradient reweighting is experimental and documented the
  development validation matrix in `docs/VALIDATION.md`.
