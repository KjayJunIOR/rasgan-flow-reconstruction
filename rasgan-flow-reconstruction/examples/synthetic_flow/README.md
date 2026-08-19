# Synthetic flow-field smoke test

This directory contains a tiny analytic dataset that exercises the same public
interface as the research workflow:

- three channels: pressure-like, crossflow-velocity-like, and vorticity-like;
- paired `16 x 16` coarse inputs and `32 x 32` targets;
- train-only target statistics applied to both input and target fields;
- NCHW storage in a compressed HDF5 file.

It is **not CFD**, it is **not a scientific benchmark**, and it cannot validate
the physical accuracy of a trained model. Its purpose is to let a reader inspect
the data contract, build the TensorFlow models, and exercise a short training or
inference workflow without downloading the private multi-gigabyte datasets.

Regenerate the committed example:

```bash
python examples/synthetic_flow/generate_data.py
rasgan-inspect-data examples/synthetic_flow/demo_flow.h5
```

A one-epoch content-stage smoke run is:

```bash
rasgan-train \
  --data examples/synthetic_flow/demo_flow.h5 \
  --ckpt-dir runs/demo/checkpoints \
  --save-dir runs/demo/outputs \
  --batch-size 2 \
  --init-epochs 1 \
  --adv-epochs 0 \
  --save-every 1
```

The full model is large relative to this toy dataset. The command tests the
pipeline; it is not expected to produce a meaningful trained surrogate.
