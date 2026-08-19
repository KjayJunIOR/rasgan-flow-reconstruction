# Data format

## HDF5 contract

The preferred training file uses compressed NCHW arrays:

```text
/train/lr   float32 [N_train, 3, H_lr, W_lr]
/train/hr   float32 [N_train, 3, H_hr, W_hr]
/test/lr    float32 [N_test,  3, H_lr, W_lr]
/test/hr    float32 [N_test,  3, H_hr, W_hr]

/meta/meanshr   float32 [3]
/meta/stdshr    float32 [3]
/meta/meanslr   float32 [3]
/meta/stdslr    float32 [3]
/meta/perm      int64   [N_total]   optional but recommended
/meta/train_idx int64   [N_train]   optional
/meta/test_idx  int64   [N_test]    optional
```

The loader also recognizes flat names such as `lr_train`, `hr_train`,
`lr_test`, and `hr_test`, plus several common aliases.

## Supported spatial relationships

The training pipeline supports:

- same-grid refinement: `H_hr == H_lr` and `W_hr == W_lr`; or
- 2x super-resolution: `H_hr == 2*H_lr` and `W_hr == 2*W_lr`.

Other scale factors are rejected by the current training path. The paired array
shapes are authoritative; `--scale` is checked only as an expected value.

## Normalization

For each physical channel, preprocessing computes

```text
mean_c = mean(HR training channel c)
std_c  = std(HR training channel c)
```

and applies those same statistics to both the LR and HR fields. The validation
or test split is excluded from the statistics.

This approach preserves the physical relationship between low- and
high-fidelity channels while preventing validation leakage.

## Preparing MATLAB arrays

The original preprocessing source expected two MATLAB-v7.3/HDF5 files with:

```text
HR : [C, N, H_hr, W_hr]
LR : [C, N, H_lr, W_lr]
```

The public CLI accepts classic MAT or v7.3/HDF5 inputs and either `C,N,H,W` or
`N,C,H,W`:

```bash
rasgan-prepare-data \
  --hr HRresidual.mat \
  --lr LRresidual.mat \
  --hr-key HR \
  --lr-key LR \
  --input-layout cnhw \
  --train-count 2000 \
  --seed 123 \
  --out data/residual_data.h5
```

The saved `perm` is important when an external POD coefficient sidecar must be
aligned with the shuffled train/test order.

## POD sidecar

An optional MAT file may contain, case-insensitively:

```text
phiu, phiv, phip       spatial POD modes [H_lr*W_lr, M]
TimCoeU, TimCoeV,
TimCoeP                coefficients [N_total, M] or [N_train, M]
```

The loader accepts transposed MATLAB layouts when the mode/sample dimension can
be inferred. The first `--pod-k` columns are used.

For training with a saved H5 permutation, coefficient arrays should cover all
samples so the same permutation can be applied before splitting. Transformer
and composite generators concatenate the selected U, V, and P coefficients for
FiLM conditioning.

## Inference output

`rasgan-infer` writes a MATLAB file containing:

```text
valid_gen     generated normalized fields [N, 3, H_hr, W_hr]
valid_lr      normalized input fields
valid_hr      normalized target fields
weights_path  loaded weight-file path
```

When POD coefficients are supplied, the test coefficients are also included.
The current inference export preserves normalized model-space values; use the
stored HR mean and standard deviation to return to physical units.
