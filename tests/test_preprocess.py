from pathlib import Path

import h5py
import numpy as np

from rasgan.cli.preprocess import prepare_dataset
from rasgan.data import load_h5


def test_prepare_dataset_from_cnhw_hdf5(tmp_path: Path):
    rng = np.random.default_rng(7)
    # Source layout follows the original MATLAB exporter: C,N,H,W.
    hr = rng.normal(size=(3, 10, 8, 8)).astype(np.float32)
    lr = hr[:, :, ::2, ::2] + 0.01

    hr_path = tmp_path / "hr.mat"
    lr_path = tmp_path / "lr.mat"
    with h5py.File(hr_path, "w") as f:
        f.create_dataset("HR", data=hr)
    with h5py.File(lr_path, "w") as f:
        f.create_dataset("LR", data=lr)

    out = tmp_path / "paired.h5"
    prepare_dataset(
        hr_path=str(hr_path),
        lr_path=str(lr_path),
        output_path=str(out),
        train_count=8,
        seed=123,
    )

    data = load_h5(str(out))
    assert data.hr_train.shape == (8, 3, 8, 8)
    assert data.lr_train.shape == (8, 3, 4, 4)
    assert data.hr_test.shape == (2, 3, 8, 8)
    assert data.lr_test.shape == (2, 3, 4, 4)

    # HR training data are normalized with their own per-channel statistics.
    np.testing.assert_allclose(data.hr_train.mean(axis=(0, 2, 3)), 0.0, atol=1e-6)
    np.testing.assert_allclose(data.hr_train.std(axis=(0, 2, 3)), 1.0, atol=2e-6)
