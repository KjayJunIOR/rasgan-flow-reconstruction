from pathlib import Path

import h5py
import numpy as np

from rasgan.data import load_h5


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "synthetic_flow" / "demo_flow.h5"


def test_committed_demo_contract():
    data = load_h5(str(DEMO))
    assert data.lr_train.shape == (32, 3, 16, 16)
    assert data.hr_train.shape == (32, 3, 32, 32)
    assert data.lr_test.shape == (8, 3, 16, 16)
    assert data.hr_test.shape == (8, 3, 32, 32)
    assert data.perm is not None and data.perm.shape == (40,)
    assert np.isfinite(data.lr_train).all()
    assert np.isfinite(data.hr_train).all()
    assert np.all(np.asarray(data.stds_hr) > 0)


def test_demo_is_explicitly_marked_synthetic():
    with h5py.File(DEMO, "r") as f:
        note = str(f["meta"].attrs["note"]).lower()
        assert "synthetic" in note
        assert "not cfd" in note
