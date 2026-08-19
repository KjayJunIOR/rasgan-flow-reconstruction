from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from rasgan.data import load_h5


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a RASGAN paired HDF5 dataset.")
    p.add_argument("data", help="Path to HDF5 dataset")
    args = p.parse_args()

    d = load_h5(args.data)
    print(f"file: {Path(args.data).resolve()}")
    print(f"train LR: {d.lr_train.shape} {d.lr_train.dtype}")
    print(f"train HR: {d.hr_train.shape} {d.hr_train.dtype}")
    print(f"test  LR: {d.lr_test.shape} {d.lr_test.dtype}")
    print(f"test  HR: {d.hr_test.shape} {d.hr_test.dtype}")
    print(f"HR mean: {np.asarray(d.means_hr)}")
    print(f"HR std : {np.asarray(d.stds_hr)}")
    print(f"permutation stored: {d.perm is not None}")

    with h5py.File(args.data, "r") as f:
        print("datasets:")
        f.visititems(lambda name, obj: print(f"  {name}: {obj.shape} {obj.dtype}") if isinstance(obj, h5py.Dataset) else None)


if __name__ == "__main__":
    main()
