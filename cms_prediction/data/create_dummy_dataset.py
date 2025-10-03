# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utils to create a toy dummy dataset."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def run(dest_folder: str = None):
    """Create a dummy dataset named DummyDataset.
    
    Placed in `dest_folder` if `dest_folder`is provided,
    otherwise placed in a `.data` folder within the repository.
    """
    N_patients = 60
    N_slides = 80
    dummy_dataset_name = "DummyDataset"
    if dest_folder is None:
        dataset_path = Path(__file__).parent.parent / ".data" / dummy_dataset_name
    else:
        dataset_path = Path(dest_folder) / dummy_dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    features_dir = dataset_path / "features"
    features_dir.mkdir(exist_ok=True)

    df_labels = pd.DataFrame(
        columns=["patient", "slide", *[f"RF.CMS{i}" for i in range(1, 5)]]
    )
    for i in range(N_slides):
        patient_id = f"patient_{i % N_patients}"
        name = f"slide_{i}"
        maj_cms = i % 4
        label = np.random.rand(4)
        label[maj_cms] += 1
        label = label / label.sum()
        coords = np.random.randint(30, size=(150, 2))
        coords = np.unique(coords, axis=0)
        n_tiles = len(coords)
        level = 16 * np.ones(n_tiles)
        features = np.random.rand(n_tiles, 40)
        features[:, maj_cms * 10 : (maj_cms + 1) * 10] += 1
        concat_features = np.hstack([level[:, np.newaxis], coords, features])
        ft_path = features_dir / f"{name}.npy"
        np.save(ft_path, concat_features)
        df_labels.loc[i] = [patient_id, name, *label]
    df_labels.to_csv(dataset_path / "labels.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest_folder", type=str, default=None)
    args = parser.parse_args()
    dest_folder = args.dest_folder
    run(dest_folder)
