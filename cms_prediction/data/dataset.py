# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Definition of a CMS dataset."""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union


class CmsDataset(torch.utils.data.Dataset):
    """Torch Dataset for CMS model training and inference."""

    def __init__(
        self,
        base_path: Union[str, Path],  # str or pathlib.Path
        max_tiles: Optional[int] = 5000,  # can be None
        preload_items: bool = True,
    ):
        """Init CmsDataset instance.

        Parameters
        ----------
        base_path : Optional[str, Path]
            Location of the dataset on disk.
        max_tiles : Optional[int]
            Maximmum number of tiles per WSI. If None, all tiles are kept.
        preload_items : bool
            Preload features into memory.
        """    
        self.base_path = Path(base_path)
        self.max_tiles = max_tiles
        self.preload_items = preload_items
        self.labels = pd.read_csv(self.base_path / "labels.csv")

        if self.preload_items:
            self.preloaded_items = []
            for idx in tqdm(range(len(self))):
                item = self.load_item(idx)
                self.preloaded_items.append(item)
        else:
            self.preloaded_items = None

    def load_item(self, idx):
        """Load one item of the dataset at a given index."""
        ft_path = (
            self.base_path
            / "features"
            / f"{self.labels.iloc[idx]['slide']}.npy"
        )
        features_and_coords = np.load(ft_path)
        if self.max_tiles is not None:
            n_tiles = len(features_and_coords)
            if n_tiles > self.max_tiles:
                idx = np.random.choice(
                    range(len(features_and_coords)), self.max_tiles, replace=False
                )
                features_and_coords = features_and_coords[idx]
        item = {
            **self.labels.iloc[idx],
            "level": features_and_coords[:, 0].astype(int),
            "coords": features_and_coords[:, 1:3].astype(int),
            "features": features_and_coords[:, 3:].astype(np.float32),
        }
        return item

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.preload_items:
            return self.preloaded_items[idx]
        else:
            return self.load_item(idx)
