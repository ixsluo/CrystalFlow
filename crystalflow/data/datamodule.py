import random
from pathlib import Path
from dataclasses import dataclass

import hydra
import torch
import numpy as np
import omegaconf
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasets: Dataset|list[Dataset],
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_datasets: list[Dataset]|None = None,
        test_datasets: list[Dataset]|None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if isinstance(train_datasets, list) or OmegaConf.is_list(train_datasets):
            self.train_dataset = ConcatDataset(train_datasets)
        else:
            self.train_dataset = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self, shuffle=False) -> list[DataLoader]|None:
        return [
            DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ] if self.val_datasets else None

    def test_dataloader(self, shuffle=False) -> list[DataLoader]|None:
        return [
            DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ] if self.test_datasets else None
