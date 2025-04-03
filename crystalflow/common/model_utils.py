import itertools

import numpy as np
import torch
import hydra
import lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader


def load_model(ckpt_path, load_data=False, testing=True, test_bs: int = None, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    config = DictConfig(ckpt["config"])
    pl_model: pl.LightningModule = instantiate(config.pl_model).__class__.load_from_checkpoint(ckpt_path)
    if load_data:
        if test_bs is None:
            test_bs = config.data.batch_size.test
        pl_data: pl.LightningDataModule = instantiate(config.data)
        if testing:
            dataset = ConcatDataset(pl_data.test_datasets)
            loader = DataLoader(dataset, batch_size=test_bs)
        else:
            raise NotImplementedError("Only loading test data is allowed.")
    else:
        loader = None
    return pl_model, loader, config
