import random
from pathlib import Path

import hydra
import torch
import numpy as np
import omegaconf
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from crystalflow.common.globals import PACKAGE_ROOT


@hydra.main(str(PACKAGE_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: DictConfig):
    pl_data: pl.LightningDataModule = instantiate(config.data)
    print(pl_data)
    train_loader = pl_data.train_dataloader()
    for batch in train_loader:
        print(batch)
        break
    print(len(pl_data.train_dataloader()))
    print(pl_data.val_dataloader()[0].dataset[0])


if __name__ == "__main__":
    main()
