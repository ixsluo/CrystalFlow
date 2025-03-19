from pathlib import Path

import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from crystalflow.common.data_utils import Preprocess


class GraphDataset(Dataset):
    def __init__(
        self,
        raw_file,
        cache_file,
        properties: list[str],
        preprocess: Preprocess,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.raw_file = Path(raw_file)
        self.cache_file = Path(cache_file)
        if not self.cache_file.exists() and not self.raw_file.exists():
            raise FileNotFoundError(f"Neither {self.raw_file} nor {self.cache_file} exists.")

        if self.cache_file.exists() and self.raw_file.exists():
            # compare time
            if self.raw_file.stat().st_mtime > self.cache_file.stat().st_mtime:
                need_preprocess = True
            else:
                need_preprocess = False
        elif not self.cache_file.exists() and self.raw_file.exists():
            need_preprocess = True
        else:
            need_preprocess = False

        if need_preprocess:
            self.cache_data = preprocess.preprocess(raw_file)
            torch.save(self.cache_data, cache_file)
        else:
            print(f"Loading {cache_file} ...")
            self.cache_data = torch.load(cache_file)
        self.cache_data = [Data.from_dict(d) for d in self.cache_data]

    def __len__(self):
        return len(self.cache_data)

    def __getitem__(self, index) -> Data:
        return self.cache_data[index]
