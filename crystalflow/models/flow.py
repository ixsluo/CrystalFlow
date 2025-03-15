import math
import logging

import hydra
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm


class FlowModule(nn.Module):
    def __init__(
        self,
        vfield: nn.Module,
        *args,
        **kwargs
    ):
        super().__init__()
        self.vfield = vfield

    def forward(self, batch):
        batch_size = batch.batch_size
        #times = torch.rand(batch_size, device=batch.device)
