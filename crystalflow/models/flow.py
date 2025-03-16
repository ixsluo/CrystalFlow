import math
import logging
from typing import Any, Literal

import hydra
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from omegaconf import DictConfig
from tqdm import tqdm



class FlowModule(nn.Module):
    def __init__(
        self,
        mode: Literal['csp', 'dng'],
        vfield: nn.Module,
        cost: DictConfig,
        lattice_polar_sigma: float,
        *args,
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        self.vfield = vfield
        self.cost = cost
        self.lattice_polar_sigma = lattice_polar_sigma

    def forward(self, batch):
        batch_size = batch.batch_size
        times = torch.rand(batch_size, device=batch.frac_coords.device)

        l_polar_1 = batch.l_polar
        l_polar_0 = self.sample_lattice_polar_like(l_polar_1)
        l_polar_tar = l_polar_1 - l_polar_0

        f_1 = batch.frac_coords
        f_0 = torch.rand_like(f_1)
        f_tar = (f_1 - f_0 - 0.5) % 1 - 0.5

        input_l_polar = l_polar_0 + times[:, None] * l_polar_tar
        input_f = f_0 + times.repeat_interleave(batch.num_atoms)[:, None] * f_tar
        input_atom_types = batch.atom_types

        type_pred, l_polar_pred, frac_coords_pred = self.vfield(
            t=times,
            num_atoms=batch.num_atoms,
            atom_types=batch.atom_types,
            frac_coords=input_f,
            l_polar=input_l_polar,
            node2graph=batch.batch
        )

        if self.mode == "csp":
            loss_l_polar = F.mse_loss(l_polar_pred, l_polar_tar)
            loss_frac_coords = F.mse_loss(frac_coords_pred, f_tar)
            loss = self.cost["l_polar"] * loss_l_polar \
                 + self.cost["frac_coords"] * loss_frac_coords
            return {"loss": loss, "loss_l_polar": loss_l_polar, "loss_frac_coords": loss_frac_coords}
        elif self.mode == "dng":
            raise NotImplementedError("dng")

    def sample_lattice_polar_like(self, l1):
        l0 = torch.randn([l1.shape[0], 6], device=l1.device) * self.lattice_polar_sigma
        l0[:, -1] = l0[:, -1] + 1
        return l0

    def sample_lattice_polar(self, batch_size):
        l0 = torch.randn([batch_size, 6], device=self.device) * self.lattice_polar_sigma
        l0[:, -1] = l0[:, -1] + 1
        if self.from_cubic:
            l0[:, :5] = 0
        return l0