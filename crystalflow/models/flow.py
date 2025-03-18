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
from hydra.utils import instantiate

from crystalflow.models.property_embeddings import PropertyEmbedding
from crystalflow.models.type_model import AtomTypeModuleBase


class FlowModule(nn.Module):
    def __init__(
        self,
        mode: Literal['CSP', 'DNG'],
        vfield: nn.Module,
        type_model: AtomTypeModuleBase,
        cost: DictConfig,
        lattice_polar_sigma: float,
        properties: list[str],  # names of properties
        property_embeddings: dict[str, PropertyEmbedding],  # config dicts of named property embedding modules
        *args,
        **kwargs
    ):
        super().__init__()
        if mode == "DNG" and not type_model.decodeable:
            raise RuntimeError("A decodeable type module is required in DNG mode")

        self.mode = mode
        self.vfield = vfield
        self.type_model = type_model

        self.cost = cost
        self.lattice_polar_sigma = lattice_polar_sigma
        self.properties = properties if properties is not None else []
        unknown_properties = set(properties) - set(property_embeddings.keys())
        if unknown_properties:
            raise ValueError(f"Unknown properties: {unknown_properties}. You need to add into config `pl_model/model/default.yaml`")
        self.property_embeddings = nn.ModuleDict({name: model for name, model in property_embeddings.items() if name in properties})

    def forward(self, batch):
        batch_size = batch.batch_size
        device = batch.atom_types.device
        times = torch.rand(batch_size, device=device)

        a_1 = self.type_model(batch.atom_types)
        a_0 = self.type_model.random_encoding_from(batch.atom_types)
        a_tar = a_1 - a_0  # if CSP, MUST be zeros

        l_polar_1 = batch.l_polar
        l_polar_0 = self.sample_lattice_polar_like(l_polar_1)
        l_polar_tar = l_polar_1 - l_polar_0

        f_1 = batch.frac_coords
        f_0 = torch.rand_like(f_1)
        f_tar = (f_1 - f_0 - 0.5) % 1 - 0.5

        input_a = a_0 + times.repeat_interleave(batch.num_atoms)[:, None] * a_tar
        input_l_polar = l_polar_0 + times[:, None] * l_polar_tar
        input_f = f_0 + times.repeat_interleave(batch.num_atoms)[:, None] * f_tar

        type_pred, l_polar_pred, frac_coords_pred = self.vfield(
            t=times,
            num_atoms=batch.num_atoms,
            atom_types=input_a,
            frac_coords=input_f,
            l_polar=input_l_polar,
            node2graph=batch.batch
        )

        if self.mode == "CSP":
            loss_l_polar = F.mse_loss(l_polar_pred, l_polar_tar)
            loss_frac_coords = F.mse_loss(frac_coords_pred, f_tar)
            loss = self.cost["l_polar"] * loss_l_polar \
                 + self.cost["frac_coords"] * loss_frac_coords
            return {"loss": loss, "loss_l_polar": loss_l_polar, "loss_frac_coords": loss_frac_coords}
        elif self.mode == "DNG":
            loss_atom_types = F.mse_loss(type_pred, a_tar)
            loss_l_polar = F.mse_loss(l_polar_pred, l_polar_tar)
            loss_frac_coords = F.mse_loss(frac_coords_pred, f_tar)
            loss = self.cost["l_polar"] * loss_l_polar \
                 + self.cost["frac_coords"] * loss_frac_coords \
                 + self.cost["atom_types"] * loss_atom_types
            return {"loss": loss, "loss_l_polar": loss_l_polar, "loss_frac_coords": loss_frac_coords, "loss_atom_types": loss_atom_types}
        else:
            raise RuntimeError(f"Unknown mode: {self.mode}")

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