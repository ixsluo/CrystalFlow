import math
import logging
from typing import Any, Literal, Protocol

import hydra
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate
from flow_matching.path.scheduler import Scheduler
from flow_matching.path import ProbPath

from crystalflow.models.property_embeddings import PropertyEmbedding
from crystalflow.models.types_probpath import AtomTypesProbPath
from crystalflow.models.lattice_probpath import LatticeProbPath


class ProbPathPartial(Protocol):
    def __call__(self, scheduler: Scheduler, *args, **kwargs) -> AtomTypesProbPath | LatticeProbPath:
        raise NotImplemented


class VectorFieldPartial(Protocol):
    def __call__(self, *args, **kwargs) -> nn.Module:
        raise NotImplemented


class FlowModule(nn.Module):
    def __init__(
        self,
        vector_field: VectorFieldPartial,
        scheduler: Scheduler,  # time scheduler
        cost: DictConfig,
        types_probpath: ProbPathPartial,
        lattice_probpath: ProbPathPartial,
        frac_coords_probpath: ProbPathPartial,
        properties: list[str] = None,  # names of properties
        property_embeddings: dict[str, PropertyEmbedding] = None,  # config dicts of named property embedding modules
        *args,
        **kwargs
    ):
        super().__init__()
        self.vector_field = vector_field()
        self.types_probpath: AtomTypesProbPath = types_probpath(scheduler=scheduler)
        self.lattice_probpath: LatticeProbPath = lattice_probpath(scheduler=scheduler)
        self.frac_coords_probpath = frac_coords_probpath(scheduler=scheduler)
        self.cost = cost

        self.properties = properties if properties is not None else []
        property_embeddings = property_embeddings if property_embeddings is not None else None
        unknown_properties = set(properties) - set(property_embeddings.keys())
        if unknown_properties:
            raise ValueError(f"Unknown properties: {unknown_properties}! You need to add them into config `pl_model/model/default.yaml`")
        self.property_embeddings = nn.ModuleDict({name: model for name, model in property_embeddings.items() if name in properties})

    def forward(self, batch):
        batch_size = batch.batch_size
        device = batch.atom_types.device
        times = torch.rand(batch_size, device=device)

        types_path_sample = self.types_probpath.sample_union(batch.atom_types, t=times.repeat_interleave(batch.num_atoms))
        lattice_path_sample = self.lattice_probpath.sample_union(batch, t=times)
        frac_coords_path_sample = self.frac_coords_probpath.sample_union(batch, t=times.repeat_interleave(batch.num_atoms))

        types_pred, lattice_pred, frac_coords_pred = self.vector_field(
            t=times,
            num_atoms=batch.num_atoms,
            atom_types=types_path_sample.x_t,
            lattice=lattice_path_sample.x_t,
            frac_coords=frac_coords_path_sample.x_t,
            node2graph=batch.batch,
        )

        if types_pred.size == 0:
            loss_atom_types = 0
        else:
            loss_atom_types = self.types_probpath.get_loss(types_pred, types_path_sample)
        loss_lattice = self.lattice_probpath.get_loss(lattice_pred, lattice_path_sample)
        loss_frac_coords = self.frac_coords_probpath.get_loss(frac_coords_pred, frac_coords_path_sample)
        loss = self.cost["lattice"] * loss_lattice \
             + self.cost["frac_coords"] * loss_frac_coords \
             + self.cost["atom_types"] * loss_atom_types
        return {"loss": loss, "loss_lattice": loss_lattice, "loss_frac_coords": loss_frac_coords, "loss_atom_types": loss_atom_types}

    @torch.no_grad()
    def sample(self, batch, num_steps: int, return_intermediates=False):
        batch_size = batch.batch_size
        num_atoms = batch.num_atoms
        device = next(self.parameters()).device

        t_series = torch.linspace(0, 1, num_steps + 1, device=device)
        step_size = t_series[1] - t_series[0]
        t_per_atom = t_series[0].repeat_interleave(sum(num_atoms))

        if self.mode == "CSP":
            atom_types_placeholder = batch.atom_types
        elif self.mode == "DNG":
            atom_types_placeholder = torch.ones(sum(num_atoms), dtype=torch.long, device=device)
        atom_types_t = self.types_probpath.sample_union(atom_types_placeholder, t=t_per_atom).x_0

        lattice_t = self.lattice_probpath.random(batch_size, device=device)
        frac_coords_t = self.frac_coords_probpath.random(num_atoms, device=device)

        traj = [None] * (num_steps + 1)
        traj.clear()  # pre allocate and clear

        def detach_to_structure(num_atoms, atom_types, frac_coords, lattice):
            return {
                'num_atoms': num_atoms.to("cpu"),
                'atom_types': self.types_probpath.decode(atom_types).clone().detach().to("cpu"),
                'frac_coords': frac_coords.clone().detach().to("cpu"),
                'lattice': self.lattice_probpath.decode(lattice).clone().detach().to("cpu"),
            }

        for t in tqdm(t_series):
            if return_intermediates:
                traj.append(detach_to_structure(num_atoms, atom_types_t, frac_coords_t, lattice_t))

            times = t.repeat_interleave(batch_size)
            types_pred, lattice_pred, frac_coords_pred = self.vector_field(
                t=times,
                num_atoms=batch.num_atoms,
                atom_types=atom_types_t,
                lattice=lattice_t,
                frac_coords=frac_coords_t,
                node2graph=batch.batch,
            )

            atom_types_t = self.types_probpath.update(atom_types_t, types_pred, t, step_size)
            lattice_t = lattice_t + step_size * lattice_pred
            frac_coords_t = (frac_coords_t + step_size * frac_coords_pred) % 1

        # always store the last one
        traj.append(detach_to_structure(num_atoms, atom_types_t, frac_coords_t, lattice_t))

        # traj shape as list of dict of batched info
        # [{"k": stacked_tensor, ...}, ...]
        return traj
