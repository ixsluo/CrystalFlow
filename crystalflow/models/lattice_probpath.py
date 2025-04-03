from abc import ABC, abstractmethod
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import ProbPath
from flow_matching.path.path_sample import PathSample, DiscretePathSample
from flow_matching.path.scheduler import Scheduler

from crystalflow.common.data_utils import lattice_polar_build_torch


class LatticeProbPath(Protocol):
    output_dim: int
    decode_dim: int

    def sample_union(self, batch) -> PathSample:
        raise NotImplemented

    def sample(self, num_atoms, *, device) -> torch.Tensor:
        raise NotImplemented

    def get_loss(self, pred, path_sample: PathSample, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented


class ProbPathPartial(Protocol):
    def __call__(self, scheduler: Scheduler, *args, **kwargs) -> ProbPath:
        raise NotImplemented


class LatticePolarPathUnion(nn.Module, ProbPath):
    def __init__(
        self,
        scheduler: Scheduler,
        probpath: ProbPathPartial,
        sigma: float = 0.1,
    ):
        super().__init__()
        self.probpath: ProbPath = probpath(scheduler)
        self.sigma = sigma
        self.output_dim = 6
        self.decode_dim = 6

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        path_sample = self.probpath.sample(t=t, x_0=x_0, x_1=x_1)
        return path_sample

    def sample_union(self, batch, t: torch.Tensor):
        l_1 = batch.l_polar
        l_0 = self.random_like(l_1)
        path_sample = self.sample(x_0=l_0, x_1=l_1, t=t)
        return path_sample

    def random_like(self, l_1):
        l_0 = torch.randn_like(l_1) * self.sigma
        l_0[:, -1] = l_0[:, -1] + 1
        return l_0

    def random(self, batch_size, device=None):
        l_0 = torch.randn((batch_size, 6), device=device) * self.sigma
        l_0[:, -1] = l_0[:, -1] + 1
        return l_0

    def get_loss(self, x_pred, path_sample: PathSample):
        return F.mse_loss(x_pred, path_sample.dx_t)

    def decode(self, x: torch.Tensor):
        return lattice_polar_build_torch(x)
