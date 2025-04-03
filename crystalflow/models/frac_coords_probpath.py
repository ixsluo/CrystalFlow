from abc import ABC, abstractmethod
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import ProbPath
from flow_matching.path.path_sample import PathSample, DiscretePathSample
from flow_matching.path.scheduler import Scheduler
from flow_matching.utils import expand_tensor_like


class FracCoordsProbPath(Protocol):
    output_dim: int
    decode_dim: int

    def sample_union(self, batch) -> PathSample:
        raise NotImplemented

    def sample(self, batch_size, *, device) -> torch.Tensor:
        raise NotImplemented

    def get_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented


class ProbPathPartial(Protocol):
    def __call__(self, scheduler: Scheduler, *args, **kwargs) -> ProbPath:
        raise NotImplemented


class WrapGaussianPathUnion(ProbPath):
    def __init__(
        self,
        scheduler: Scheduler,
    ):
        super().__init__()
        self.scheduler = scheduler

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        dx_t = (x_1 - x_0 - 0.5) % 1 - 0.5
        x_t = (x_0 + dx_t * expand_tensor_like(self.scheduler(t).alpha_t, x_0)) % 1
        return PathSample(
            x_1=x_1,
            x_0=x_0,
            x_t=x_t,
            dx_t=dx_t,
            t=t,
        )

    def sample_union(self, batch, t: torch.Tensor):
        f_1 = batch.frac_coords
        f_0 = self.random_like(f_1)
        path_sample = self.sample(x_0=f_0, x_1=f_1, t=t)
        return path_sample

    def random(self, num_atoms, device=None):
        f_0 = torch.rand((sum(num_atoms), 3), device=device)
        return f_0

    def random_like(self, f_1):
        f_0 = torch.rand_like(f_1)
        return f_0

    def get_loss(self, x_pred, path_sample: PathSample):
        return F.mse_loss(x_pred, path_sample.dx_t)
