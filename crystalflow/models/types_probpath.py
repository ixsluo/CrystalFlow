from abc import ABC, abstractmethod
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import ProbPath, MixtureDiscreteProbPath, AffineProbPath
from flow_matching.path.path_sample import PathSample, DiscretePathSample
from flow_matching.path.scheduler import Scheduler
from flow_matching.utils import expand_tensor_like

from crystalflow.common.data_utils import MAX_ATOMIC_NUMBER
from crystalflow.models.type_model import ReorderedTable, EmbeddingMinus1


class ProbPathPartial(Protocol):
    def __call__(self, scheduler: Scheduler, *args, **kwargs) -> ProbPath:
        raise NotImplemented


class AtomTypesProbPath(ProbPath):
    pass


class IdentityFakePathModel(nn.Module):
    """Always use the same x_0 and x_1

    Minus 1 in input, add 1 in decode
    """
    def __init__(self, scheduler: Scheduler):
        super().__init__()
        self.output_dim = ("embedding", MAX_ATOMIC_NUMBER)  # CSP
        self.decode_dim = None  # CSP
        self.discrete = True

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        return PathSample(x_0=x_1, x_1=x_1, x_t=x_1, t=t, dx_t=torch.zeros_like(x_1))

    def sample_union(self, atom_types, t: torch.Tensor) -> PathSample:
        x_1 = atom_types - 1
        return self.sample(x_0=x_1, x_1=x_1, t=t)

    def decode(self, x):
        return x + 1

    def encode_random_like(self, x: torch.Tensor):
        return x.clone().detach()

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError("You should NOT use this")

#! todo
class DiscreteEmbeddingPathUnion(nn.Module, ProbPath):
    """Discrete prob path and followed by embedding

    Are categories in prob path, thus decodeable.
    """
    def __init__(
        self,
        scheduler: Scheduler,
        probpath: ProbPathPartial,
        source_distr: Literal["mask", "uniform"],
    ):
        super().__init__()
        self.probpath: MixtureDiscreteProbPath = probpath(scheduler)
        self.mode = "DNG"
        self.discrete = True

        if source_distr == "mask":
            self.output_dim = ("embedding", MAX_ATOMIC_NUMBER + 1)
            self.decode_dim = MAX_ATOMIC_NUMBER + 1
        elif source_distr == "uniform":
            self.output_dim = ("embedding", MAX_ATOMIC_NUMBER)
            self.decode_dim = MAX_ATOMIC_NUMBER
        else:
            raise NotImplementedError(f"Soruce distribution {source_distr} not supported")
        self.source_distr = source_distr

        self.loss_fn = MixturePathGeneralizedKL(self.probpath, reduction="mean")


    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> DiscretePathSample:
        # x_0, x_1, x_t are all int categories
        path_sample = self.probpath.sample(t=t, x_0=x_0, x_1=x_1)
        return path_sample

    def sample_union(self, atom_types, t: torch.Tensor):
        x_1 = atom_types
        x_0 = self.encode_random_like(x_1)
        if self.source_distr == "uniform":
            x_1 = x_1 - 1
            x_0 = x_0 - 1
        path_sample = self.sample(x_0=x_0, x_1=x_1, t=t)
        return path_sample

    def decode(self, x: torch.Tensor):
        if self.source_distr == "mask":
            return x
        elif self.source_distr == "uniform":
            return x + 1
        else:
            raise NotImplementedError()

    def encode_random_like(self, x: torch.Tensor):
        if self.source_distr == "mask":
            return torch.zeros_like(x, dtype=x.dtype)  # int 0 is masked elements
        elif self.source_distr == "uniform":
            return torch.randint_like(x, low=1, high=MAX_ATOMIC_NUMBER + 1, dtype=x.dtype)
        else:
            raise RuntimeError(f"Error source distribution: {self.source_distr}")

    def get_loss(self, logits: torch.Tensor, path_sample: DiscretePathSample):
        return self.loss_fn(logits=logits, x_1=path_sample.x_1, x_t=path_sample.x_t, t=path_sample.t)

    def update(self, x_t: torch.Tensor, pred_t: torch.Tensor, t: torch.Tensor, step_size: float):
        h = step_size
        pred_prob = F.softmax(pred_t, dim=-1)
        x_1 = torch.multinomial(pred_prob, 1, replacement=True).view(*pred_prob.shape[:-1])
        scheduler_output = self.probpath.scheduler(t)
        k_t = scheduler_output.alpha_t
        d_k_t = scheduler_output.d_alpha_t

        delta_1 = F.one_hot(x_1, num_classes=self.decode_dim).to(k_t.dtype)
        u = d_k_t / (1 - k_t) * delta_1

        # Set u_t(x_t|x_t,x_1) = 0
        delta_t = F.one_hot(x_t, num_classes=self.decode_dim)
        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

        # Sample x_t ~ u_t( \cdot |x_t,x_1)
        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
        mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(-h * intensity)

        if mask_jump.sum() > 0:
            mask_prob = u[mask_jump].to(dtype=torch.get_default_dtype())
            x_t[mask_jump] = torch.multinomial(mask_prob, 1, replacement=True).view(*mask_prob.shape[:-1])

        return x_t


class TypeTablePathModel(nn.Module, AtomTypesProbPath):
    def __init__(
        self,
        scheduler: Scheduler,
    ):
        super().__init__()
        self.probpath = AffineProbPath(scheduler)
        self.model = ReorderedTable()

    def forward(self, atom_types: torch.Tensor):
        return self.model.forward(atom_types)

    def encode_types(self, atom_types: torch.Tensor) -> torch.Tensor:
        """atomic number to concat-one-hot table coding"""
        return self.model.encode_types(atom_types)

    def random_x0_like_x1(self, x1: torch.Tensor) -> torch.Tensor:
        """random sampling x0 shaped like x1 from normal distribution"""
        return torch.randn_like(x1)

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        path_sample = self.probpath.sample(t=t, x_0=x_0, x_1=x_1)
        return path_sample

    def sample_from_types(self, atom_types: torch.Tensor, t: torch.Tensor) -> PathSample:
        x_1 = self.encode_types(atom_types).type_as(t)
        x_0 = self.random_x0_like_x1(x_1)
        return self.sample(x_0=x_0, x_1=x_1, t=t)

    def decode_types(self, encoded_types: torch.Tensor):
        assert encoded_types.shape[1] == self.num_rowcol, f"TypeTable must decode shape must be (*,{self.num_rowcol}), but got {encoded_types.shape}"
        return self.model.decode_types(encoded_types)

    def get_loss(self, pred, path_sample: PathSample):
        return F.mse_loss(pred, path_sample.dx_t)
