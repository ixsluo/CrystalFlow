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


class EmbeddingMinus1(nn.Module):
    def __init__(self, embedding_dim):
        """Embedding minus 1

        Arguments
        ---------
            dim: int
                "nn.Embedding" out dim.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = nn.Embedding(MAX_ATOMIC_NUMBER, embedding_dim)
        self.decodeable = False

    def forward(self, x: torch.Tensor):
        return self.model(x - 1)

    def random_encoding_from(self, atom_types: torch.Tensor):
        return self(atom_types).clone().detach()

    # remove this later
    def encode(self, x):
        return self(x)

    decode = None  # this method should not be used


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