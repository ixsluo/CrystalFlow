from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import Act


class Dense(torch.nn.Module):
    """
    Combines dense layer with activation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self._activation = Act(activation)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self._activation(x)
        return x
