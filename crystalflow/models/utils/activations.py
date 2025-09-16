from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Act(nn.Module):
    def __init__(self, act: str | None, slope: float = 0.05) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

        if act == "relu":
            self._activation = nn.ReLU()
        elif act == "leaky_relu":
            self._activation = nn.LeakyReLU(slope)
        elif act == "sp":
            self._activation = nn.Softplus(beta=1)
        elif act == "leaky_sp":
            self._activation = LeakySoftplus(beta=1, slope=slope)
        elif act == "elu":
            self._activation = nn.ELU(alpha=1)
        elif act == "leaky_elu":
            self._activation = LeakyELU(alpha=1, slope=slope)
        elif act == "ssp":
            self._activation = ShiftSoftplus(beta=1)
        elif act == "leaky_ssp":
            self._activation = LeakyShiftSoftplus(beta=1, slope=slope)
        elif act == "tanh":
            self._activation = nn.Tanh()
        elif act == "leaky_tanh":
            self._activation = LeakyTanh(slope=slope)
        elif act == "swish":
            self._activation = ScaledSiLU()
        elif act == "silu":
            self._activation = ScaledSiLU()
        elif act == "siqu":
            self._activation = SiQU()
        elif act == '' or self.act is None:
            self._activation = nn.Identity()
        else:
            raise RuntimeError(f"Undefined activation called {act}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._activation(input)


class LeakySoftplus(nn.Module):
    def __init__(self, beta=1, slope=0.05):
        super().__init__()
        self.beta = beta
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta) - self.slope * F.relu(-x)


class LeakyELU(nn.Module):
    def __init__(self, alpha=1, slope=0.05):
        super().__init__()
        self.alpha = alpha
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha) - self.slope * F.relu(-x)


class ShiftSoftplus(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta) - self.shift


class LeakyShiftSoftplus(nn.Module):
    def __init__(self, beta=1, slope=0.05):
        super().__init__()
        self.beta = beta
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta) - self.slope * F.relu(-x) - self.shift


class LeakyTanh(nn.Module):
    def __init__(self, slope=0.05):
        super().__init__()
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) + self.slope * x


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6

    def forward(self, x: torch.Tensor):
        return F.sigmoid(x) * x * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * F.silu(x)
