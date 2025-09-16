import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import TransformerConv, LayerNorm
from torch_scatter import scatter

from crystalflow.common.data_utils import MAX_ATOMIC_NUMBER


class CrysFormer(nn.Modele):
    def __init__(self, ):
        super().__init__()


class CrysFormerLayer(nn.Module):
    def __init__(self, channels, edge_dim):
        super().__init__()
        self.ln_pre_attn = LayerNorm(channels)
        self.conv = TransformerConv(
            channels,
            channels,
            heads=8,
            dropout=0.0,
            edge_dim=edge_dim,
            bias=True,
        )
        self.gate_attn = GatedResidual(channels)
        self.ln_pre_ff = LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.gate_ff = GatedResidual(channels)

    def forward(self, h, edge_index, edge_attr):
        h = self.ln_pre_attn(h)
        v = self.conv(h, edge_index, edge_attr)
        h = self.gate_attn(h, v)

        h = self.ln_pre_ff(h)
        v = self.feedforward(h)
        h = self.gate_ff(h, v)
        return h


class GatedResidual(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 3, channels * 3 // 2),
            nn.SiLU(),
            nn.Linear(channels * 3 // 2, channels * 3 // 4),
            nn.SiLU(),
            nn.Linear(channels * 3 // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, u, v):
        alpha = self.mlp(torch.concat([u, v, u - v], dim=-1))
        return alpha * u + (1 - alpha) * v
