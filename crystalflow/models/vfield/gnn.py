import math
from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleGNN(nn.Module):
    def __init__(
        self,
        mode: Literal['CSP', 'DNG'],
        type_dim: int,
        type_need_smooth: bool,
        time_dim: int,
        hidden_dim: int,
        num_layers: int,
        frac_emb_freq: int,
        edge_style="fc",
        act_fn: str|nn.Module = "",
        final_layer_norm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.edge_style = edge_style
        if isinstance(act_fn, nn.Module):
            self.act_fn = act_fn
        elif act_fn.lower() == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"act_fn: {act_fn} not supported")

        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        if type_need_smooth:
            self.type_smooth = nn.Linear(type_dim, hidden_dim)
        else:
            self.type_smooth = nn.Identity()

        self.node_embedding = nn.Linear(hidden_dim + time_dim, hidden_dim)
        self.frac_embedding = SinusoidsEmbedding(frac_emb_freq)
        edge_dim = self.frac_embedding.dim + 6
        self.gnn_layers = nn.ModuleList([
            SimpleGNNLayer(edge_dim=edge_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim) if final_layer_norm else nn.Identity()

        if self.mode == "DNG":
            self.type_readout = nn.Linear(hidden_dim, type_dim)
        else:
            self.type_readout = nn.Identity()
        self.l_polar_readout = nn.Linear(hidden_dim, 6, bias=False)
        self.frac_coords_readout = nn.Linear(hidden_dim, 3, bias=False)

    def gen_edges(self, num_atoms, frac_coords, node2graph):
        if self.edge_style == 'fc':
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        else:
            raise NotImplementedError(f"edge_style: {self.edge_style} not supported")

    def build_edge_features(self, frac_coords, l_polar, edge_index, edge2graph):
        ei, ej = edge_index[0], edge_index[1]
        frac_diff_emb = self.frac_embedding((frac_coords[ej] - frac_coords[ei]) % 1)
        edge_features = torch.cat([frac_diff_emb, l_polar[edge2graph]], dim=1)
        return edge_features

    def build_node_features(self, t, atom_types, num_atoms):
        type_emb = self.type_smooth(atom_types)
        time_emb = self.time_embedding(t).repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([type_emb, time_emb], dim=1)
        node_features = self.node_embedding(node_features)
        return node_features

    def forward(self, t, num_atoms, atom_types, frac_coords, l_polar, node2graph):
        # this atom_types is alread encoded
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords, node2graph)
        edge2graph = node2graph[edge_index[0]]
        node_features = self.build_node_features(t, atom_types, num_atoms)
        edge_features = self.build_edge_features(frac_coords, l_polar, edge_index, edge2graph)
        for layer in self.gnn_layers:
            node_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
                edge2graph=edge2graph,
            )
        node_features = self.final_layer_norm(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')

        type_pred = self.type_readout(node_features)
        l_polar_pred = self.l_polar_readout(graph_features)
        frac_coords_pred = self.frac_coords_readout(node_features)
        return type_pred, l_polar_pred, frac_coords_pred


class SimpleGNNLayer(nn.Module):
    def __init__(
        self,
        edge_dim,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        layer_norm: bool = True,
    ):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        self.agg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        self.layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(self, node_features, edge_features, edge_index, edge2graph):
        message = self.message_model(node_features, edge_features, edge_index)
        message = scatter(message, edge_index[0], dim=0, reduce="mean", dim_size=node_features.shape[0])
        node_features = node_features + self.agg_model(node_features, message)
        return node_features

    def message_model(self, node_features, edge_features, edge_index):
        node_features = self.layer_norm(node_features)
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        mij = self.message_mlp(torch.cat([hi, hj, edge_features], dim=1))
        return mij

    def agg_model(self, node_features, message):
        agg = self.agg_mlp(torch.cat([node_features, message], dim=1))
        return agg
