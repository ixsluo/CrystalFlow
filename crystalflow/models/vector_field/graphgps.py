import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GPSConv, GINEConv
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


class GraphGPS(nn.Module):
    def __init__(
        self,
        types_input_dim: tuple[str, int],
        types_output_dim: int | None,
        lattice_input_dim: int,
        lattice_output_dim: int,
        time_dim: int,
        frac_emb_freq: int,
        hidden_dim: int,
        num_layers: int,
        edge_style="fc",
        act_fn: str|nn.Module = "",
        final_layer_norm: bool = True,
    ):
        super().__init__()
        self.edge_style = edge_style
        if isinstance(act_fn, nn.Module):
            self.act_fn = act_fn
        elif act_fn.lower() == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"act_fn: {act_fn} not supported")

        if types_input_dim[0].lower() == "linear":
            self.types_smooth = nn.Linear(types_input_dim[1], hidden_dim)
            types_emb_dim = hidden_dim
        elif types_input_dim[0].lower() == "embedding":
            self.types_smooth = nn.Embedding(types_input_dim[1], hidden_dim)
            types_emb_dim = hidden_dim
        else:
            raise RuntimeError(f"Unsupported types input: {types_input_dim[0]} and dim: {types_emb_dim[1]}")
        self.node_align = nn.Linear(types_emb_dim + time_dim, hidden_dim)

        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.frac_embedding = SinusoidsEmbedding(frac_emb_freq)
        edge_dim = self.frac_embedding.dim + lattice_input_dim

        self.convs = nn.ModuleList(
            [
                GraphGPSLayer(channels=hidden_dim, edge_dim=edge_dim, act_fn=self.act_fn)
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(hidden_dim) if final_layer_norm else nn.Identity()

        if types_output_dim is not None:
            self.types_readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                self.act_fn,
                nn.Linear(hidden_dim // 2, types_output_dim),
            )
        else:
            self.types_readout = nn.Identity()
        self.lattice_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.act_fn,
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            self.act_fn,
            nn.Linear(hidden_dim // 4, lattice_output_dim, bias=False),
        )
        self.frac_coords_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.act_fn,
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            self.act_fn,
            nn.Linear(hidden_dim // 4, 3, bias=False),
        )

    def gen_edges(self, num_atoms, frac_coords, node2graph):
        if self.edge_style == 'fc':
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        else:
            raise NotImplementedError(f"edge_style: {self.edge_style} not supported")

    def build_edge_features(self, frac_coords, lattice, edge_index, edge2graph):
        ei, ej = edge_index[0], edge_index[1]
        frac_diff_emb = self.frac_embedding((frac_coords[ej] - frac_coords[ei]) % 1)
        edge_features = torch.cat([frac_diff_emb, lattice[edge2graph]], dim=1)
        return edge_features

    def build_node_features(self, t, atom_types, num_atoms):
        type_emb = self.types_smooth(atom_types)
        time_emb = self.time_embedding(t).repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([type_emb, time_emb], dim=1)
        node_features = self.node_align(node_features)
        return node_features

    def forward(self, t, num_atoms, atom_types, frac_coords, lattice, node2graph):
        # this atom_types is alread encoded
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords, node2graph)
        edge2graph = node2graph[edge_index[0]]
        node_features = self.build_node_features(t, atom_types, num_atoms)
        edge_features = self.build_edge_features(frac_coords, lattice, edge_index, edge2graph)
        for conv in self.convs:
            node_features = conv(node_features, edge_index, edge_features, node2graph)
        node_features = self.final_layer_norm(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')

        types_pred = self.types_readout(node_features)
        lattice_pred = self.lattice_readout(graph_features)
        frac_coords_pred = self.frac_coords_readout(node_features)
        return types_pred, lattice_pred, frac_coords_pred


class GraphGPSLayer(nn.Module):
    def __init__(self, channels, edge_dim, act_fn: nn.Module, attn_kwargs=None):
        super().__init__()
        nn_model = nn.Sequential(
            nn.Linear(channels, channels),
            act_fn,
            nn.Linear(channels, channels),
        )
        self.conv = GPSConv(
            channels,
            GINEConv(nn=nn_model, edge_dim=edge_dim),
            heads=4,
            attn_type="multihead",
            attn_kwargs=attn_kwargs,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.conv(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
