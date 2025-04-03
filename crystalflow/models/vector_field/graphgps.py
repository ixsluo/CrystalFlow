import math

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


class GPS(nn.Module):
    def __init__(
        self,
        types_input_dim: tuple[str, int],
        types_output_dim: int | None,
        lattice_input_dim: int,
        lattice_output_dim: int,
        time_dim,
        frac_emb_freq,
        hidden_dim,
    ):
        super().__init__()

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

        self.types_readout = nn.Linear(hidden_dim, types_output_dim) if types_output_dim is not None else nn.Identity()
        self.lattice_readout = nn.Linear(hidden_dim, lattice_output_dim, bias=False)
        self.frac_coords_readout = nn.Linear(hidden_dim, 3, bias=False)

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

    def forward(self, t, num_atoms, atom_types, frac_coords, lattice, node2graph):
        pass