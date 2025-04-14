import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import TransformerConv, LayerNorm
from torch_scatter import scatter

from crystalflow.common.data_utils import MAX_ATOMIC_NUMBER


class EmbeddingMinus1(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb_model = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.emb_model(x - 1)


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


class CrysFormer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_freqs: int,
        hidden_dim: int,
        num_layers: int,
        edge_style="fc",
        act_fn: str|nn.Module = "",
        final_layer_norm: bool = True,
        type_encoding=None | nn.Module,
        cemb_dim=1,
        pred_type=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        lattice_input_dim = 6
        lattice_output_dim = 6
        time_dim = latent_dim
        self.pred_type = pred_type

        self.edge_style = edge_style
        if isinstance(act_fn, nn.Module):
            self.act_fn = act_fn
        elif act_fn.lower() == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"act_fn: {act_fn} not supported")

        if type_encoding is None:
            self.types_smooth = EmbeddingMinus1(MAX_ATOMIC_NUMBER, hidden_dim)
            types_output_dim = None
        else:
            self.types_smooth = nn.Sequential(
                type_encoding,
                nn.Linear(type_encoding.out_dim, hidden_dim),
            )
            types_output_dim = type_encoding.out_dim
        self.node_align = nn.Linear(hidden_dim + time_dim, hidden_dim)

        # self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.time_embedding = nn.Identity()
        self.frac_embedding = SinusoidsEmbedding(num_freqs)
        edge_dim = self.frac_embedding.dim + lattice_input_dim
        self.edge_ln = nn.LayerNorm(edge_dim)

        self.convs = nn.ModuleList(
            [
                CrysFormerLayer(
                    channels=hidden_dim,
                    edge_dim=edge_dim,
                )
                for _ in range(num_layers)
            ]
        )

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
        edge_features = self.edge_ln(edge_features)
        return edge_features

    def build_node_features(self, t, atom_types, num_atoms):
        type_emb = self.types_smooth(atom_types)
        time_emb = self.time_embedding(t).repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([type_emb, time_emb], dim=1)
        node_features = self.node_align(node_features)
        return node_features

    def forward(self, t, atom_types, frac_coords, lattices_rep, num_atoms, node2graph, lattices_mat, cemb, guide_indicator):
        lattice = lattices_rep
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords, node2graph)
        edge2graph = node2graph[edge_index[0]]
        node_features = self.build_node_features(t, atom_types, num_atoms)
        edge_features = self.build_edge_features(frac_coords, lattice, edge_index, edge2graph)
        for conv in self.convs:
            node_features = conv(node_features, edge_index, edge_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')

        types_pred = self.types_readout(node_features)
        lattice_pred = self.lattice_readout(graph_features)
        frac_coords_pred = self.frac_coords_readout(node_features)

        if self.pred_type:
            return types_pred, lattice_pred, frac_coords_pred
        else:
            return lattice_pred, frac_coords_pred


class CrysFormerLayer(nn.Module):
    def __init__(self, channels, edge_dim):
        super().__init__()
        self.ln_pre_attn = LayerNorm(channels)
        self.conv = TransformerConv(
            channels,
            channels,
            heads=8,
            concat=False,
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
