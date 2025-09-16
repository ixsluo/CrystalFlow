import math
from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter

from crystalflow.models.utils.activations import Act
from crystalflow.models.utils.basis import SinusoidalTimeEmbeddings, SinusoidsEmbedding


class EGNN(nn.Module):
    def __init__(
        self,
        latent_dim: int | None,  # z
        time_dim: int,
        atom_embedding: torch.nn.Module,  # encoding atom types
        node_dim: int,
        edge_dim: int,
        num_layers: int,
        edge_style: str,
        frac_emb_freq: int,
        activation: str,
        final_layer_norm: bool = True,
        type_out_dim = 0,
        lattice_out_dim = 6,
        coord_out_dim = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.atom_embedding = atom_embedding
        if hasattr(atom_embedding, "out_features"):
            type_dim = getattr(self.atom_embedding, "out_features")
        elif hasattr(atom_embedding, "embedding_dim"):
            type_dim = getattr(self.atom_embedding, "embedding_dim")
        elif hasattr(atom_embedding, "out_dim"):
            type_dim = getattr(self.atom_embedding, "out_dim")
        else:
            raise ValueError("Not knowing type dim from atom_embedding module.")
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.edge_style = edge_style
        self.activation = Act(activation) if isinstance(activation, str) else activation

        self.time_embedding = SinusoidalTimeEmbeddings(dim=time_dim)
        self.frac_embedding = SinusoidsEmbedding(frac_emb_freq)
        self.node_align = nn.Linear(type_dim + time_dim, node_dim)

        self.gnn_layers = nn.ModuleList(
            [
                ENGNLayer(edge_dim=self.edge_dim, node_dim=self.node_dim, activation=self.activation)
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(node_dim) if final_layer_norm else nn.Identity()

        self.type_out = nn.Linear(node_dim, type_out_dim)
        self.lattice_out = nn.Linear(node_dim, lattice_out_dim, bias=False)
        self.coord_out = nn.Linear(node_dim, coord_out_dim, bias=False)

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
        type_emb = self.atom_embedding(atom_types)
        time_emb = self.time_embedding(t).repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([type_emb, time_emb], dim=1)
        node_features = self.node_align(node_features)
        return node_features

    def forward(self, t, num_atoms, atom_types, frac_coords, lattice, node2graph):
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords, node2graph)
        edge2graph = node2graph[edge_index[0]]
        node_features = self.build_node_features(t, atom_types, num_atoms)
        edge_features = self.build_edge_features(frac_coords, lattice, edge_index, edge2graph)
        for layer in self.gnn_layers:
            node_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
                edge2graph=edge2graph,
            )
        node_features = self.final_layer_norm(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')

        type_pred = self.type_out(node_features)
        lattice_pred = self.lattice_out(graph_features)
        coord_pred = self.coord_out(node_features)

        return {
            "type_pred": type_pred,
            "lattice_pred": lattice_pred,
            "coord_pred": coord_pred,
        }


class ENGNLayer(nn.Module):
    def __init__(
        self,
        edge_dim,
        node_dim=128,
        activation: str | nn.Module = 'silu',
        layer_norm: bool = True,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = node_dim
        self.activation = Act(activation) if isinstance(activation, str) else activation

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            self.activation,
            nn.Linear(node_dim, node_dim),
            self.activation,
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            self.activation,
            nn.Linear(node_dim, edge_dim),
            self.activation,
        )

    def forward(self, node_features, edge_features, edge_index, edge2graph):
        node_features_in = node_features
        node_features = self.layer_norm(node_features)
        message = self.message_model(node_features, edge_features, edge_index)
        message = scatter(message, edge_index[0], dim=0, reduce="mean", dim_size=node_features.shape[0])
        node_features_out = node_features_in + self.agg_model(node_features, message)
        return node_features_out

    def message_model(self, node_features, edge_features, edge_index):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        mij = self.message_mlp(torch.cat([hi, hj, edge_features], dim=1))
        return mij

    def aggregation_model(self, node_features, message):
        agg = torch.cat([node_features, message], dim=1)
        agg = self.aggregation_mlp(agg)
        return agg