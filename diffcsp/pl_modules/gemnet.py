import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from diffcsp.common.gemnet.gemnet import GemNetT


class GemNetTWrapper(nn.Module):
    def __init__(
        self,
        num_targets: int,
        latent_dim: int,
        atom_embedding: torch.nn.Module | dict,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        regress_stress: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        max_cell_images_per_dim: int = 5,
        encoder_mode: bool = False,  #
        pred_type = False,
        type_encoding = None | nn.Module,
        **kwargs,
    ):
        super().__init__()
        otf_graph = True
        if not isinstance(atom_embedding, nn.Module):
            import hydra
            atom_embedding = hydra.utils.instantiate(atom_embedding)
        self.gemnet = GemNetT(
            num_targets,
            latent_dim,
            atom_embedding,
            num_spherical,
            num_radial,
            num_blocks,
            emb_size_atom,
            emb_size_edge,
            emb_size_trip,
            emb_size_rbf,
            emb_size_cbf,
            emb_size_bil_trip,
            num_before_skip,
            num_after_skip,
            num_concat,
            num_atom,
            regress_stress,
            cutoff,
            max_neighbors,
            rbf,
            envelope,
            cbf,
            otf_graph,
            output_init,
            activation,
            max_cell_images_per_dim,
            encoder_mode,  #
            pred_type,
            type_encoding,
            **kwargs,
        )
        pass

    def forward(self, t, atom_types, frac_coords, lattices_rep, num_atoms, node2graph, lattices_mat=None, cemb=None, guide_indicator=None):
        return self.gemnet(
            z=t,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=node2graph,
            lattice=lattices_mat,
        )
