import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat

from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch,
    get_pbc_distances,
    radius_graph_pbc,
    frac_to_cart_coords,
    repeat_blocks,
    get_reciprocal_lattice_torch,
    get_max_neighbors_mask,
)

MAX_ATOMIC_NUM = 100


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


class PeriodicNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, metrics, scaled_r):
        """
        Parameters
        ----------
        metrics: torch.Tensor
            Metrics matrix, symmetric. Shape of (E, 3, 3).
        scaled_r: torch.Tensor
            Vectors in fractional coordinates of the lattice cell. Shape of (E, 3).
        Results
        -------
        norm: torch.Tensor, shape (E,)
        """
        a = 1 - torch.cos(2 * math.pi * scaled_r)
        b = torch.sin(2 * math.pi * scaled_r)
        cos_term = torch.einsum("em,emn,en->e", a, metrics, a)
        sin_term = torch.einsum("em,emn,en->e", b, metrics, b)
        return (1 / (2 * math.pi)) * torch.sqrt(cos_term + sin_term)


class RecSinusoidsEmbedding(nn.Module):
    def __init__(self, n_millers=10):
        super().__init__()
        self.n_millers = n_millers
        self.millerindices = torch.cartesian_prod(
            torch.arange(self.n_millers),
            torch.arange(self.n_millers),
            torch.arange(self.n_millers),
        ).to(torch.get_default_dtype())
        self.dim = 2 * self.millerindices.shape[0]

    def forward(self, frac_diff, lattice):
        cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattice)  # (E,3)
        R = get_reciprocal_lattice_torch(lattice)
        hb = torch.einsum('mi,eij->emj', self.millerindices.to(R.device), R)  # (E, M, 3)
        hbX = torch.einsum("emj,ej->em", hb, cart_diff)  # (E, M)
        emb = torch.cat([hbX.cos(), hbX.sin()], dim=-1)  # (E, 2M)
        return emb


class CSPLayer(nn.Module):
    """Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        lattice_dim=9,
        act_fn=nn.SiLU(),
        dis_emb=None,
        rec_emb=None,
        na_emb=None,
        periodic_norm=None,
        ln=False,
        ip=True,
        use_angles=False,
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.rec_emb = rec_emb
        self.na_emb = na_emb
        self.periodic_norm = periodic_norm
        self.ip = ip
        self.use_angles = use_angles

        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if rec_emb is not None:
            self.dis_dim += rec_emb.dim
        if na_emb is not None:
            self.dis_dim += na_emb.embedding_dim
        if use_angles:
            self.dis_dim += 3
        if periodic_norm is not None:
            self.dis_dim += 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + lattice_dim + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), act_fn, nn.Linear(hidden_dim, hidden_dim), act_fn
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices_rep,
        edge_index,
        edge2graph,
        num_atoms,
        frac_diff=None,
        lattices_mat=None,
    ):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        inputs = [hi, hj]

        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0

        if self.rec_emb is not None:
            rec_diff = self.rec_emb(frac_diff, lattices_mat[edge2graph])
            inputs.append(rec_diff)

        if self.dis_emb is not None:
            frac_diff_emb = self.dis_emb(frac_diff)
            inputs.append(frac_diff_emb)

        if self.na_emb is not None:
            na_emb = self.na_emb(num_atoms.repeat_interleave(num_atoms, dim=0))[edge_index[0]]
            inputs.append(na_emb)

        if self.periodic_norm is not None:
            metrics = (lattices_mat.transpose(-1, -2) @ lattices_mat)[edge2graph]
            norm = self.periodic_norm(metrics, frac_diff)
            norm = norm.view(-1, 1)
            inputs.append(norm)

        if self.use_angles:  # angles between edges and lattices vectors
            cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattices_mat[edge2graph])  # (E,3)
            inner_dot = torch.einsum("ei,eji->ej", cart_diff, lattices_mat[edge2graph])  # (E,3)
            # norm = torch.linalg.norm(cart_diff, axis=1)[:, None] \
            #      * torch.linlag.norm(lattices_mat, dim=2)[edge2graph]  # (E,3)
            norm = torch.linalg.norm(inner_dot, axis=1)[:, None]  # (E,1)
            cos_angles = inner_dot / norm
            cos_angles = torch.where(cos_angles.isnan(), 0, cos_angles)
            inputs.append(cos_angles)

        if self.ip:
            lattice_ips = lattices_rep @ lattices_rep.transpose(-1, -2)
        else:
            lattice_ips = lattices_rep
        lattice_ips_flatten = torch.flatten(lattice_ips, start_dim=1)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        inputs.append(lattice_ips_flatten_edges)

        edges_input = torch.cat(inputs, dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[0], dim=0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self, node_features, frac_coords, lattices_rep, edge_index, edge2graph,
        num_atoms, frac_diff=None, lattices_mat=None
    ):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices_rep, edge_index, edge2graph, num_atoms, frac_diff, lattices_mat
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        lattice_dim=9,
        cemb_dim=1,
        num_layers=4,
        max_atoms=100,
        act_fn='silu',
        dis_emb='sin',  # fractional distance embedding
        num_freqs=10,
        rec_emb='none',  # reciprocal distance embedding
        num_millers=5,
        periodic_norm=False,
        na_emb=0,  # number of atoms embedding
        edge_style='fc',
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        use_angles=False,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
        type_encoding: None | nn.Module = None,
    ):
        super(CSPNet, self).__init__()

        self.ip = ip
        self.smooth = smooth
        self.type_encoding = type_encoding
        if self.type_encoding is None:
            if self.smooth:
                self.node_embedding = nn.Linear(MAX_ATOMIC_NUM, hidden_dim)
            else:
                self.node_embedding = nn.Embedding(MAX_ATOMIC_NUM, hidden_dim)
        else:
            self.node_embedding = nn.Linear(self.type_encoding.out_dim, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)  # no trainable params
        elif dis_emb == 'none':
            self.dis_emb = None
        else:
            raise ValueError(f"Unknown fractional distance embedding: {dis_emb=}")
        if rec_emb == "sin":
            self.rec_emb = RecSinusoidsEmbedding(n_millers=num_millers)  # no trainable params
        elif rec_emb == "none":
            self.rec_emb = None
        else:
            raise ValueError(f"Unknown reciprocal distance embedding: {rec_emb=}")
        if na_emb > 0:
            self.na_emb = nn.Embedding(max_atoms, na_emb)  # with trainable params, but same in each layer
        else:
            self.na_emb = None
        if periodic_norm:
            self.periodic_norm = PeriodicNorm()
        else:
            self.periodic_norm = None

        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i,
                CSPLayer(
                    hidden_dim=hidden_dim,
                    lattice_dim=lattice_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    rec_emb=self.rec_emb,
                    na_emb=self.na_emb,
                    periodic_norm=self.periodic_norm,
                    ln=ln,
                    ip=ip,
                    use_angles=use_angles,
                ),
            )
            self.add_module(
                "cemb_mixin_%d" % i,
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
            self.add_module(
                "cemb_adapter_%d" % i,
                nn.Sequential(
                    nn.Linear(cemb_dim, hidden_dim),
                    self.act_fn,
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act_fn,
                ),
            )

        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, lattice_dim, bias=False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        print(f"edge_style: {edge_style}")
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            type_out_dim = MAX_ATOMIC_NUM if self.type_encoding is None else self.type_encoding.out_dim
            self.type_out = nn.Linear(hidden_dim, type_out_dim)
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_vector_new = self.select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):

        if self.edge_style == 'fc':
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

        elif (self.edge_style == 'knn') or (self.edge_style == "knn_cart"):
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)

            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords,
                None,
                None,
                num_atoms,
                self.cutoff,
                self.max_neighbors,
                device=num_atoms.device,
                lattices=lattices,
            )

            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )

            return edge_index_new, -edge_vector_new

        elif (self.edge_style == "knn_frac"):
            # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
            num_atoms_per_image = num_atoms
            num_atoms_per_image_sqr = (num_atoms_per_image**2).long()
            # index offset between images
            index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

            index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
            num_atoms_per_image_expand = torch.repeat_interleave(num_atoms_per_image, num_atoms_per_image_sqr)

            num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
            index_sqr_offset = (torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr)
            index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
            atom_count_sqr = (torch.arange(num_atom_pairs, device=num_atoms.device) - index_sqr_offset)

            index1 = torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor") + index_offset_expand
            index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand

            frac_pos1 = torch.index_select(frac_coords, 0, index1)
            frac_pos2 = torch.index_select(frac_coords, 0, index2)
            frac_dist_mic_sqr = torch.sum(((frac_pos1 - frac_pos2 - 0.5) % 1 - 0.5) ** 2, dim=1).view(-1)

            mask_num_neighbors, num_bonds = get_max_neighbors_mask(
                natoms=num_atoms,
                index=index1,
                atom_distance=frac_dist_mic_sqr,
                max_num_neighbors_threshold=self.max_neighbors,
            )
            if not torch.all(mask_num_neighbors):
                # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
                index1 = torch.masked_select(index1, mask_num_neighbors)
                index2 = torch.masked_select(index2, mask_num_neighbors)

            edge_index = torch.stack((index2, index1))
            j_index, i_index = edge_index
            edge_vector = (frac_coords[j_index] - frac_coords[i_index]) % 1

            return edge_index, edge_vector

        else:
            raise ValueError(f"Unknown type of edge style: {self.edge_style}")

    def forward(self, t, atom_types, frac_coords, lattices_rep, num_atoms, node2graph, lattices_mat=None, cemb=None, guide_indicator=None):

        if lattices_mat is None:
            lattices_mat = lattices_rep
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices_mat, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            # may exist cemb
            if cemb is not None:
                cemb_mixin = self._modules["cemb_mixin_%d" % i]
                cemb_adapter = self._modules["cemb_adapter_%d" % i]
                cemb_bias = (cemb_mixin(cemb_adapter(cemb)) * guide_indicator[:, None]).repeat_interleave(num_atoms, dim=0)
                node_features = node_features + cemb_bias
            # csp layer
            csp_layer = self._modules["csp_layer_%d" % i]
            node_features = csp_layer(
                node_features,
                frac_coords,
                lattices_rep,
                edges,
                edge2graph,
                num_atoms=num_atoms,
                frac_diff=frac_diff,
                lattices_mat=lattices_mat,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(lattices_rep.shape)
        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices_rep)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out
