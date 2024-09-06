import torch
import torch.nn as nn
from torch_scatter import scatter


class SymmetrizeAnchor(nn.Module):
    def __init__(self):
        super().__init__()


class SymmetrizeRotavg(nn.Module):
    def __init__(self):
        super().__init__()

    def symmetrize_rank1_scaled(
        self,
        scaled_forces: torch.Tensor,      # (Nat, 3)
        num_atoms: torch.Tensor,
        general_ops: torch.Tensor,        # (B, 192, 4, 4)
        symm_map,                         # (Nat, 192)
        num_general_ops: torch.Tensor,    # (B,)
    ):
        na = scaled_forces.shape[0]
        general_ops = general_ops[:, :, :3, :3].repeat_interleave(num_atoms, dim=0)
        transformed_forces = torch.einsum('nmij,nj->nmi', general_ops, scaled_forces)
        scaled_symmetrized_forces = scatter(transformed_forces, symm_map, dim=1, dim_size=na)  # (Nat, Nat, 3)
        scaled_symmetrized_forces = scaled_symmetrized_forces.sum(dim=0)  # sum over each target atom index
        scaled_symmetrized_forces /= num_general_ops.repeat_interleave(num_atoms, dim=0)[:, None]
        return scaled_symmetrized_forces  # 42ms

    def symmetrize_rank1(
        self,
        lattices: torch.Tensor,
        inv_lattices: torch.Tensor,
        forces: torch.Tensor,
        num_atoms: torch.Tensor,
        general_ops: torch.Tensor,  # (Nop_sum,3,3)
        symm_map: list[list[list[int]]],  # (B,Nop,Nat)
        num_general_ops: torch.Tensor,  # number of operations for each ops
    ):
        lattices = lattices.repeat_interleave(num_atoms, dim=0)
        inv_lattices = inv_lattices.repeat_interleave(num_atoms, dim=0)
        scaled_forces = torch.einsum('nji,nj->ni', inv_lattices, forces)
        scaled_symmetrized_forces = self.symmetrize_rank1_scaled(
            scaled_forces, num_atoms, general_ops, symm_map, num_general_ops
        )
        symmetrized_forces = torch.einsum('nji,nj->ni', lattices, scaled_symmetrized_forces)
        return symmetrized_forces
