import torch
import torch.nn as nn


class SymmetrizeAnchor(nn.Module):
    def __init__(self):
        super().__init__()


class SymmetrizeRotavg(nn.Module):
    def __init__(self):
        super().__init__()

    def symmetrize_rank1(
        self,
        lattices: torch.Tensor,
        inv_lattices: torch.Tensor,
        forces: torch.Tensor,
        batch,
        num_atoms: torch.Tensor,
        general_ops: torch.Tensor,  # (Nop_sum,3,3)
        symm_map: list[list[list[int]]],  # (B,Nop,Nat)
        num_general_ops: torch.Tensor,  # number of operations for each ops
    ):
        lattices = lattices.repeat_interleave(num_atoms, dim=0)
        inv_lattices = inv_lattices.repeat_interleave(num_atoms, dim=0)
        scaled_symmetrized_forces_T = torch.zeros_like(forces.T)
        scaled_forces_T = torch.einsum('nji,nj->in', inv_lattices, forces)
        na_start = 0
        for na, nop, iops, isymm_map in zip(num_atoms, num_general_ops, general_ops.split(num_general_ops.tolist()), symm_map):
            for op, this_op_map in zip(iops, isymm_map):
                transformed_forces_T = torch.einsum('ij,jn->in', op[:3, :3], scaled_forces_T[:, na_start: na_start + na])
                scaled_symmetrized_forces_T[:, [m + na_start for m in this_op_map]] += transformed_forces_T
            scaled_symmetrized_forces_T[:, na_start: na_start + na] /= nop
            na_start += na
        symmetrized_forces = torch.einsum('nji,jn->ni', lattices, scaled_symmetrized_forces_T)
        return symmetrized_forces
