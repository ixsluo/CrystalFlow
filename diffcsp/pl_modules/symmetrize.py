import torch
import torch.nn as nn


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
        general_ops: torch.Tensor,        # (192, 4, 4)
        symm_map,                         # (Nat, 192)
        num_general_ops: torch.Tensor,    # (B,)
    ):
        scaled_symmetrized_forces = torch.zeros_like(scaled_forces)
        for iop, op in enumerate(general_ops):
            transformed_forces = torch.einsum('ij,nj->ni', op[:3, :3], scaled_forces)
            scaled_symmetrized_forces[symm_map[:, iop], :] += transformed_forces
        scaled_symmetrized_forces /= num_general_ops.repeat_interleave(num_atoms, dim=0)[:, None]
        return scaled_symmetrized_forces

        na_start = 0
        for na, nop, iops, isymm_map in zip(num_atoms, num_general_ops, general_ops.split(num_general_ops.tolist()), symm_map):
            for op, this_op_map in zip(iops, isymm_map):
                transformed_forces = torch.einsum('ij,nj->ni', op[:3, :3], scaled_forces[na_start: na_start + na])
                scaled_symmetrized_forces[[m + na_start for m in this_op_map], :] += transformed_forces
            scaled_symmetrized_forces[na_start: na_start + na] /= nop
            na_start += na
        return scaled_symmetrized_forces

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
