import math

import torch
import torch.nn as nn

from diffcsp.common.data_utils import lattice_polar_build_torch, lattice_polar_decompose_torch


class LatticeDecompNN(nn.Module):
    def __init__(self):
        super().__init__()
        basis = self.get_basis()
        masks, biass = self.get_spacegroup_constraints()
        family = self.get_family_idx()
        self.register_buffer("basis", basis)
        self.register_buffer('masks', masks)
        self.register_buffer('biass', biass)
        self.register_buffer('family', family)

    @torch.no_grad()
    def build(self, k):
        return lattice_polar_build_torch(k)

    @torch.no_grad()
    def decompose(self, lattices):
        return lattice_polar_decompose_torch(lattices)

    def get_basis(self):
        basis = torch.FloatTensor([
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ])
        # # Normalize
        # basis = basis / basis.norm(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
        return basis

    def get_spacegroup_constraint(self, spacegroup):
        mask = torch.ones(6)
        bias = torch.zeros(6)
        if 195 <= spacegroup <= 230:
            pos = [0,1,2,3,4]
            mask[pos] = 0.
        elif 143 <= spacegroup <= 194:
            pos = [0,1,2,3]
            mask[pos] = 0.
            # bias[0] = -0.25 * np.log(3) * np.sqrt(2)
            bias[0] = - math.log(3) / 4
        elif 75 <= spacegroup <= 142:
            pos = [0,1,2,3]
            mask[pos] = 0.
        elif 16 <= spacegroup <= 74:
            pos = [0,1,2]
            mask[pos] = 0.
        elif 3 <= spacegroup <= 15:
            pos = [0,2]
            mask[pos] = 0.
        elif 0 <= spacegroup <= 2:
            pass
        else:
            raise ValueError("Invalid spacegroup.")
        return mask, bias

    def get_spacegroup_constraints(self):
        masks, biass = [], []
        for i in range(231):
            mask, bias = self.get_spacegroup_constraint(i)
            masks.append(mask.unsqueeze(0))
            biass.append(bias.unsqueeze(0))
        return torch.cat(masks, dim = 0), torch.cat(biass, dim = 0)

    def get_family_idx(self):
        family = []
        for spacegroup in range(231):
            if 195 <= spacegroup <= 230:
                family.append(6)
            elif 143 <= spacegroup <= 194:
                family.append(5)
            elif 75 <= spacegroup <= 142:
                family.append(4)
            elif 16 <= spacegroup <= 74:
                family.append(3)
            elif 3 <= spacegroup <= 15:
                family.append(2)
            elif 0 <= spacegroup <= 2:
                family.append(1)
            else:
                raise ValueError("Invalid spacegroup.")
        return torch.LongTensor(family)

    def proj_k_to_spacegroup(self, vec, spacegroup):
        batch_size, dims = vec.shape
        if dims == 6:
            masks = self.masks[spacegroup, :] # B * 6
            biass = self.biass[spacegroup, :] # B * 6  
        elif dims == 5:
            # - volume
            masks = self.masks[spacegroup, :-1] # B * 5
            biass = self.biass[spacegroup, :-1] # B * 5
        return vec * masks + biass

    def proj_kdiff_to_spacegroup(self, vec, spacegroup):
        batch_size, dims = vec.shape
        if dims == 6:
            masks = self.masks[spacegroup, :] # B * 6
            biass = self.biass[spacegroup, :] # B * 6  
        elif dims == 5:
            # - volume
            masks = self.masks[spacegroup, :-1] # B * 5
            biass = self.biass[spacegroup, :-1] # B * 5
        return vec * masks
