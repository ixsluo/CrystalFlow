from abc import ABC, abstractmethod
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import ProbPath, MixtureDiscreteProbPath
from flow_matching.path.path_sample import PathSample, DiscretePathSample
from flow_matching.path.scheduler import Scheduler
from flow_matching.utils import expand_tensor_like

from crystalflow.common.data_utils import MAX_ATOMIC_NUMBER


class AtomTypesProbPath(Protocol):
    mode: Literal["CSP", "DNG"]
    discrete: bool
    output_dim: tuple[str, int]  # dimention of output x_t
    decode_dim: int | None | str  # dimention requried to decode, None if CSP mode

    def sample_union(self, atom_types, t) -> PathSample | DiscretePathSample:
        raise NotImplemented

    def get_loss(self, pred, path_sample: PathSample | DiscretePathSample, *args, **kwargs) -> torch.Tensor:
        """Loss for DNG"""
        raise NotImplemented

    def decode(self, atom_types: torch.Tensor) -> torch.Tensor:
        raise NotImplemented


class ProbPathPartial(Protocol):
    def __call__(self, scheduler: Scheduler, *args, **kwargs) -> ProbPath:
        raise NotImplemented


class IdentityFakePathUnion(ProbPath):
    """Always use the same x_0 and x_1

    Minus 1 in input, add 1 in decode
    """
    def __init__(self, scheduler: Scheduler):
        super().__init__()
        self.output_dim = ("embedding", MAX_ATOMIC_NUMBER)  # CSP
        self.decode_dim = None  # CSP
        self.mode = "CSP"
        self.discrete = True

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        return PathSample(x_0=x_1, x_1=x_1, x_t=x_1, t=t, dx_t=torch.zeros_like(x_1))

    def sample_union(self, atom_types, t: torch.Tensor) -> PathSample:
        x_1 = atom_types - 1
        return self.sample(x_0=x_1, x_1=x_1, t=t)

    def decode(self, x):
        return x + 1

    def encode_random_like(self, x: torch.Tensor):
        return x.clone().detach()

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError("You should NOT use this")


class DiscreteEmbeddingPathUnion(nn.Module, ProbPath):
    """Discrete prob path and followed by embedding

    Are categories in prob path, thus decodeable.
    """
    def __init__(
        self,
        scheduler: Scheduler,
        probpath: ProbPathPartial,
        source_distr: Literal["mask", "uniform"],
    ):
        super().__init__()
        self.probpath: MixtureDiscreteProbPath = probpath(scheduler)
        self.mode = "DNG"
        self.discrete = True

        if source_distr == "mask":
            self.output_dim = ("embedding", MAX_ATOMIC_NUMBER + 1)
            self.decode_dim = MAX_ATOMIC_NUMBER + 1
        elif source_distr == "uniform":
            self.output_dim = ("embedding", MAX_ATOMIC_NUMBER)
            self.decode_dim = MAX_ATOMIC_NUMBER
        else:
            raise NotImplementedError(f"Soruce distribution {source_distr} not supported")
        self.source_distr = source_distr

        self.loss_fn = MixturePathGeneralizedKL(self.probpath, reduction="mean")


    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> DiscretePathSample:
        # x_0, x_1, x_t are all int categories
        path_sample = self.probpath.sample(t=t, x_0=x_0, x_1=x_1)
        return path_sample

    def sample_union(self, atom_types, t: torch.Tensor):
        x_1 = atom_types
        x_0 = self.encode_random_like(x_1)
        if self.source_distr == "uniform":
            x_1 = x_1 - 1
            x_0 = x_0 - 1
        path_sample = self.sample(x_0=x_0, x_1=x_1, t=t)
        return path_sample

    def decode(self, x: torch.Tensor):
        if self.source_distr == "mask":
            return x
        elif self.source_distr == "uniform":
            return x + 1
        else:
            raise NotImplementedError()

    def encode_random_like(self, x: torch.Tensor):
        if self.source_distr == "mask":
            return torch.zeros_like(x, dtype=x.dtype)  # int 0 is masked elements
        elif self.source_distr == "uniform":
            return torch.randint_like(x, low=1, high=MAX_ATOMIC_NUMBER + 1, dtype=x.dtype)
        else:
            raise RuntimeError(f"Error source distribution: {self.source_distr}")

    def get_loss(self, logits: torch.Tensor, path_sample: DiscretePathSample):
        return self.loss_fn(logits=logits, x_1=path_sample.x_1, x_t=path_sample.x_t, t=path_sample.t)

    def update(self, x_t: torch.Tensor, pred_t: torch.Tensor, t: torch.Tensor, step_size: float):
        h = step_size
        pred_prob = F.softmax(pred_t, dim=-1)
        x_1 = torch.multinomial(pred_prob, 1, replacement=True).view(*pred_prob.shape[:-1])
        scheduler_output = self.probpath.scheduler(t)
        k_t = scheduler_output.alpha_t
        d_k_t = scheduler_output.d_alpha_t

        delta_1 = F.one_hot(x_1, num_classes=self.decode_dim).to(k_t.dtype)
        u = d_k_t / (1 - k_t) * delta_1

        # Set u_t(x_t|x_t,x_1) = 0
        delta_t = F.one_hot(x_t, num_classes=self.decode_dim)
        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

        # Sample x_t ~ u_t( \cdot |x_t,x_1)
        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
        mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(-h * intensity)

        if mask_jump.sum() > 0:
            mask_prob = u[mask_jump].to(dtype=torch.get_default_dtype())
            x_t[mask_jump] = torch.multinomial(mask_prob, 1, replacement=True).view(*mask_prob.shape[:-1])

        return x_t


class TypeTablePathUnion(nn.Module, ProbPath):
    def __init__(
        self,
        scheduler: Scheduler,
        probpath: ProbPathPartial,
    ):
        super().__init__()
        self.register_buffer("reordered_map", reordered_map)
        self.register_buffer("reordered_indices", reordered_indices)
        mask = torch.where(self.reordered_map > 0, 1.0, 0.0)
        self.register_buffer("mask", mask)
        self.num_row = self.reordered_map.shape[0]  # 13
        self.num_col = self.reordered_map.shape[1]  # 15
        self.num_rowcol = self.num_row + self.num_col  # 28

        self.probpath: ProbPath = probpath(scheduler)
        self.output_dim = ("linear", self.num_rowcol)
        self.decode_dim = self.num_rowcol
        self.mode = "DNG"
        self.discrete = False

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        # x_0 continous and x_1 onehot
        path_sample = self.probpath.sample(t=t, x_0=x_0, x_1=x_1)
        return path_sample

    def sample_union(self, atom_types: torch.Tensor, t: torch.Tensor):
        x_1 = self.to_onehot(atom_types).type_as(t)
        x_0 = self.encode_random_like(x_1)
        path_sample = self.sample(x_0=x_0, x_1=x_1, t=t)
        return path_sample

    def to_onehot(self, atom_types: torch.Tensor):  # (N,)
        # atom_types: atomic number
        encoded_types = self.reordered_indices[atom_types - 1]
        encoded_types = torch.hstack(
            [
                F.one_hot(encoded_types[:, 0], self.num_row),
                F.one_hot(encoded_types[:, 1], self.num_col)
            ]
        )
        return encoded_types  # (N, 28)

    def decode(self, encoded_types: torch.Tensor):
        assert encoded_types.shape[1] == self.num_rowcol, f"TypeTable must decode shape must be (*,{self.num_rowcol}), but got {encoded_types.shape}"
        rows = encoded_types[:, :self.num_row]
        cols = encoded_types[:, self.num_row:]
        row_indices = torch.argmax(rows, dim=-1)
        col_indices = torch.argmax(F.softmax(cols, dim=1) * self.mask[row_indices], dim=-1)
        atom_types = self.reordered_map[[row_indices, col_indices]]
        return atom_types

    def encode_random_like(self, x: torch.Tensor):
        return torch.randn_like(x)

    def get_loss(self, pred, path_sample: PathSample):
        return F.mse_loss(pred, path_sample.dx_t)


reordered_table = [
    #   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14
    [ 'H', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'He', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 0  A  1
    ['Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 1  A  2
    ['Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 2  A  3
    [ 'K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 3  A  4
    ['Rb', 'Sr', 'In', 'Sn', 'Sb', 'Te',  'I', 'Xe', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 4  A  5
    ['Cs', 'Ba', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 5  A  6
    ['Fr', 'Ra', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 6  A  7
    ['Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 7  B  4
    [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 8  B  5
    ['Xx', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 9  B  6
    ['Xx', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Xx', 'Xx', 'Xx', 'Xx', 'Xx',],  # 10 B  7
    ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',],  # 11 B  6
    ['Ac', 'Th', 'Pa',  'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',],  # 12 B  7
]  # fmt: off
reordered_map = torch.LongTensor([
    #   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14
    [   1,    0,    0,    0,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0,    0,],  # 0  A  1
    [   3,    4,    5,    6,    7,    8,    9,   10,    0,    0,    0,    0,    0,    0,    0,],  # 1  A  2
    [  11,   12,   13,   14,   15,   16,   17,   18,    0,    0,    0,    0,    0,    0,    0,],  # 2  A  3
    [  19,   20,   31,   32,   33,   34,   35,   36,    0,    0,    0,    0,    0,    0,    0,],  # 3  A  4
    [  37,   38,   49,   50,   51,   52,   53,   54,    0,    0,    0,    0,    0,    0,    0,],  # 4  A  5
    [  55,   56,   81,   82,   83,   84,   85,   86,    0,    0,    0,    0,    0,    0,    0,],  # 5  A  6
    [  87,   88,  113,  114,  115,  116,  117,  118,    0,    0,    0,    0,    0,    0,    0,],  # 6  A  7
    [  21,   22,   23,   24,   25,   26,   27,   28,   29,   30,    0,    0,    0,    0,    0,],  # 7  B  4
    [  39,   40,   41,   42,   43,   44,   45,   46,   47,   48,    0,    0,    0,    0,    0,],  # 8  B  5
    [   0,   72,   73,   74,   75,   76,   77,   78,   79,   80,    0,    0,    0,    0,    0,],  # 9  B  6
    [   0,  104,  105,  106,  107,  108,  109,  110,  111,  112,    0,    0,    0,    0,    0,],  # 10 B  7
    [  57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,],  # 11 B  6
    [  89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,  100,  101,  102,  103,],  # 12 B  7
])  # fmt: off
reordered_indices = torch.LongTensor(
    [
        [0, 0],  # H
        [0, 7],  # He
        [1, 0],  # Li
        [1, 1],  # Be
        [1, 2],  # B
        [1, 3],  # C
        [1, 4],  # N
        [1, 5],  # O
        [1, 6],  # F
        [1, 7],  # Ne
        [2, 0],  # Na
        [2, 1],  # Mg
        [2, 2],  # Al
        [2, 3],  # Si
        [2, 4],  # P
        [2, 5],  # S
        [2, 6],  # Cl
        [2, 7],  # Ar
        [3, 0],  # K
        [3, 1],  # Ca
        [7, 0],  # Sc
        [7, 1],  # Ti
        [7, 2],  # V
        [7, 3],  # Cr
        [7, 4],  # Mn
        [7, 5],  # Fe
        [7, 6],  # Co
        [7, 7],  # Ni
        [7, 8],  # Cu
        [7, 9],  # Zn
        [3, 2],  # Ga
        [3, 3],  # Ge
        [3, 4],  # As
        [3, 5],  # Se
        [3, 6],  # Br
        [3, 7],  # Kr
        [4, 0],  # Rb
        [4, 1],  # Sr
        [8, 0],  # Y
        [8, 1],  # Zr
        [8, 2],  # Nb
        [8, 3],  # Mo
        [8, 4],  # Tc
        [8, 5],  # Ru
        [8, 6],  # Rh
        [8, 7],  # Pd
        [8, 8],  # Ag
        [8, 9],  # Cd
        [4, 2],  # In
        [4, 3],  # Sn
        [4, 4],  # Sb
        [4, 5],  # Te
        [4, 6],  # I
        [4, 7],  # Xe
        [5, 0],  # Cs
        [5, 1],  # Ba
        [11, 0],  # La
        [11, 1],  # Ce
        [11, 2],  # Pr
        [11, 3],  # Nd
        [11, 4],  # Pm
        [11, 5],  # Sm
        [11, 6],  # Eu
        [11, 7],  # Gd
        [11, 8],  # Tb
        [11, 9],  # Dy
        [11, 10],  # Ho
        [11, 11],  # Er
        [11, 12],  # Tm
        [11, 13],  # Yb
        [11, 14],  # Lu
        [9, 1],  # Hf
        [9, 2],  # Ta
        [9, 3],  # W
        [9, 4],  # Re
        [9, 5],  # Os
        [9, 6],  # Ir
        [9, 7],  # Pt
        [9, 8],  # Au
        [9, 9],  # Hg
        [5, 2],  # Tl
        [5, 3],  # Pb
        [5, 4],  # Bi
        [5, 5],  # Po
        [5, 6],  # At
        [5, 7],  # Rn
        [6, 0],  # Fr
        [6, 1],  # Ra
        [12, 0],  # Ac
        [12, 1],  # Th
        [12, 2],  # Pa
        [12, 3],  # U
        [12, 4],  # Np
        [12, 5],  # Pu
        [12, 6],  # Am
        [12, 7],  # Cm
        [12, 8],  # Bk
        [12, 9],  # Cf
        [12, 10],  # Es
        [12, 11],  # Fm
        [12, 12],  # Md
        [12, 13],  # No
        [12, 14],  # Lr
        [10, 1],  # Rf
        [10, 2],  # Db
        [10, 3],  # Sg
        [10, 4],  # Bh
        [10, 5],  # Hs
        [10, 6],  # Mt
        [10, 7],  # Ds
        [10, 8],  # Rg
        [10, 9],  # Cn
        [6, 2],  # Nh
        [6, 3],  # Fl
        [6, 4],  # Mc
        [6, 5],  # Lv
        [6, 6],  # Ts
        [6, 7],  # Og
    ],
)