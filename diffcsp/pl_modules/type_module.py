# Encoding atom types
import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeTableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.reordered_table = [
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
        self.register_buffer("reordered_map", reordered_map)
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
        self.register_buffer("reordered_indices", reordered_indices)
        mask = torch.where(self.reordered_map > 0, 1.0, 0.0)
        self.register_buffer("mask", mask)
        self.num_row = 13
        self.num_col = 15
        self.out_dim = 28

    def forward(self, atom_types: torch.Tensor):
        return self.encode_types(atom_types)

    def encode_types(self, atom_types: torch.Tensor):  # (N,)
        # atom_types: atomic number
        encoded_types = self.reordered_indices[atom_types - 1]
        encoded_types = torch.hstack(
            [
                F.one_hot(encoded_types[:, 0], self.num_row),
                F.one_hot(encoded_types[:, 1], self.num_col)
            ]
        )
        return encoded_types  # (N, 28)

    def decode_types(self, encoded_types: torch.Tensor):
        rows = encoded_types[:, :self.num_row]
        cols = encoded_types[:, self.num_row:]
        row_indices = torch.argmax(rows, dim=-1)
        col_indices = torch.argmax(F.softmax(cols, dim=1) * self.mask[row_indices], dim=-1)
        atom_types = self.reordered_map[[row_indices, col_indices]]
        return atom_types

    def get_rd_encoded_types(self, num_nodes: int, device=None):
        encoded_types = torch.randn((num_nodes, self.out_dim), device=device)
        return encoded_types
