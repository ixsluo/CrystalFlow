import argparse
import os
import time
from pathlib import Path

import chemparse
import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from torch.optim import Adam
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# LOCALFOLDER
from eval_utils import get_crystals_list, lattices_to_params_shape, load_model, get_t_span  # isort: skip

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']  # fmt: skip

def diffusion(loader, model, num_evals, t_span, solver, integrate_sequence, return_traj):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []

    all_frac_coords = []
    all_lattices = []
    all_lengths = []
    all_angles = []

    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample_ode(batch, t_span, solver, integrate_sequence)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

        if return_traj:
            all_frac_coords.append(traj["all_frac_coords"].detach().cpu())
            all_lattices.append(traj["all_lattices"].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    if return_traj:
        all_frac_coords = torch.cat(all_frac_coords, dim=1)  # dim 0 for traj
        all_frac_coords = [torch.cat(f, dim=0) for f in zip(all_frac_coords)]
        all_lattices = torch.cat(all_lattices, dim=1)  # dim 0 for traj
        all_lattices = [torch.cat(l, dim=0) for l in zip(all_lattices)]
        all_lengths_angles = [lattices_to_params_shape(l) for l in all_lattices]
        all_lengths = [lengths_angles[0] for lengths_angles in all_lengths_angles]
        all_angles = [lengths_angles[1] for lengths_angles in all_lengths_angles]

    traj_list = [
        all_frac_coords,
        all_lengths,
        all_angles,
        num_atoms,
        atom_types,
    ]

    return (
        frac_coords,
        atom_types,
        lattices,
        lengths,
        angles,
        num_atoms,
        traj_list,
    )

class SampleDataset(Dataset):

    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )


def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None


def get_trajectory(structures):
    try:
        return Trajectory.from_structures(structures, constant_lattice=False, coords_are_displacement=False)
    except Exception:
        return None


def load_formula_tabular_file(formula_file):
    with open(formula_file, "r") as f:
        line = f.readline().split()
        if 'formula' not in line:
            print("First line inferred NOT a HEADER, assume no header line")
            header = None
        else:
            header = 0
    formula_tabular = pd.read_csv(formula_file, sep=r'\s+', header=header)
    if header is None:
        print("Assume first column as formulas")
        formula_list = formula_tabular[0].astype(str).tolist()
        if len(formula_tabular.columns) > 1:
            print("Assume second column as num_evals")
            num_evals_list = formula_tabular[1].astype(int).tolist()
        else:
            num_evals_list = None
    else:
        formula_list = formula_tabular["formula"].tolist()
        if "num_evals" in formula_tabular.columns:
            num_evals_list = formula_tabular["num_evals"].astype(int).tolist()
        else:
            num_evals_list = None
    return formula_list, num_evals_list


def main(args):
    print("Loading model...")
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)
    if torch.cuda.is_available():
        model.to('cuda')

    t_span = get_t_span(args.ode_scheduler, args.ode_int_steps)
    if args.integrate_sequence in ["lf", "lattice_first"]:
        integrate_sequence = "lattice_first"
    elif args.integrate_sequence in ["cf", "coords_first"]:
        integrate_sequence = "coords_first"
    else:
        raise NotImplementedError("Unknown integrate sequence")

    assert (args.formula or args.formula_file), "At least one of formula or formula_list should be provided."
    if args.formula_file is not None:
        print(f"Trying reading sampling formula and num_evals from '{args.formula_file}'...")
        formula_list, num_evals_list = load_formula_tabular_file(args.formula_file)
        if num_evals_list is None:
            num_evals_list = [args.num_evals for _ in formula_list]
    else:
        formula_list = args.formula
        num_evals_list = [args.num_evals]

    for formula, num_evals in zip(formula_list, num_evals_list):
        tar_dir = os.path.join(args.save_path, formula)
        os.makedirs(tar_dir, exist_ok=True)

        print(f'Sampling {formula} times {num_evals}...')

        test_set = SampleDataset(formula, num_evals)
        test_loader = DataLoader(test_set, batch_size = min(args.batch_size, num_evals))

        start_time = time.time()
        (
            frac_coords,
            atom_types,
            lattices,
            lengths,
            angles,
            num_atoms,
            traj_list,
        ) = diffusion(
            test_loader, model, args.num_evals, t_span, args.solver, integrate_sequence,
            return_traj=args.traj,
        )

        crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
        crystal_traj_list = [
            get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
            for frac_coords, lengths, angles in zip(traj_list[0], traj_list[1], traj_list[2])
        ]

        print("Translating sample endpoint...")
        strcuture_list = p_map(get_pymatgen, crystal_list, ncols=79)
        print("Translating trajectory...")
        structure_traj_list = [
            p_map(get_pymatgen, crystal_list, desc=f"{itraj=}", ncols=79)
            for itraj, crystal_list in enumerate(crystal_traj_list)
        ]
        traj_list = [get_trajectory(t) for t in zip(*structure_traj_list)]
        if not args.traj:
            print("Trajectory not saved.")

        for i, structure in enumerate(strcuture_list):
            tar_file = os.path.join(tar_dir, f"{formula}_{i+1}.cif")
            if structure is not None:
                writer = CifWriter(structure)
                writer.write_file(tar_file)
            else:
                print(f"{i+1} Error Structure.")
        for i, traj in enumerate(traj_list):
            tar_file = os.path.join(tar_dir, f"{formula}_{i+1}.XDATCAR")
            if traj is not None:
                traj.write_Xdatcar(tar_file, system=formula)
            else:
                print(f"{i+1} Error Trajectory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', required=True, help="Directory of model, '`pwd`' for example.")
    formula_group = parser.add_mutually_exclusive_group(required=True)
    formula_group.add_argument('-f', '--formula', nargs='+', help="Formula string, multiple values are acceptable.")
    formula_group.add_argument('-F', '--formula_file', help="Formula tabular file with HEADER `formula` and `num_evals`(optional), split by WHITESPACE characters.")  # fmt: skip
    parser.add_argument('-d', '--save_path', required=True, help="Directory to save results, subdir named by formula.")
    parser.add_argument('--traj', action="store_true", help="Save trajectory.")
    parser.add_argument('-n', '--num_evals', default=1, type=int, help="Sampling times of each formula.")
    parser.add_argument('-B', '--batch_size', default=500, type=int, help="How to split sampling times of each formula.")
    parser.add_argument('-N', '--ode_int_steps', type=int, default=20, help="ODE integrate steps number.")
    parser.add_argument('--ode_scheduler', choices=['linspace'], default='linspace', help="ODE integrate time spam scheduler.")
    parser.add_argument('--solver', choices=[
        'euler', 'midpoint',
        'rk4', 'rk-4', 'RungeKutta4',
        'ieuler', 'implicit_euler',
        # 'alf', 'AsynchronousLeapfrog'
        ], default="euler", help="ODE integrate solver, default: euler")
    parser.add_argument('-seq', "--integrate_sequence", choices=['lf', 'lattice_first', 'cf', 'coords_first'], default='lf', help="Which to integrate first.")

    args = parser.parse_args()


    main(args)
