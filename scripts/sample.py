# Sample structures for CSP

import argparse
import os
import re
import time
from itertools import chain
from pathlib import Path
from typing import Any

import chemparse
import numpy as np
import pandas as pd
import torch
from p_tqdm import p_map
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from pyxtal.symmetry import Wyckoff_position as wp
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# LOCALFOLDER
from eval_utils import get_crystals_list, lattices_to_params_shape, load_model  # isort: skip

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

def diffusion(loader, model, return_traj, **sample_kwargs):

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
        outputs, traj = model.sample(batch, **sample_kwargs)
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
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, traj_list,
    )


class SymData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        elif key == 'symm_map':
            return self.num_nodes
        else:
            return 0


class SampleDataset(Dataset):
    def __init__(self, formula, num_evals, conditions: dict):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.wyckoff = conditions.pop('wyckoff', None)
        self.conditions = {k: torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v for k, v in conditions.items()}

        if self.wyckoff is None:
            self.get_structure()
        else:
            self.wyckoff_info: dict = parse_wyckoff(self.wyckoff)

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
        if self.wyckoff is None:
            return Data(
                atom_types=torch.LongTensor(self.chem_list),
                num_atoms=len(self.chem_list),
                num_nodes=len(self.chem_list),
                **{
                    key: val.view(1, -1)
                    for key, val in self.conditions.items()
                },
            )
        else:
            return SymData(
                atom_types=torch.LongTensor(self.wyckoff_info["atom_types"]),
                num_atoms=self.wyckoff_info['num_atoms'],
                num_nodes=self.wyckoff_info['num_atoms'],
                spacegroup=self.wyckoff_info['spacegroup'],
                ops=torch.Tensor(self.wyckoff_info['ops']),
                ops_inv=torch.Tensor(self.wyckoff_info['ops_inv']),
                anchor_index=torch.LongTensor(self.wyckoff_info['anchor_index']),
                **{
                    key: val.view(1, -1)
                    for key, val in self.conditions.items()
                },
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
        line = f.readline().strip()
        if 'formula' not in line:
            print("First line inferred NOT a HEADER, assume no header line")
            header = None
        else:
            header = 0
    formula_tabular = pd.read_csv(formula_file, header=header)
    if header is None:
        columns = list(formula_tabular.columns)
        print("Assume first column as formulas")
        columns[0] = "formula"
        if len(formula_tabular.columns) > 1:
            print("Assume second column as num_evals")
            columns[1] = "num_evals"
        formula_tabular.columns = columns

    formula_list = formula_tabular["formula"].tolist()
    if "num_evals" in formula_tabular.columns:
        num_evals_list = formula_tabular["num_evals"].astype(int).tolist()
    else:
        num_evals_list = None

    # assume any other columns are conditions
    keys = [k for k in formula_tabular.columns if (k != "formula") and (k != "num_evals")]
    conditions_list = list(formula_tabular[keys].T.to_dict().values())

    return formula_list, num_evals_list, conditions_list


def parse_conditions(cond_string: str | None) -> dict:
    conditions = {}
    if cond_string is None:
        return conditions
    for cond in cond_string.split(';'):
        key, val = cond.split('=', 1)
        if ',' in val:
            raise ValueError("vector condition is not supported yet")
        else:
            val = float(val)
        conditions[key] = val
    return conditions


def parse_wyckoff(wyckoff_string: str):  # 39_Li1x4a_Li1x4b_Li6x4c_Li7x8d
    spg, *orbits = wyckoff_string.split('_')
    spg = int(spg)
    elements, occupations, letters, multiplicities = [], [], [], []
    for orbit in orbits:
        m = re.match(r'(?P<element>([A-Z][a-z]?))(?P<occupation>(\d+))x(?P<letter>((?P<multiplicity>\d+)\w))', orbit)
        if m is None:
            raise ValueError(f"parse wyckoff error: {orbit}")
        elements.append(m.group('element'))
        occupations.append(int(m.group('occupation')))
        letters.append(m.group('letter'))
        multiplicities.append(int(m.group('multiplicity')))

    elements_on_orbit = [elem for elem, occ in zip(elements, occupations) for _ in range(occ)]
    letters_on_orbit = [letter for letter, occ in zip(letters, occupations) for _ in range(occ)]
    multiplicities_on_orbit = [mul for mul, occ in zip(multiplicities, occupations) for _ in range(occ)]
    num_atoms = sum(multiplicities_on_orbit)

    spacegroup = spg
    atom_types = np.array(list(chain.from_iterable([Element(elem).Z] * mul for elem, mul in zip(elements_on_orbit, multiplicities_on_orbit))))
    ops = np.array([op.affine_matrix for letter in letters_on_orbit for op in wp.from_group_and_letter(spg, letter)])
    ops_inv = np.linalg.pinv(ops[:, :3, :3])
    anchor_index = np.repeat(np.cumsum([0] + multiplicities_on_orbit[:-1]), multiplicities_on_orbit)

    return {
        "spacegroup": spacegroup,
        "num_atoms": num_atoms,
        "atom_types": atom_types,
        "ops": ops,
        "ops_inv": ops_inv,
        "anchor_index": anchor_index,
    }



def main(args):
    print("Loading model...")
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)

    if args.formula_file is not None:
        print(f"Trying reading sampling formula and num_evals from '{args.formula_file}'...")
        formula_list, num_evals_list, conditions_list = load_formula_tabular_file(args.formula_file)
        if num_evals_list is None:
            num_evals_list = [args.num_evals for _ in formula_list]
        # if args.guide_factor is None:
        #     conditions_list = [{}] * len(formula_list)
    else:
        formula_list = args.formula
        num_evals_list = [args.num_evals]
        conditions_list = [{}] * len(formula_list)

    # update argument conditions
    arg_conditions = parse_conditions(args.conditions)
    for c in conditions_list:
        c.update(arg_conditions)

    if conditions_list[0]:  # there exists any condition
        conditions_df = pd.DataFrame(conditions_list)
        normed_conditions = {}  # {A: [1, 2]}
        for k, v in conditions_df.to_dict('list').items():
            if k in cfg.data.properties:
                scaler_index = cfg.data.properties.index(k)
                normed_conditions[k] = model.scalers[scaler_index].transform(v)
            else:
                normed_conditions[k] = v
        normed_conditions_list = list(pd.DataFrame(normed_conditions).T.to_dict().values())
    else:
        normed_conditions_list = conditions_list

    if torch.cuda.is_available():
        model.to('cuda')

    test_set = ConcatDataset(
        SampleDataset(formula, num_evals, conditions=normed_conditions)
        for formula, num_evals, normed_conditions in zip(formula_list, num_evals_list, normed_conditions_list)
    )
    print(test_set[0])
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, traj_list) = diffusion(
        test_loader, model, return_traj=args.traj,
        step_lr=args.step_lr, N=args.ode_int_steps,
        anneal_lattice=args.anneal_lattice, anneal_coords=args.anneal_coords, anneal_slope=args.anneal_slope, anneal_offset=args.anneal_offset,
        guide_factor=args.guide_factor,
    )
    stop_time = time.time()
    print("Model time:", stop_time - start_time)

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

    torch.save(
        {
            "time": stop_time - start_time,
            "crystal_list": crystal_list,
            "args": vars(args),
        },
        str(args.save_path) + ".pt",
    )

    crystal_traj_list = [
        get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
        for frac_coords, lengths, angles in zip(traj_list[0], traj_list[1], traj_list[2])
    ]
    print("Translating sample endpoint...")
    strcuture_list = p_map(get_pymatgen, crystal_list)
    print("Translating trajectory...")
    structure_traj_list = [
        p_map(get_pymatgen, crystal_list, desc=f"{itraj=}", ncols=79)
        for itraj, crystal_list in enumerate(crystal_traj_list)
    ]
    traj_list = [get_trajectory(t) for t in zip(*structure_traj_list)]
    if not args.traj:
        print("Trajectory not saved.")

    tar_dir = os.path.join(args.save_path, "cif")
    os.makedirs(tar_dir, exist_ok=True)
    for i, structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{i}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i} is Error Structure. Skipped.")
    for i, traj in enumerate(traj_list):
        tar_file = os.path.join(tar_dir, f"{i}.XDATCAR")
        if traj is not None:
            traj.write_Xdatcar(tar_file)
        else:
            print(f"{i} is Error Trajectory. Skipped.")
    tar_dir = os.path.join(args.save_path, "vasp")
    os.makedirs(tar_dir, exist_ok=True)
    for i, structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{i}.vasp")
        if structure is not None:
            structure.to_file(tar_file, fmt="poscar")
        else:
            print(f"{i} is Error Structure. Skipped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', required=True, help="Directory of model, '`pwd`' for example.")
    parser.add_argument('-d', '--save_path', required=True, help="Directory to save results, subdir named by formula.")
    parser.add_argument('--traj', action="store_true", help="Save trajectory.")
    parser.add_argument('-n', '--num_evals', default=1, type=int, help="Sampling times of each formula. Overwrited by `num_evals` in formula_file.")
    parser.add_argument('-B', '--batch_size', default=500, type=int, help="How to split sampling times of each formula.")

    step_group = parser.add_argument_group('integrate step')
    step_group.add_argument('--step_lr', default=1e-5, type=float, help="step_lr for SDE/ODE (default 1e-5)")
    step_group.add_argument('-N', '--ode-int-steps', metavar='N', default=None, type=int, help="ODE integrate steps number; overwrite step_lr")

    formula_group = parser.add_argument_group('formula')
    formula_group = formula_group.add_mutually_exclusive_group(required=True)
    formula_group.add_argument('-f', '--formula', nargs='+', help="Formula string, multiple values are acceptable.")
    formula_group.add_argument('-F', '--formula_file', help="Formula tabular file with HEADER `formula` and `num_evals`(optional), split by WHITESPACE characters.")  # fmt: skip

    anneal_group = parser.add_argument_group('annealing')
    anneal_group.add_argument('--anneal_lattice', action="store_true", help="Anneal lattice.")
    anneal_group.add_argument('--anneal_coords', action="store_true", help="Anneal coords.")
    # anneal_group.add_argument('--anneal_type', action="store_true", help="Anneal type.")
    anneal_group.add_argument('--anneal_slope', type=float, default=0.0, help="Anneal scope")
    anneal_group.add_argument('--anneal_offset', type=float, default=0.0, help="Anneal offset.")

    guidance_group = parser.add_argument_group('guidance')
    guidance_group.add_argument('--guide-factor', type=float, help='guidance factor')
    guidance_group.add_argument('--conditions', help='conditions string as "a=b;c=d,e", conditions are splited by ";", values are treated by float or float vector')

    args = parser.parse_args()
    main(args)
