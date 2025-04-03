from typing import Any

import torch
import numpy
from pymatgen.core.structure import Structure


def resove_sample_output(res_list: list[dict[str, Any]], traj=False) -> tuple[list, list]:
    """resolve model sample output

    Shape as list of each ODE step, each step is a dict of stacked tensor
        [
            {
                'num_atoms': ...,
                'atom_types': ...,
                'frac_coords': ...,
                'lattice': ...,
            },
            <next step>
        ]

    Returns
    -------
    traj_list : list[list[Structure]]
        List trajectory of all structures.
    structure_list : list[Structure]
        The last structures. Same as `traj_list[-1]`.
    """
    traj_list = []
    for istep, step_output in enumerate(res_list):
        if not traj and (istep < len(res_list) - 1):  # skip unless the last step if trajectory is not desired
                continue
        num_atoms = step_output["num_atoms"]
        atom_types = step_output["atom_types"]
        frac_coords = step_output["frac_coords"]
        lattice = step_output["lattice"]
        if len(num_atoms) != lattice.size(0):
            raise RuntimeError(f"Number of structures mismatch! {num_atoms.shape=}, {lattice.shape=}")
        if not (num_atoms.sum() == atom_types.size(0) == frac_coords.size(0)):
            raise RuntimeError(f"Number of atoms mismatch! {num_atoms.sum()=}, {atom_types.shape=}, {frac_coords.shape=}")

        structure_list = []
        num_atoms = num_atoms.tolist()
        for one_atom_types, one_frac_coords, one_lattice in zip(
            torch.split(atom_types, num_atoms), torch.split(frac_coords, num_atoms), lattice
        ):
            try:
                structure = Structure(
                    lattice=one_lattice.numpy(),
                    species=one_atom_types.numpy(),
                    coords=one_frac_coords.numpy(),
                    coords_are_cartesian=False,
                )
            except Exception as e:
                structure = None
            structure_list.append(structure)
        traj_list.append(structure_list)
    return traj_list, structure_list
