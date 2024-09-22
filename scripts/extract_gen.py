import argparse
import torch
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from p_tqdm import p_map

from compute_metrics import Crystal, get_crystal_array_list
from eval_utils import get_crystals_list


def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        return structure
    except:
        return None


def main(args):
    for pt in args.pt:
        print(f"Extracting {pt}")
        crys_array_list, _ = get_crystal_array_list(pt, batch_idx=-2)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=args.njobs)

        strcuture_list = [c.structure if c.constructed else None for c in gen_crys]

        extract_dir = Path(pt).with_suffix(".dir")
        extract_dir.mkdir(exist_ok=True)
        extract_cif_dir = extract_dir / "cif"
        extract_cif_dir.mkdir(exist_ok=True)
        extract_vasp_dir = extract_dir / "vasp"
        extract_vasp_dir.mkdir(exist_ok=True)
        for i, structure in enumerate(strcuture_list):
            if structure is not None:
                CifWriter(structure).write_file(extract_cif_dir / f"{i}.cif")
                Poscar(structure).write_file(extract_vasp_dir / f"{i}.vasp")
            else:
                print(f"Error Structure index: {i}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pt", nargs="+", help="Evaluate torch pt files list")
    parser.add_argument('-j', '--njobs', default=32, type=int)

    args = parser.parse_args()
    main(args)
