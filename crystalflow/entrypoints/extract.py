import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from crystalflow.common.eval_utils import resove_sample_output


def extract(pt_file):
    stored_data = torch.load(pt_file, weights_only=False)
    res_list = stored_data["res_list"]
    extract_dir = Path(pt_file).with_suffix(".dir")
    extract_dir.mkdir(exist_ok=True)
    traj_list, structure_list = resove_sample_output(res_list)

    # write final structures
    vasp_dir = extract_dir.joinpath("vasp")
    vasp_dir.mkdir(exist_ok=True)
    for i, structure in tqdm(enumerate(structure_list), total=len(structure_list)):
        if structure is None:
            continue
        structure.to_file(vasp_dir.joinpath(f"{i}.vasp"), fmt="poscar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_files", nargs="+")
    args = parser.parse_args()
    for pt_file in args.pt_files:
        extract(pt_file)
