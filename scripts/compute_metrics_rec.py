import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_map

from compute_metrics import Crystal, RecEval, get_file_paths, get_gt_crys_ori
from eval_utils import load_data, get_crystals_list


def get_crystal_array_list(file_path, njobs):
    data = load_data(file_path)
    assert data['frac_coords'].dim() == 3
    batch_size = data['frac_coords'].shape[0]
    print("Reading batched generated crystals...")
    batched_crys_array_list= p_map(
        get_crystals_list,
        data['frac_coords'],
        data['atom_types'],
        data['lengths'],
        data['angles'],
        data['num_atoms'],
        num_cpus=njobs,
        ncols=79,
    )
    print("Reading ground truth crystals...")
    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        print("No ground truth crystals found.")
        true_crystal_array_list = None
    return batched_crys_array_list, true_crystal_array_list


def main(args):
    recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
    batched_crys_array_list, true_crystal_array_list = get_crystal_array_list(recon_file_path, args.njobs)
    print("Parsing ground truth...")
    if args.gt_file != '':
        csv = pd.read_csv(args.gt_file)
        gt_crys = p_map(get_gt_crys_ori, csv['cif'], num_cpus=args.njobs, ncols=79)
    else:
        gt_crys = p_map(lambda x: Crystal(x, True, False, True), true_crystal_array_list, num_cpus=args.njobs, ncols=79)

    print("Parsing batched predicted structures...")
    batched_rms_dists = []
    for ibatch, crys_array_list in enumerate(batched_crys_array_list):
        pred_crys = p_map(lambda x: Crystal(x, True, False, True), crys_array_list, num_cpus=args.njobs, ncols=79, desc=f"{ibatch=}")
        rec_evaluator = RecEval(pred_crys, gt_crys, njobs=args.njobs)
        _ = rec_evaluator.get_metrics()
        rms_dists = rec_evaluator.rms_dists
        batched_rms_dists.append(rms_dists)
    batched_rms_dists = np.array(batched_rms_dists, dtype=float)

    recon_file_path = Path(recon_file_path)
    metrics_dir = recon_file_path.with_name(recon_file_path.stem)
    metrics_dir.mkdir(exist_ok=True)
    np.save(metrics_dir / f"rms.npy", batched_rms_dists)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('-j', '--njobs', default=32, type=int)
    args = parser.parse_args()
    main(args)

