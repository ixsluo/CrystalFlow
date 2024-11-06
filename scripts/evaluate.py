# evaluate for CSP

import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy

import numpy as np


def diffusion(loader, model, num_evals, **sample_kwargs):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        batch_all_frac_coords = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):

            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, **sample_kwargs)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()


    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)


    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
    )



def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True, test_bs=args.test_bs)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    if args.ode_int_steps is not None:
        args.step_lr = 1 / args.ode_int_steps
    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi'][args.dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = diffusion(
        test_loader, model, num_evals=args.num_evals,
        step_lr=step_lr, N=args.ode_int_steps,
        anneal_lattice=args.anneal_lattice, anneal_coords=args.anneal_coords, anneal_type=args.anneal_type, anneal_slope=args.anneal_slope, anneal_offset=args.anneal_offset,
        guide_factor=args.guide_factor,
    )

    if args.label == '':
        diff_out_name = 'eval_diff.pt'
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'time': time.time() - start_time,
    }, model_path / diff_out_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--num_evals', metavar='NEVAL', default=1, type=int, help="num repeat for each sample.")
    parser.add_argument('--test_bs', type=int, help="overwrite testset batchsize.")
    parser.add_argument('--label', default='', help="label for output")

    step_group = parser.add_argument_group('evaluate step')
    step_group.add_argument('--dataset', help='load default step_lr of which dataset; effect when step_lr is -1')
    step_group.add_argument('--step_lr', default=-1, type=float, help="Step interval for ODE/SDE, -1 for SDE dataset defaults.")
    step_group.add_argument('-N', '--ode-int-steps', metavar='N', default=None, type=int, help="ODE integrate steps number; overwrite step_lr (default: None)")

    anneal_group = parser.add_argument_group('annealing')
    anneal_group.add_argument('--anneal_lattice', action="store_true", help="Anneal lattice.")
    anneal_group.add_argument('--anneal_coords', action="store_true", help="Anneal coords.")
    anneal_group.add_argument('--anneal_type', action="store_true", help="Anneal type.")
    anneal_group.add_argument('--anneal_slope', type=float, default=0.0, help="Anneal scope")
    anneal_group.add_argument('--anneal_offset', type=float, default=0.0, help="Anneal offset.")

    guidance_group = parser.add_argument_group('guidance')
    guidance_group.add_argument('--guide-factor', type=float, help='guidance factor (default: None)')

    args = parser.parse_args()
    main(args)
