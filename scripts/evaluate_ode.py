import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, lattices_to_params_shape, get_t_span

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy

import numpy as np


def diffusion(
    loader, model, num_evals, t_span, solver, integrate_sequence,
    **sample_kwargs,
):

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
            outputs, traj = model.sample_ode(batch, t_span, solver, integrate_sequence, **sample_kwargs)
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
        frac_coords,
        atom_types,
        lattices,
        lengths,
        angles,
        num_atoms,
        input_data_batch,
    )



def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True, test_bs=args.test_bs)
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    t_span = get_t_span(args.ode_scheduler, args.ode_int_steps)
    if args.integrate_sequence in ["lf", "lattice_first"]:
        integrate_sequence = "lattice_first"
    elif args.integrate_sequence in ["cf", "coords_first"]:
        integrate_sequence = "coords_first"
    else:
        raise NotImplementedError("Unknown integrate sequence")

    start_time = time.time()
    (
        frac_coords,
        atom_types,
        lattices,
        lengths,
        angles,
        num_atoms,
        input_data_batch,
    ) = diffusion(
        test_loader, model, args.num_evals, t_span, args.solver, integrate_sequence,
        anneal_lattice=args.anneal_lattice, anneal_coords=args.anneal_coords,
        anneal_slope=args.anneal_slope, anneal_offset=args.anneal_offset,
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
    parser.add_argument('-m', '--model_path', required=True, help="Directory of model, '`pwd`' for example.")
    parser.add_argument('-N', '--ode_int_steps', type=int, default=20, help="ODE integrate steps number.")
    parser.add_argument('--ode_scheduler', choices=['linspace'], default='linspace', help="ODE integrate time spam scheduler.")
    parser.add_argument('--solver', choices=[
        'euler', 'midpoint',
        'rk4', 'rk-4', 'RungeKutta4',
        'ieuler', 'implicit_euler',
        # 'alf', 'AsynchronousLeapfrog'
        ], default="euler", help="ODE integrate solver.")
    parser.add_argument('-seq', "--integrate_sequence", choices=['lf', 'lattice_first', 'cf', 'coords_first'], default='lf', help="Which to integrate first")
    parser.add_argument('--anneal_lattice', action="store_true", help="Anneal lattice.")
    parser.add_argument('--anneal_coords', action="store_true", help="Anneal coords.")
    parser.add_argument('--anneal_slope', type=float, default=0.0, help="Anneal scope")
    parser.add_argument('--anneal_offset', type=float, default=0.0, help="Anneal offset.")
    parser.add_argument('--num_evals', default=1, type=int, help="num repeat for each sample.")
    parser.add_argument('--test_bs', type=int, help="overwrite testset batchsize.")
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)
