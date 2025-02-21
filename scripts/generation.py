# generation for abinit

import time
import argparse
import torch
from ast import literal_eval

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import pdb

import os

train_dist = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp_20' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301],
    'alex_20': [0.0,
            7.122341643112791e-05,
            0.0025145293865150077,
            0.03703922353526378,
            0.26655192206101525,
            0.17283409400195998,
            0.06460116219857165,
            0.02456446119104596,
            0.0844690675178944,
            0.028310355846661698,
            0.07688929633930683,
            0.029568763161573176,
            0.0820745134050468,
            0.007102155327225894,
            0.02186025660997105,
            0.010584104380773333,
            0.028149246193451178,
            0.00163128284799209,
            0.026717160387135452,
            0.00816136560046101,
            0.026305816591704338,
    ],
    'alex_mp_20': [0.0,
            0.0002122817324164079,
            0.0027662449007130364,
            0.019320928839542985,
            0.1635523784604802,
            0.04660818222658853,
            0.0779666372105193,
            0.027198391266499145,
            0.11488555710790001,
            0.048953154852118624,
            0.12601800609857441,
            0.035773585899227064,
            0.14604654071283876,
            0.006003129921357023,
            0.028677781014114268,
            0.020232588372556086,
            0.04501524643605301,
            0.0013033111013472484,
            0.038740593368581974,
            0.007013525143866128,
            0.04371193533470576,
    ],
    'alex': [0.0,
            7.100223977261463e-05,0.0024171389939845788,0.03611229602866374,0.25964405324221285,0.16890263393249944,0.06286454777420508,0.02403551114373192,0.08231916147190149,0.02749401632104818,0.07488146802560312,
            0.028860322166398066,0.0798588642537414,0.006913390632604897,0.02134633611736364,0.010433709524546727,0.027449744336248787,0.0016099409818245403,0.02598013719303365,0.0079901187157449,0.025557743476425587,
            0.0005883440495668028,0.0026137177441001315,0.00026507502848442796,0.004468686063179185,0.0001637228117109702,0.0014994002399040384,0.0001431182401691134,0.003724972406580543,0.0010199262913219112,0.0015394956223638678,
            5.513115088226548e-05,0.001783687639150468,2.7287135285161703e-05,0.00012863935205861945,3.0349977000843116e-05,0.0004986863193441286,2.116145185379887e-05,0.00013894163782954784,3.452657934040868e-05,0.0004519083731409943,
            2.50596140373934e-05,0.00015202832516018663,8.63164483510217e-06,0.00022525808618056956,2.227521247768302e-05,0.00011193294270035719,1.1972926706754624e-05,0.00026089842614486237,1.25298070186967e-05,5.7080231974062744e-05,
            6.4041235873338685e-06,0.00015007924406838937,2.7844015597103775e-06,9.828937505777633e-05,9.188525147044246e-06,0.00017012693529830407,4.1766023395655665e-06,4.566418557925019e-05,5.568803119420755e-06,0.00011471734426006756,
            1.3922007798551888e-06,3.229905809264038e-05,3.0628417156814155e-06,0.00013754943704969265,6.125683431362831e-06,2.3110532945596136e-05,5.568803119420755e-07,8.659488850699274e-05,4.1766023395655665e-06,2.867933606501689e-05,
            1.9490810917972645e-06,0.00011861550644366209,2.227521247768302e-06,1.8098610138117454e-05,1.3922007798551888e-06,6.431967602930973e-05,2.227521247768302e-06,2.7008695129190662e-05,2.7844015597103775e-06,0.0036166591859078095,
            1.9490810917972645e-06,1.1972926706754624e-05,1.6706409358262265e-06,5.01192280747868e-05,5.568803119420755e-07,9.745405458986322e-06,1.3922007798551888e-06,3.9538502147887365e-05,1.113760623884151e-06,7.796324367189058e-06,
            1.3922007798551888e-06,3.118529746875623e-05,0.0,4.1766023395655665e-06,8.353204679131133e-07,4.037382261580048e-05,2.7844015597103775e-07,6.9610038992759445e-06,1.6706409358262265e-06,2.1439892009769908e-05,
            0.0,3.619722027623491e-06,0.0,2.8122455753074815e-05,8.353204679131133e-07,2.004769122991472e-05,2.7844015597103775e-07,1.754172982617538e-05,2.7844015597103775e-07,1.3922007798551888e-06,
            0.0,1.893393060603057e-05,0.0,1.6706409358262265e-06,0.0,7.239444055246982e-06,2.7844015597103775e-07,1.113760623884151e-06,0.0,6.9610038992759445e-06,
            0.0,1.113760623884151e-06,0.0,3.619722027623491e-06,2.7844015597103775e-07,1.6706409358262265e-06,0.0,5.8472432753917935e-06,0.0,2.7844015597103775e-07,
            0.0,3.619722027623491e-06,0.0,0.0,2.7844015597103775e-07,3.898162183594529e-06,0.0,1.3922007798551888e-06,0.0,3.898162183594529e-06,
            0.0,2.7844015597103775e-07,0.0,3.898162183594529e-06,0.0,0.0,2.7844015597103775e-07,1.113760623884151e-06,0.0,0.0,
            0.0,2.50596140373934e-06,0.0,0.0,0.0,1.113760623884151e-06,0.0,0.0,0.0,4.7334826515076424e-06,
            0.0,0.0,0.0,2.7844015597103775e-07,0.0,0.0,0.0,1.113760623884151e-06,0.0,0.0,
            0.0,2.7844015597103775e-07,0.0,0.0,0.0,5.568803119420755e-07,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,5.568803119420755e-07,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,2.227521247768302e-06,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,2.7844015597103775e-07,
    ]

}


def diffusion(
    loader, model,
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
        outputs, traj = model.sample(batch, **sample_kwargs)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

class SampleDataset(Dataset):
    def __init__(self, dataset, total_num, conditions: dict):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon_24'
        self.conditions = {k: torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v for k, v in conditions.items()}

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            **{
                key: val.view(1, -1)
                for key, val in self.conditions.items()
            },
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data


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


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)

    if args.guide_factor is not None:
        conditions = parse_conditions(args.conditions)
        for k, v in conditions.items():
            if k in cfg.data.properties:
                scaler_index = cfg.data.properties.index(k)
                conditions[k] = model.scalers[scaler_index].transform(v)
            else:
                conditions[k] = v
    else:
        conditions = {}

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples, conditions=conditions)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    if args.ode_int_steps is not None:
        args.step_lr = 1 / args.ode_int_steps
    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['gen'][args.dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(
        test_loader, model,
        step_lr=step_lr, N=args.ode_int_steps,
        anneal_lattice=args.anneal_lattice, anneal_coords=args.anneal_coords, anneal_type=args.anneal_type, anneal_slope=args.anneal_slope, anneal_offset=args.anneal_offset,
        guide_factor=args.guide_factor,
    )

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'time': time.time() - start_time,
    }, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-S', '--num_batches_to_samples', default=20, type=int, help='number of batches to sample (default: 20)')
    parser.add_argument('-B', '--batch_size', default=500, type=int, help='sample batch size (default: 500)')
    parser.add_argument('--label', default='')

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
    guidance_group.add_argument('--conditions', help='conditions string as "a=b;c=d,e", conditions are splited by ";", values are treated by float or float vector')

    args = parser.parse_args()


    main(args)
