import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from crystalflow.common.model_utils import load_model

train_dist = {
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
}


class SampleDataset(Dataset):
    def __init__(self, dataset, total_num, conditions: dict = None):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        conditions = conditions if conditions is not None else {}
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
        return data


def main(args):
    print(vars(args))
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    pl_model, loader, config = load_model(args.ckpt, map_location=device)
    pl_model.eval()
    test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples, conditions=None)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    res_list = []  # [[{k:v}, {k:v} /next step/], /next batch/]
    for batch in test_loader:
        res = pl_model.model.sample(batch.to(device), num_steps=args.ode_int_steps)
        res_list.append(res)

    # [{k:v}, /next step/]
    res_list = [{key: torch.cat([d[key] for d in i]) for key in i[0]} for i in zip(*res_list)]

    save_file = Path(args.ckpt).with_name(f"eval-gen-{args.label}.pt")
    torch.save(
        {
            "res_list": res_list,
        },
        save_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="checkpoint file name")
    parser.add_argument('-S', '--num_batches_to_samples', default=20, type=int, help='number of batches to sample (default: 20)')
    parser.add_argument('-B', '--batch_size', default=500, type=int, help='sample batch size (default: 500)')
    parser.add_argument('--dataset', required=True, help='dataset name for num atoms distribution')
    parser.add_argument('--device', help="device, by default auto find available GPU, otherwise CPU")

    step_group = parser.add_argument_group('evaluate step')
    step_group.add_argument('-N', '--ode-int-steps', metavar='N', default=None, type=int, help="ODE integrate steps number (default: None)")

    parser.add_argument('--label', default='unlabeled')

    args = parser.parse_args()

    main(args)