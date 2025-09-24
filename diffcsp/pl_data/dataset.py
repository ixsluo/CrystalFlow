import os
import pickle
from typing import Any

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
import pandas as pd
from omegaconf import ValueNode
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


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


class CrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        prop: ValueNode,
        properties: ValueNode = None,
        conditions: ValueNode = None,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.prop = prop
        self.properties = [] if properties is None else properties
        self.conditions = [] if conditions is None else conditions

        self.preprocess(save_path, preprocess_workers, self.properties)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None
        self.scalers = None

    def preprocess(self, save_path, preprocess_workers, properties):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=properties,
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        assert isinstance(index, int), "Only one item can be returned at one time."
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        if isinstance(self.scalers, list):
            prop_dict = {
                key: scaler.transform(data_dict[key])
                for key, scaler in zip(self.properties, self.scalers, strict=True)
            }
        elif isinstance(self.scalers, dict):
            if any(prop not in self.scalers for prop in self.properties):
                raise ValueError(f"Property {prop} not found in scalers.")
            prop_dict = {
                key: scaler.transform(data_dict[key])
                for key, scaler in self.scalers.items()
            }
        else:
            raise ValueError("Scalers must be either a list or a dict.")

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, lattice_polar) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = SymData(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            lattice_polar=torch.Tensor(lattice_polar).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            **{
                key: val.view(1, -1)
                for key, val in prop_dict.items()
            }
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])
            data.ops_inv = torch.linalg.pinv(data.ops[:,:3,:3])
            data.num_general_ops = len(data_dict['general_wyckoff_ops'])          # Nops
            data.general_ops = torch.Tensor(data_dict['general_wyckoff_ops'])     # (Nops, 4, 4)
            data.general_ops = F.pad(data.general_ops, (0, 0, 0, 0, 0, 192 - data.num_general_ops)).view(1, 192, 4, 4)
            data.symm_map = torch.LongTensor(data_dict['symm_map'])               # (Nops, Nat)
            data.symm_map = F.pad(data.symm_map, (0, 0, 0, 192 - data.num_general_ops)).transpose(-1, -2)  # (Nat, 192)


        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, lattice_polar) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            lattice_polar=torch.Tensor(lattice_polar).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    dataset.lattice_scaler = get_scaler_from_data_list(dataset.cached_data, key='scaled_lattice')
    dataset.scaler = get_scaler_from_data_list(dataset.cached_data, key=dataset.prop)
    dataset.scalers = [get_scaler_from_data_list(dataset.cached_data, key=key) for key in dataset.properties]

    for valcfg in cfg.data.datamodule.datasets.val:
        hydra.utils.instantiate(valcfg, _recursive_=False)
    for testcfg in cfg.data.datamodule.datasets.test:
        hydra.utils.instantiate(testcfg, _recursive_=False)

    print("One Sample: ", dataset[0])
    print(dataset[0].anchor_index)
    print(dataset[0].ops[:4])
    print(dataset[0].ops_inv[:4])
    data_list = [dataset[i] for i in range(len(dataset))[:2]]
    batch = Batch.from_data_list(data_list)
    print("Total Batch: ", batch)
    print(batch.anchor_index)
    return batch


if __name__ == "__main__":
    main()
