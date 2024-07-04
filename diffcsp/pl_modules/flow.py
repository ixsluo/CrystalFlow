# Training example:
#
# HYDRA_FULL_ERROR=1 python diffcsp/run.py \
# data=mp_20 model=flow \
# logging.wandb.group=mp_20 expname=flow-test-01

import copy
import math
from typing import Any, Dict

import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from tqdm import tqdm

import hydra
from diffcsp.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lattice_params_to_matrix_torch,
    lattice_polar_build_torch,
    lattice_polar_decompose_torch,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
)
from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from diffcsp.pl_modules.hungarian import HungarianMatcher

MAX_ATOMIC_NUM = 100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial")
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.optim.lr_scheduler, optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPFlow(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder, latent_dim=self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False
        )
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.ot = self.hparams.get("ot", False)
        self.permute_l = HungarianMatcher("norm")
        self.permute_f = HungarianMatcher("norm_mic")
        self.lattice_polar = self.hparams.get("lattice_polar", False)

    def sample_lengths(self, num_atoms, batch_size):
        loc = math.log(2)
        scale = math.log(1)
        lengths = torch.randn((batch_size, 3), device=self.device)
        lengths = torch.exp(lengths * scale + loc)
        lengths = num_atoms[:, None] * lengths
        return lengths

    def sample_angles(self, num_atoms, batch_size):
        angles = torch.rand((batch_size, 3), device=self.device)  # [60, 120)
        angles = angles * 60 + 60
        return angles

    def sample_lattice(self, batch_size):
        l0 = torch.randn([batch_size, 3, 3], device=self.device)
        return l0

    def sample_lattice_polar(self, batch_size):
        l0 = torch.randn([batch_size, 6], device=self.device)
        l0[:, -1] = l0[:, -1] + 1
        return

    def forward(self, batch):

        batch_size = batch.num_graphs
        eps = 1e-3
        times = torch.rand(batch_size, device=self.device) * (1 - eps) + eps  # [eps, 1]
        time_emb = self.time_embedding(times)

        # Build time stamp T and 0
        lattices_mat_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        if self.lattice_polar:
            lattices_rep_T = lattice_polar_decompose_torch(lattices_mat_T)
            lattices_rep_0 = self.sample_lattice_polar(batch_size)
        else:
            lattices_rep_T = lattices_mat_T
            lattices_rep_0 = self.sample_lattice(batch_size)

        frac_coords = batch.frac_coords
        f0 = torch.rand_like(frac_coords)

        if self.ot:
            _, lattices_rep_0 = self.permute_l(lattices_rep_T, lattices_rep_0)
            _, f0 = self.permute_f(frac_coords, f0)

        # Build time stamp t
        tar_l = lattices_rep_T - lattices_rep_0
        tar_f = (frac_coords - f0) % 1 - 0.5

        input_lattice_rep = lattices_rep_0 + times[:, None, None] * tar_l
        input_frac_coords = f0 + times.repeat_interleave(batch.num_atoms)[:, None] * tar_f

        if self.lattice_polar:
            input_lattice_mat = lattice_polar_build_torch(input_lattice_rep)
        else:
            input_lattice_mat = input_lattice_rep

        #
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice_rep = lattices_rep_T
            input_lattice_mat = lattices_mat_T

        pred_l, pred_f = self.decoder(
            t=time_emb,
            atom_types=batch.atom_types,
            frac_coords=input_frac_coords,
            lattices_rep=input_lattice_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=input_lattice_mat,
        )

        loss_lattice = F.mse_loss(pred_l, tar_l)
        loss_coord = F.mse_loss(pred_f, tar_f)

        loss = self.hparams.cost_lattice * loss_lattice + self.hparams.cost_coord * loss_coord

        return {'loss': loss, 'loss_lattice': loss_lattice, 'loss_coord': loss_coord}

    @torch.no_grad()
    def sample(self, batch, step_lr=None, N=None):
        if N is None:
            N = int(1 / step_lr)

        batch_size = batch.num_graphs

        # time stamp T
        if self.lattice_polar:
            l_T = self.sample_lattice_polar(batch_size)
            lattices_mat_T = lattice_polar_build_torch(l_T)
        else:
            l_T = self.sample_lattice(batch_size)
            lattices_mat_T = l_T

        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)

        #
        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            lattices_mat_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            if self.lattice_polar:
                l_T = lattice_polar_decompose_torch(lattices_mat_T)
            else:
                l_T = lattices_mat_T

        traj = {
            0: {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': x_T % 1.0,
                'lattices': lattices_mat_T,
            }
        }

        l_t = l_T.clone().detach()
        lattices_mat_t = lattices_mat_T.detach().clone()
        x_t = x_T.clone().detach()

        for t in tqdm(range(1, N + 1)):

            times = torch.full((batch_size,), t, device=self.device) / N
            time_emb = self.time_embedding(times)

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T
                lattices_mat_t = lattices_mat_T

            pred_l, pred_x = self.decoder(
                t=time_emb,
                atom_types=batch.atom_types,
                frac_coords=x_t,
                lattices_rep=l_t,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat_t,
            )

            x_t = x_t + pred_x / N if not self.keep_coords else x_t
            l_t = l_t + pred_l / N if not self.keep_lattice else l_t
            x_t = x_t % 1.0

            if self.lattice_polar:
                lattices_mat_t = lattice_polar_build_torch(l_t)
            else:
                lattices_mat_t = l_t

            traj[t] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': x_t,
                'lattices': lattices_mat_t,
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(0, N + 1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(0, N + 1)]),
        }

        return traj[N], traj_stack

    def training_step(self, batch, batch_idx: int):

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        self.log_dict(
            {'train_loss': loss, 'lattice_loss': loss_lattice, 'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {f'{prefix}_loss': loss, f'{prefix}_lattice_loss': loss_lattice, f'{prefix}_coord_loss': loss_coord}

        return log_dict, loss
