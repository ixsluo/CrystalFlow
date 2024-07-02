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

    def forward(self, batch):

        batch_size = batch.num_graphs
        # times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        # time_emb = self.time_embedding(times)
        eps = 1e-3
        times = torch.rand(batch_size, device=self.device) * (1 - eps) + eps  # [eps, 1]
        time_emb = self.time_embedding(times)

        # alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        # beta = self.beta_scheduler.betas[times]

        # c0 = torch.sqrt(alphas_cumprod)
        # c1 = torch.sqrt(1. - alphas_cumprod)

        # sigmas = self.sigma_scheduler.sigmas[times]
        # sigmas_norm = self.sigma_scheduler.sigmas_norm[times]
        # sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        # sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        # rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        # lengths0 = self.sample_lengths(batch.num_atoms, batch_size)
        # angles0 = self.sample_angles(batch.num_atoms, batch_size)
        # l0 = lattice_params_to_matrix_torch(lengths0, angles0)
        l0 = torch.randn_like(lattices)
        f0 = torch.rand_like(frac_coords)

        if self.ot:
            _, l0 = self.permute_l(lattices, l0)
            _, f0 = self.permute_f(frac_coords, f0)

        tar_l = lattices - l0
        tar_f = (frac_coords - f0) % 1 - 0.5

        input_lattice = l0 + times[:, None, None] * tar_l
        input_frac_coords = f0 + times.repeat_interleave(batch.num_atoms)[:, None] * tar_f

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_f = self.decoder(
            time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch
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

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {
            0: {'num_atoms': batch.num_atoms, 'atom_types': batch.atom_types, 'frac_coords': x_T % 1.0, 'lattices': l_T}
        }

        for t in tqdm(range(1, N + 1)):

            times = torch.full((batch_size,), t, device=self.device) / N

            time_emb = self.time_embedding(times)

            x_t = traj[t - 1]['frac_coords']
            l_t = traj[t - 1]['lattices']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            x_t = x_t + pred_x / N if not self.keep_coords else x_t
            l_t = l_t + pred_l / N if not self.keep_lattice else l_t

            traj[t] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': x_t % 1.0,
                'lattices': l_t,
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
