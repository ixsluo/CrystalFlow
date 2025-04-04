import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.lattice.crystal_family import CrystalFamily
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

from copy import deepcopy as dc

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "frequency": 5, "monitor": "val_loss"}}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
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


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.time_dim, pred_type = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.crystal_family = CrystalFamily()




    def forward(self, batch, batch_idx = None):


        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        
        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, batch.spacegroup)

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        rand_t = torch.randn_like(gt_atom_types_onehot)[batch.anchor_index]
        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        atom_type_probs = atom_type_probs[batch.anchor_index]

        pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_crys_fam, batch.num_atoms, batch.batch)
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, batch.spacegroup)

        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(sigmas_per_atom * rand_x_anchor, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)
        loss_coord = F.mse_loss(pred_x_proj, tar_x_anchor)
        loss_type = F.mse_loss(pred_t, rand_t)


        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type
        }

    @torch.no_grad()
    def sample(self, batch, diff_ratio = 1.0, step_lr = 1e-5):


        batch_size = batch.num_graphs

        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        crys_fam_T = torch.randn([batch_size, 6]).to(self.device)
        crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)


        time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = torch.cat([x_T[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_T.device)], dim=-1).unsqueeze(-1) # N * 4 * 1

        x_T = (batch.ops @ x_T_all).squeeze(-1)[:,:3] % 1. # N * 3

        t_T = t_T[batch.anchor_index]

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T,
            'crys_fam': crys_fam_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)


            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            crys_fam_t = traj[t]['crys_fam']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, t_t, x_t, crys_fam_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]

            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1) 

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            frac_coords_all = torch.cat([x_t_minus_05[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_t_minus_05.device)], dim=-1).unsqueeze(-1) # N * 4 * 1

            x_t_minus_05 = (batch.ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3

            t_t_minus_05 = t_t

            # Predictor

            rand_crys_fam = torch.randn_like(crys_fam_T)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
            ori_crys_fam = crys_fam_t_minus_05
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            rand_t = rand_t[batch.anchor_index]

            pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, crys_fam_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            
            crys_fam_t_minus_1 = c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(crys_fam_t_minus_1, batch.spacegroup)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]
            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1) 

            pred_t = scatter(pred_t, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]


            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)


            frac_coords_all = torch.cat([x_t_minus_1[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_t_minus_1.device)], dim=-1).unsqueeze(-1) # N * 4 * 1

            x_t_minus_1 = (batch.ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3

            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            t_t_minus_1 = t_t_minus_1[batch.anchor_index]


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'crys_fam': crys_fam_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = dc(traj[0])
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return res, traj_stack

    def on_after_backward(self):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here

        total_norm = 0.
        for nm,p in self.decoder.named_parameters():
            try:
                param_norm = p.grad.data.norm(2)
                total_norm = total_norm + param_norm.item() ** 2
            except:
                pass
        total_norm = total_norm ** (1. / 2)

        self.log_dict({
            'grad_norm':total_norm
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'type_loss': loss_type},
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
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
        }

        return log_dict, loss
