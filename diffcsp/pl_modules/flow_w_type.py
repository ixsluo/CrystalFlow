# Training example:
#
# HYDRA_FULL_ERROR=1 python diffcsp/run.py \
# data=mp_20 model=flow \
# logging.wandb.group=mp_20 expname=flow-test-01

import copy
import math
import logging
from typing import Any, Dict
from functools import partial, partialmethod, wraps
from inspect import getfullargspec
from itertools import pairwise
from operator import itemgetter

import numpy as np
import omegaconf
import lightning as pl
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
from diffcsp.pl_modules.lattice_utils import LatticeDecompNN
from diffcsp.pl_modules.ode_solvers import str_to_solver
from diffcsp.pl_modules.symmetrize import SymmetrizeRotavg
from diffcsp.pl_modules.conditioning import MultiEmbedding
from diffcsp.pl_modules.time_scheduler import TimeScheduler

MAX_ATOMIC_NUM = 100

metriclogger = logging.getLogger("metrics")


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
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "frequency": 5, "monitor": "val_loss"}}


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


class DirectUnsqueezeTime(nn.Module):
    def forward(self, time: torch.Tensor):
        return time.unsqueeze(-1)


class CSPFlow(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.time_dim == 0:
            self.time_dim = 1
            self.time_embedding = DirectUnsqueezeTime()
        else:
            self.time_dim = self.hparams.time_dim
            self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.time_scheduler = TimeScheduler(self.hparams.get("time_scheduler", ""))

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim=self.hparams.latent_dim + self.time_dim,  # 0 + time
            pred_type=True,
            smooth=True,
            _recursive_=False,
        )
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.ot = self.hparams.get("ot", False)
        self.permute_l = HungarianMatcher("norm")
        self.permute_f = HungarianMatcher("norm_mic")
        self.lattice_polar = self.hparams.get("lattice_polar", False)
        self.lattice_polar_sigma = self.hparams.get("lattice_polar_sigma", 1.0)
        self.latticedecompnn = LatticeDecompNN()
        self.from_cubic = self.hparams.get("from_cubic", False)
        self.lattice_teacher_forcing = self.hparams.get("lattice_teacher_forcing", -1)
        self.symmetrize_anchor = self.hparams.get("symmetrize_anchor", False)
        self.symmetrize_rotavg = self.hparams.get("symmetrize_rotavg", False)
        self.post_symmetrize = self.hparams.get("post_symmetrize", True)
        self.symm_rotavg = SymmetrizeRotavg()
        self.use_symmetrize_loss = self.hparams.get("use_symmetrize_loss", False)
        self.cost_sym_lattice = self.hparams.get("cost_sym_lattice", self.hparams.cost_lattice)
        self.cost_sym_coord = self.hparams.get("cost_sym_coord", self.hparams.cost_coord)
        self.guide_threshold = self.hparams.get("guide_threshold", None)
        self.cond_emb = MultiEmbedding(self.hparams.conditions)

        if self.ot:
            hydra.utils.log.info("Using optimal transport")
        if self.lattice_polar:
            hydra.utils.log.info(f"Using lattice polar decomposition with sigma={self.lattice_polar_sigma}")
        if self.from_cubic:
            hydra.utils.log.info("Using cubic lattice")
        if self.lattice_teacher_forcing > 0:
            hydra.utils.log.info(f"Using lattice_teacher_forcing={self.lattice_teacher_forcing}")
        if self.symmetrize_anchor:
            hydra.utils.log.info("Using symmetrize_anchor")
        if self.symmetrize_rotavg:
            hydra.utils.log.info("Using symmetrize_rotavg")
        if self.symmetrize_anchor and self.symmetrize_rotavg:
            raise ValueError("You can only specify one symmetrize method from anchor|rotavg")
        if self.keep_lattice:
            hydra.utils.log.warning(f"cost_lattice={self.hparams.cost_lattice}, setting to keep lattice.")
        if self.keep_coords:
            hydra.utils.log.warning(f"cost_coords={self.hparams.cost_coord}, setting to keep coords.")

        if self.symmetrize_anchor or self.symmetrize_rotavg:
            raise ValueError("Symmetrize is forbidden in ab-init mode.")

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
        l0 = torch.randn([batch_size, 6], device=self.device) * self.lattice_polar_sigma
        l0[:, -1] = l0[:, -1] + 1
        if self.from_cubic:
            l0[:, :5] = 0
        return l0

    def forward(self, batch, guide_threshold):

        batch_size = batch.num_graphs
        times = torch.rand(batch_size, device=self.device)
        times = self.time_scheduler(times)
        time_emb = self.time_embedding(times)

        guide_threshold = self.guide_threshold if guide_threshold is None else guide_threshold
        if guide_threshold is None:
            cemb = None
            guide_indicator = None
        else:
            cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
            guide_indicator = (torch.rand(batch_size, device=self.device) - guide_threshold).heaviside(torch.tensor(1.0))

        # Build time stamp T and 0
        # lattice
        if self.lattice_polar:
            lattices_rep_T = batch.lattice_polar
            lattices_rep_0 = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor:
                lattices_rep_T = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_T, batch.spacegroup)
                lattices_rep_0 = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_0, batch.spacegroup)
            elif self.symmetrize_rotavg:
                lattices_rep_T = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_T, batch.spacegroup)
                lattices_rep_0 = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_0, batch.spacegroup)
            lattices_mat_T = lattice_polar_build_torch(lattices_rep_T)
        else:
            lattices_rep_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            lattices_rep_0 = self.sample_lattice(batch_size)
            if self.symmetrize_anchor:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            elif self.symmetrize_rotavg:
                raise NotImplementedError("symmetrize_rotavg")
            lattices_mat_T = lattices_rep_T
        # lattice teacher forcing
        lattice_teacher_forcing = self.current_epoch < self.lattice_teacher_forcing
        if lattice_teacher_forcing:
            lattices_rep_0 = lattices_rep_T

        # coords
        frac_coords = batch.frac_coords
        f0 = torch.rand_like(frac_coords)
        if self.symmetrize_anchor:
            f0_anchor = f0[batch.anchor_index]
            f0_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index], f0_anchor)
            f0 = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f0_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f0 = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f0,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        # types
        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        rd_atom_types_onehot = torch.randn_like(gt_atom_types_onehot)

        # optimal transport
        if self.ot:
            if self.symmetrize_anchor or self.symmetrize_rotavg:
                raise ValueError("OT is forbidden in symmetrize.")
            _, lattices_rep_0 = self.permute_l(lattices_rep_T, lattices_rep_0)
            _, f0 = self.permute_f(frac_coords, f0)

        # Build time stamp t
        tar_l = lattices_rep_T - lattices_rep_0
        tar_f = (frac_coords - f0 - 0.5) % 1 - 0.5
        tar_t = gt_atom_types_onehot - rd_atom_types_onehot

        # Build input lattice rep/mat and input coords
        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)
        input_lattice_rep = lattices_rep_0 + times[l_expand_dim] * tar_l
        input_frac_coords = f0 + times.repeat_interleave(batch.num_atoms)[:, None] * tar_f
        if self.lattice_polar:
            input_lattice_mat = lattice_polar_build_torch(input_lattice_rep)
        else:
            input_lattice_mat = input_lattice_rep
        input_atom_type_probs = rd_atom_types_onehot + times.repeat_interleave(batch.num_atoms)[:, None] * tar_t

        # Replace inputs if fixed
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice_rep = lattices_rep_T
            input_lattice_mat = lattices_mat_T

        # Flow
        pred_l, pred_f, pred_t = self.decoder(
            t=time_emb,
            atom_types=input_atom_type_probs,
            frac_coords=input_frac_coords,
            lattices_rep=input_lattice_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=input_lattice_mat,
            cemb=cemb, guide_indicator=guide_indicator,
        )

        loss_sym_l = 0.0
        loss_sym_f = 0.0
        if self.post_symmetrize and self.symmetrize_anchor:
            if self.lattice_polar:
                pred_l_symmetrized = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            tar_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, tar_f)
            pred_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, pred_f)
            pred_f_symmetrized = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], pred_f_anchor)
            if self.use_symmetrize_loss:
                loss_sym_l = F.mse_loss(pred_l, pred_l_symmetrized)
                loss_sym_f = F.mse_loss(pred_f, pred_f_symmetrized)
            # for loss
            pred_l = pred_l_symmetrized
            tar_f = tar_f_anchor
            pred_f = pred_f_anchor
        elif self.post_symmetrize and self.symmetrize_rotavg:
            if self.lattice_polar:
                pred_l_symmetrized = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_f_symmetrized = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=pred_f,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            )
            if self.use_symmetrize_loss:
                loss_sym_l = F.mse_loss(pred_l, pred_l_symmetrized)
                loss_sym_f = F.mse_loss(pred_f, pred_f_symmetrized)
            pred_l = pred_l_symmetrized
            pred_f = pred_f_symmetrized

        loss_lattice = F.mse_loss(pred_l, tar_l)
        loss_coord = F.mse_loss(pred_f, tar_f)
        loss_type = F.mse_loss(pred_t, tar_t)

        cost_coord = self.hparams.cost_coord
        cost_lattice = 0.0 if lattice_teacher_forcing else self.hparams.cost_lattice
        cost_type = self.hparams.cost_type
        loss = (
            cost_lattice * loss_lattice
            + cost_coord * loss_coord
            + cost_type * loss_type
            + self.cost_sym_lattice * loss_sym_l
            + self.cost_sym_coord * loss_sym_f
        )

        return {
            'loss': loss,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_type': loss_type,
            'loss_sym_lattice': loss_sym_l,
            'loss_sym_coord': loss_sym_f,
        }

    @staticmethod
    def get_anneal_factor(t, slope: float = 0.0, offset: float = 0.0):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return 1 + slope * F.relu(t - offset)

    def post_decoder_on_sample(
        self, pred_l, pred_f, pred_t,
        batch, t,
        anneal_lattice=False, anneal_coords=False, anneal_type=False,
        anneal_slope=0.0, anneal_offset=0.0,
    ):
        if self.symmetrize_anchor:
            if self.lattice_polar:
                pred_l = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, pred_f)
            pred_f_anchor = scatter(pred_f_anchor, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]
            pred_f = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], pred_f_anchor)
        elif self.symmetrize_rotavg:
            if self.lattice_polar:
                pred_l = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_f = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=pred_f,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            )
        anneal_factor = self.get_anneal_factor(t, anneal_slope, anneal_offset)
        if anneal_lattice:
            pred_l *= anneal_factor
        if anneal_coords:
            pred_f *= anneal_factor
        if anneal_type:
            pred_t *= anneal_factor
        return pred_l, pred_f, pred_t

    @torch.no_grad()
    def sample(
        self, batch, step_lr=None, N=None,
        anneal_lattice=False, anneal_coords=False, anneal_type=False,
        anneal_slope=0.0, anneal_offset=0.0,
        guide_threshold=None,
        **kwargs,
    ):
        if N is None:
            N = round(1 / step_lr)

        batch_size = batch.num_graphs

        if guide_threshold is not None:
            cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
            guide_indicator = torch.ones(batch_size, device=self.device)

        # time stamp T
        if self.lattice_polar:
            l_T = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor:
                l_T = self.latticedecompnn.proj_k_to_spacegroup(l_T, batch.spacegroup)
            elif self.symmetrize_rotavg:
                l_T = self.latticedecompnn.proj_k_to_spacegroup(l_T, batch.spacegroup)
            lattices_mat_T = lattice_polar_build_torch(l_T)
        else:
            l_T = self.sample_lattice(batch_size)
            if self.symmetrize_anchor:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            elif self.symmetrize_rotavg:
                raise NotImplementedError("symmetrize_rotavg")
            lattices_mat_T = l_T

        f_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.symmetrize_anchor:
            f_T_anchor = f_T[batch.anchor_index]
            f_T_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index, :3, :3], f_T_anchor)
            f_T = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f_T_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f_T = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f_T,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        rd_atom_types_onehot = torch.randn((batch.num_nodes, MAX_ATOMIC_NUM), device=self.device)

        #
        if self.keep_coords:
            f_T = batch.frac_coords
        if self.keep_lattice:
            lattices_mat_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            if self.lattice_polar:
                l_T = lattice_polar_decompose_torch(lattices_mat_T)
            else:
                l_T = lattices_mat_T

        traj = {
            0: {
                'num_atoms': batch.num_atoms,
                'atom_types': torch.argmax(rd_atom_types_onehot, dim=-1) + 1,
                'frac_coords': f_T % 1.0,
                'lattices': lattices_mat_T,
            }
        }

        lattices_mat_t = lattices_mat_T.clone().detach()
        l_t = l_T.clone().detach()
        f_t = f_T.clone().detach()
        t_t = rd_atom_types_onehot.clone().detach()

        for t in tqdm(range(1, N + 1)):

            times = torch.full((batch_size,), t, device=self.device) / N
            time_emb = self.time_embedding(times)

            if self.keep_coords:
                f_t = f_T
            if self.keep_lattice:
                l_t = l_T
                lattices_mat_t = lattices_mat_T

            pred_l, pred_f, pred_t = self.decoder(
                t=time_emb,
                atom_types=t_t,
                frac_coords=f_t,
                lattices_rep=l_t,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat_t,
                cemb=None, guide_indicator=None,
            )
            pred_l, pred_f, pred_t = self.post_decoder_on_sample(
                pred_l, pred_f, pred_t,
                batch=batch, t=t,
                anneal_lattice=anneal_lattice, anneal_coords=anneal_coords, anneal_type=anneal_type,
                anneal_slope=anneal_slope, anneal_offset=anneal_offset,
            )
            if guide_threshold is not None:
                pred_l_guide, pred_f_guide, pred_t_guide = self.decoder(
                    t=time_emb,
                    atom_types=batch.atom_types,
                    frac_coords=f_t,
                    lattices_rep=l_t,
                    num_atoms=batch.num_atoms,
                    node2graph=batch.batch,
                    lattices_mat=lattices_mat_t,
                    cemb=cemb, guide_indicator=guide_indicator,
                )
                pred_l_guide, pred_f_guide, pred_t_guide = self.post_decoder_on_sample(
                    pred_l_guide, pred_f_guide, pred_t_guide,
                    batch=batch, t=t,
                    anneal_lattice=anneal_lattice, anneal_coords=anneal_coords,
                    anneal_slope=anneal_slope, anneal_offset=anneal_offset,
                )
                pred_l = (1 - guide_threshold) * pred_l_guide + guide_threshold * pred_l
                pred_f = (1 - guide_threshold) * pred_f_guide + guide_threshold * pred_f
                pred_t = (1 - guide_threshold) * pred_t_guide + guide_threshold * pred_t


            l_t = l_t + pred_l / N if not self.keep_lattice else l_t
            f_t = f_t + pred_f / N if not self.keep_coords else f_t
            f_t = f_t % 1.0
            t_t = t_t + pred_t / N

            if self.lattice_polar:
                lattices_mat_t = lattice_polar_build_torch(l_t)
            else:
                lattices_mat_t = l_t

            traj[t] = {
                'num_atoms': batch.num_atoms,
                'atom_types': torch.argmax(t_t, dim=-1) + 1,
                'frac_coords': f_t,
                'lattices': lattices_mat_t,
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': torch.stack([traj[i]['atom_types'] for i in range(0, N + 1)]),
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(0, N + 1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(0, N + 1)]),
        }

        return traj[N], traj_stack

    def single_time_decoder(self, t, **kwargs):
        batch_size = kwargs["num_atoms"].shape[0]
        time_emb = self.time_embedding(t.repeat(batch_size))
        pred_l, pred_x = self.decoder(
            t=time_emb,
            **kwargs,
        )
        return pred_l, pred_x

    def _resign_lattice_decoder(self, t, x, **kwargs):
        return self.single_time_decoder(t=t, lattices_rep=x, **kwargs)

    def _resign_coords_decoder(self, t, x, **kwargs):
        return self.single_time_decoder(t=t, frac_coords=x, **kwargs)

    # return func with args [t, x] only
    def _partial_lattice_decoder(self, **kwargs):
        assert ("lattices_rep" not in kwargs) and ("num_atoms" in kwargs)

        def return_first(func):
            def wrapper(t, x):
                return func(t, x)[0]
            return wrapper

        f = return_first(partial(self._resign_lattice_decoder, **kwargs))
        assert getfullargspec(f).args == ["t", "x"]
        return f

    # return func with args [t, x] only
    def _partial_coords_decoder(self, **kwargs):
        assert ("frac_coords" not in kwargs) and ("num_atoms" in kwargs)

        def return_second(func):
            def wrapper(t, x):
                return func(t, x)[1]
            return wrapper

        f = return_second(partial(self._resign_coords_decoder, **kwargs))
        assert getfullargspec(f).args == ["t", "x"]
        return f

    def _fixed_odeint(
        self, batch, t_span, solver, integrate_sequence="lattice_first",
        anneal_lattice=False, anneal_coords=False,
        anneal_slope=0.0, anneal_offset=0.0,
        **kwargs,
    ):

        assert solver.stepping_class == "fixed"

        batch_size = batch.num_graphs

        if self.lattice_polar:
            l_t = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor:
                l_t = self.latticedecompnn.proj_k_to_spacegroup(l_t, batch.spacegroup)
            elif self.symmetrize_rotavg:
                l_t = self.latticedecompnn.proj_k_to_spacegroup(l_t, batch.spacegroup)
            lattices_mat_t = lattice_polar_build_torch(l_t)
        else:
            l_t = self.sample_lattice(batch_size)
            if self.symmetrize_anchor:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            elif self.symmetrize_rotavg:
                raise NotImplementedError("symmetrize_rotavg")
            lattices_mat_t = l_t

        f_t = torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.symmetrize_anchor:
            f_t_anchor = f_t[batch.anchor_index]
            f_t_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index, :3, :3], f_t_anchor)
            f_t = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f_t_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f_t = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f_t,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        traj = {
            0: {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': f_t.clone().detach() % 1.0,
                'lattices': lattices_mat_t.clone().detach(),
            }
        }

        """Solves IVPs with same `t_span`, using fixed-step methods"""
        _, T, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]
        pbar = tqdm(ncols=79, total=len(t_span) - 1)  # note: ignore first zero
        for steps, (_t, t) in enumerate(pairwise(t_span), 1):  # note: start from second
            dt = t - _t

            pred_l, pred_f = self.single_time_decoder(
                t=t,
                frac_coords=f_t,
                lattices_rep=l_t,
                lattices_mat=lattices_mat_t,
                atom_types=batch.atom_types,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
            )

            time_emb = self.time_embedding(t.repeat(batch_size))
            pred_l, pred_f = self.post_decoder_on_sample(
                pred_l, pred_f,
                batch=batch, t=t,
                anneal_lattice=anneal_lattice, anneal_coords=anneal_coords,
                anneal_slope=anneal_slope, anneal_offset=anneal_offset,
            )

            vf_coords = self._partial_coords_decoder(
                lattices_rep=l_t,
                atom_types=batch.atom_types,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat_t,
            )
            vf_lattice = self._partial_lattice_decoder(
                frac_coords=f_t,
                atom_types=batch.atom_types,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat_t,
            )
            if integrate_sequence == "lattice_first":
                _, l_t, _ = solver.step(f=vf_lattice, x=l_t, t=t, dt=dt, k1=pred_l)
                _, f_t, _ = solver.step(f=vf_coords, x=f_t, t=t, dt=dt, k1=pred_f)
            elif integrate_sequence == "coords_first":
                _, f_t, _ = solver.step(f=vf_coords, x=f_t, t=t, dt=dt, k1=pred_f)
                _, l_t, _ = solver.step(f=vf_lattice, x=l_t, t=t, dt=dt, k1=pred_l)
            else:
                raise NotImplementedError("Unknown ode sequence")

            f_t: torch.Tensor = f_t % 1.0
            if self.lattice_polar:
                lattices_mat_t = lattice_polar_build_torch(l_t)
            else:
                lattices_mat_t = l_t
            traj[steps] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': f_t.clone().detach(),
                'lattices': lattices_mat_t.clone().detach(),
            }

            pbar.update()

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'all_frac_coords': torch.stack([t['frac_coords'] for t in traj.values()]),
            'all_lattices': torch.stack([t['lattices'] for t in traj.values()]),
        }

        return traj[list(traj)[-1]], traj_stack

    @torch.no_grad()
    def sample_ode(
        self, batch, t_span, solver, integrate_sequence="lattice_first",
        anneal_lattice=False, anneal_coords=False,
        anneal_slope=0.0, anneal_offset=0.0,
        **kwargs,
    ):
        raise NotImplementedError("w_type ODE is not implemented")
        t_span = t_span.to(self.device)
        solver = str_to_solver(solver)
        if solver.stepping_class == "fixed":
            return self._fixed_odeint(
                batch, t_span, solver, integrate_sequence,
                anneal_lattice, anneal_coords,
                anneal_slope, anneal_offset,
                **kwargs,
            )
        else:
            raise NotImplementedError("stepping class except fixed is not accepted.")

    def training_step(self, batch, batch_idx: int):

        output_dict = self(batch)

        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': output_dict['loss_lattice'],
                'coord_loss': output_dict['loss_coord'],
                'type_loss': output_dict['loss_type'],
                'sym_lattice_loss': output_dict['loss_sym_lattice'],
                'sym_coord_loss': output_dict['loss_sym_coord'],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        if loss.isnan():
            raise RuntimeError("loss is nan!")
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
            batch_size=batch.num_graphs,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
            batch_size=batch.num_graphs,
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        loss = output_dict['loss']
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': output_dict['loss_lattice'],
            f'{prefix}_coord_loss': output_dict['loss_coord'],
            f'{prefix}_type_loss': output_dict['loss_type'],
            f'{prefix}_sym_lattice_loss': output_dict['loss_sym_lattice'],
            f'{prefix}_sym_coord_loss': output_dict['loss_sym_coord'],
        }

        return log_dict, loss

    def on_train_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_validation_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_test_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")
