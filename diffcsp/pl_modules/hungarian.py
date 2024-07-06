# Hungarian algorithm
from functools import partial

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

# from torch_linear_assignment import batch_linear_assignment


def cdist_mic(f0, f1, p=2):
    """cdist with mic

    mic dist: d = |w(f1 - f0 - 0.5) - 0.5|^2

    Parameters
    ----------
    f0 : torch.tensor
        Batch fractional coordiantes x0, shape (P,d)
    f1 : torch.tensor
        Batch fractional coordinates x1, shape (R,d)
    """
    f0 = f0[:, None, :]  # (P,1,d)
    f1 = f1[None, :, :]  # (1,R,d)
    d = (f1 - f0 - 0.5) % 1 - 0.5  # (P,R,d)
    d = torch.norm(d, p, dim=-1)  # (P,R)
    return d


class HungarianMatcher(nn.Module):
    """Permute (x1,x2) to (x1,x2') with hungarian algo"""

    def __init__(self, dist_algo: str = "norm", p=2):
        super().__init__()
        if dist_algo == "norm":
            self.dist_func = partial(torch.cdist, p=p)
        elif dist_algo == "norm_mic":
            self.dist_func = partial(cdist_mic, p=p)

    @torch.no_grad()
    def forward(self, x1: torch.tensor, x2: torch.tensor):
        x1in = x1
        x1shape = x1.shape
        x2shape = x2.shape
        if x1.dim() > 2:
            x1 = x1.reshape(x1shape[0], -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2shape[0], -1)
        cost_matrix = self.dist_func(x1, x2)
        _, col_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
        # col_idx = batch_linear_assignment(cost_matrix[None, :, :])[0]
        return x1in, x2[col_idx].reshape(x2shape).contiguous()
