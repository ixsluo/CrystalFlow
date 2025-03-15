import torch
import torch.nn as nn


class EquiformerV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ln = nn.Linear(3, 3)
