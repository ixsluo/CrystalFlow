import torch
import torch.nn as nn


class TimeScheduler(nn.Module):
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        if self.scheduler not in ["reverse-squared", ""]:
            raise ValueError(f"Unknown time scheduler: {self.scheduler}")
        super().__init__()

    def forward(self, t):
        if self.scheduler == "reverse-squared":
            return 1 - torch.square(t)
        elif self.scheduler == "":
            return t
        else:
            raise ValueError(f"Unknown time scheduler: {self.scheduler}")
