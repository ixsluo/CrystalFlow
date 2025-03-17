import torch
import torch.nn as nn


class PropertyEmbedding(nn.Module):
    def __init__(
        self,
        name,
        scaler,
    ):
        super().__init__()
        self.name = name
        self.scaler = scaler
