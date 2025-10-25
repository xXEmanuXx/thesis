from torch import nn

import utils

class DenseLayer(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, negative_slope: float) -> None:
        layer = [
            nn.Linear(in_dim, out_dim, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout)
        ]

        super().__init__(*layer)
