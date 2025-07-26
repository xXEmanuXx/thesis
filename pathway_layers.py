import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import load_idx_pathway

from utils import NEGATIVE_SLOPE

class PathwayEncoder(nn.Linear):
    mask: torch.Tensor
    idx_pathway: torch.Tensor
    scratch: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()

        self.register_buffer("idx_pathway",  load_idx_pathway())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.idx_pathway]
        z = F.leaky_relu(F.linear(x, self.weight * self.mask, self.bias), negative_slope=NEGATIVE_SLOPE)
        return z
    
class PathwayDecoder(nn.Linear):
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(F.linear(x, self.weight * self.mask, self.bias), negative_slope=NEGATIVE_SLOPE)
        return z
