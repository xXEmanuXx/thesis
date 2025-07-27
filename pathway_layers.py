"""
Pathway-level encoder and decoder

This module implement both the encoder and decoder layer described by the `metapathway_to_pathways_2025.tsv` file

**Encoder**
    It filters the metapathway nodes tensor taking only the nodes the correspond to one or more pathways and
    applies the masked weight matrix created using the file.
    
**Decoder**
    It simply makes the inverse process of applying the same but transposed masked weight matrix to go from pathways to metapathway nodes 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import load_idx_pathway

class PathwayEncoder(nn.Linear):
    mask: torch.Tensor
    idx_pathway: torch.Tensor
    scratch: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask, negative_slope: float, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)
        self.register_buffer("idx_pathway",  load_idx_pathway())

        self.negative_slope = negative_slope

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.idx_pathway]
        z = F.leaky_relu(F.linear(x, self.weight * self.mask, self.bias), negative_slope=self.negative_slope)
        return z
    
class PathwayDecoder(nn.Linear):
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, negative_slope: float, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

        self.negative_slope = negative_slope

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(F.linear(x, self.weight * self.mask, self.bias), negative_slope=self.negative_slope)
        return z
