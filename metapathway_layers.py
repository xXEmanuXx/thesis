"""
Metapathway-level encoder and decoder

This module implement both the metapathway layer encoder and decoder described by the `metapathway_edges_simplified_2025.tsv` file

**Encoder**
    It projects the filtered input nodes into the metapathway and
    applies the masked weight matrix created using the edges (source -> target) specified in the file.
    The output of the layer is then scattered inside the metapathway nodes tensor

**Decoder**
    It projects the "input" nodes from the previous layer into the metapathway tensor and
    applies the same masked weight matrix of the file.
    The output is again scattered inside the metapathway nodes tensor and lastly we filter it by taking the nodes that correspond to the reconstructed input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_loader
import utils

class MetapathwayEncoder(nn.Linear):
    mask: torch.Tensor
    idx_in: torch.Tensor
    idx_src: torch.Tensor
    idx_tgt: torch.Tensor
    scratch: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, negative_slope: float, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE)
        
        self.register_buffer("mask", mask)
        self.register_buffer("idx_in", data_loader.load_idx("idx_in"))
        self.register_buffer("idx_src", data_loader.load_idx("idx_src"))
        self.register_buffer("idx_tgt", data_loader.load_idx("idx_tgt"))

        self.register_buffer("scratch", torch.empty(0, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE), persistent=False)

        self.negative_slope = negative_slope

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BATCH = x.size(0)

        if self.scratch.numel() != BATCH * utils.METAPATHWAY_NODES:
            self.scratch = torch.zeros(BATCH, utils.METAPATHWAY_NODES, dtype=x.dtype, device=x.device)
        else:
            with torch.no_grad():
                self.scratch.zero_()

        # Place 15k input features inside the 20507 feature metapathway
        x_full = self.scratch.clone()
        x_full.scatter_(1, self.idx_in.expand(BATCH, -1), x)

        x_src = x_full[:, self.idx_src]

        z = F.leaky_relu(F.linear(x_src, self.weight * self.mask, self.bias), negative_slope=self.negative_slope)

        x_full.scatter_(1, self.idx_tgt.expand(BATCH, -1), z)

        return x_full

class MetapathwayDecoder(nn.Linear):
    mask: torch.Tensor
    idx_in: torch.Tensor
    idx_src: torch.Tensor
    idx_tgt: torch.Tensor
    idx_pathway: torch.Tensor
    scratch: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor,  negative_slope: float, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE)
        self.register_buffer("mask", mask)

        self.register_buffer("idx_pathway", data_loader.load_idx("idx_pathway"))
        self.register_buffer("idx_src", data_loader.load_idx("idx_src"))
        self.register_buffer("idx_tgt", data_loader.load_idx("idx_tgt"))
        self.register_buffer("idx_in", data_loader.load_idx("idx_in"))

        self.register_buffer("scratch", torch.empty(0, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE), persistent=False)

        self.negative_slope = negative_slope
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BATCH = x.size(0)

        if self.scratch.numel() != BATCH * utils.METAPATHWAY_NODES:
            self.scratch = torch.zeros(BATCH, utils.METAPATHWAY_NODES, dtype=x.dtype, device=x.device)
        else:
            with torch.no_grad():
                self.scratch.zero_()

        x_full = self.scratch.clone()
        x_full.scatter_(1, self.idx_pathway.expand(BATCH, -1), x)

        x_src = x_full[:, self.idx_src]

        z = F.leaky_relu(F.linear(x_src, self.weight * self.mask, self.bias), negative_slope=self.negative_slope)

        x_full.scatter_(1, self.idx_tgt.expand(BATCH, -1), z)

        return x_full[:, self.idx_in]
