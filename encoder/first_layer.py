import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import idx_in, idx_src, idx_tgt

from utils import build_weight
from utils import METAPATHWAY_NODES, NEGATIVE_SLOPE

class MetapathwayGraphEncoder(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True):
        super().__init__(in_features, out_features, bias) # Initialize weight matrix and bias vector
        self.register_buffer("mask", mask) # Mask specifying which edges are effectively there

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            #self.weight.zero_()
            #self.weight = build_weight(self.weight, self.mask)
            self.bias.zero_()

        # Buffers used to filter and scatter metapathway layer
        self.register_buffer("idx_in", idx_in)
        self.register_buffer("idx_src", idx_src)
        self.register_buffer("idx_tgt", idx_tgt)

        self.register_buffer("scratch", torch.zeros(0), persistent=False)

    def forward(self, x):
        B = x.size(0)

        if self.scratch.numel() != B * METAPATHWAY_NODES:
            self.scratch = torch.zeros(B, METAPATHWAY_NODES, device=x.device)
        else:
            with torch.no_grad():
                self.scratch.zero_()

        # Place 15k input features inside the 20507 feature metapathway
        x_full = self.scratch.clone().scatter(1, self.idx_in.expand(B, -1), x)

        x_src = x_full[:, self.idx_src]

        z = F.leaky_relu(F.linear(x_src, self.weight * self.mask, self.bias), negative_slope=NEGATIVE_SLOPE)

        x_full = x_full.scatter(1, self.idx_tgt.expand(B, -1), z)

        return x_full