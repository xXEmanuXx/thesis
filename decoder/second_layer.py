import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import idx_src, idx_tgt, idx_pathway, idx_in

from utils import build_weight
from utils import METAPATHWAY_NODES, NEGATIVE_SLOPE

class MetapathwayGraphDecoder(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            #self.weight.zero_()
            #self.weight = build_weight(self.weight, self.mask)
            self.bias.zero_()

        self.register_buffer("idx_pathway",  idx_pathway)
        self.register_buffer("idx_src", idx_src)
        self.register_buffer("idx_tgt", idx_tgt)
        self.register_buffer("idx_in", idx_in)

        self.register_buffer("scratch", torch.zeros(0), persistent=False)

    def forward(self, x):
        B = x.size(0)

        if self.scratch.numel() != B * METAPATHWAY_NODES:
            self.scratch = torch.zeros(B, METAPATHWAY_NODES, device=x.device)
        else:
            with torch.no_grad():
                self.scratch.zero_()

        x_full = self.scratch.clone().scatter(1, self.idx_pathway.expand(B, -1), x)

        x_src = x_full[:, self.idx_src]

        z = F.leaky_relu(F.linear(x_src, self.weight * self.mask, self.bias), negative_slope=NEGATIVE_SLOPE)

        x_full = x_full.scatter(1, self.idx_tgt.expand(B, -1), z)

        return x_full[:, self.idx_in]