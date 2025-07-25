import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import load_idx_src, load_idx_tgt, load_idx_pathway, load_idx_in

from utils import METAPATHWAY_NODES, NEGATIVE_SLOPE, DEFAULT_DEVICE, DEFAULT_DTYPE

class MetapathwayGraphDecoder(nn.Linear):
    mask: torch.Tensor
    idx_in: torch.Tensor
    idx_src: torch.Tensor
    idx_tgt: torch.Tensor
    idx_pathway: torch.Tensor
    scratch: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            self.bias.zero_()

        self.register_buffer("idx_pathway", load_idx_pathway())
        self.register_buffer("idx_src", load_idx_src())
        self.register_buffer("idx_tgt", load_idx_tgt())
        self.register_buffer("idx_in", load_idx_in())

        self.register_buffer("scratch", torch.empty(0, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BATCH = x.size(0)

        if self.scratch.numel() != BATCH * METAPATHWAY_NODES:
            self.scratch = torch.zeros(BATCH, METAPATHWAY_NODES, dtype=x.dtype, device=x.device)
        else:
            with torch.no_grad():
                self.scratch.zero_()

        x_full = self.scratch.clone()
        x_full.scatter_(1, self.idx_pathway.expand(BATCH, -1), x)

        x_src = x_full[:, self.idx_src]

        z = F.leaky_relu(F.linear(x_src, self.weight * self.mask, self.bias), negative_slope=NEGATIVE_SLOPE)

        x_full.scatter_(1, self.idx_tgt.expand(BATCH, -1), z)

        return x_full[:, self.idx_in]