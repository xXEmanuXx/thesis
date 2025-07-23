import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import idx_pathway

from utils import build_weight
from utils import NEGATIVE_SLOPE

class MetapathwayToPathwayEncoder(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)
            self.weight *= self.mask
            #self.weight.zero_()
            #self.weight = build_weight(self.weight, self.mask)
            self.bias.zero_()

        self.register_buffer("idx_pathway",  idx_pathway)

    def forward(self, x):
        x = x[:, self.idx_pathway]
        z = F.leaky_relu(F.linear(x, self.weight * self.mask, self.bias), negative_slope=NEGATIVE_SLOPE)
        return z
