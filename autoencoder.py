from __future__ import annotations

import torch
import torch.nn as nn

from metapathway_layers import MetapathwayEncoder, MetapathwayDecoder
from pathway_layers import PathwayEncoder, PathwayDecoder
from dense_layers import DenseEncoder, DenseDecoder

class Encoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, dropout: float, negative_slope: float) -> None:
        super().__init__()
        self.first_layer = MetapathwayEncoder(metapathway_mask.size(1), metapathway_mask.size(0), metapathway_mask, negative_slope=negative_slope)
        self.second_layer = PathwayEncoder(pathway_mask.size(1), pathway_mask.size(0), pathway_mask, negative_slope=negative_slope)
        self.d1 = DenseEncoder(pathway_mask.size(0), p_drop=dropout, negative_slope=negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.second_layer(x)
        #x = self.d1(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, dropout: float, negative_slope: float) -> None:
        super().__init__()
        self.d1 = DenseDecoder(pathway_mask.size(1), p_drop=dropout, negative_slope=negative_slope)
        self.first_layer = PathwayDecoder(pathway_mask.size(1), pathway_mask.size(0), pathway_mask, negative_slope=negative_slope)
        self.second_layer = MetapathwayDecoder(metapathway_mask.size(1), metapathway_mask.size(0), metapathway_mask, negative_slope=negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.d1(x)
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, dropout: float, negative_slope: float) -> None:
        super().__init__()
        self.encoder = Encoder(metapathway_mask, pathway_mask, dropout, negative_slope)
        self.decoder = Decoder(metapathway_mask, pathway_mask.T, dropout, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x