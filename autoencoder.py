"""
This module describes the definition of the autoencoder

**AutoEncoder**
    Combines Encoder and Decoder classes
**Encoder**
    Metapathway (edges simplified) -> pathway (nodes_to_pathways) -> dense intermediate layers -> latent space
**Decoder**
    latent space -> dense intermediate layers -> pathway (pathways_to_nodes mirrored from encoder) -> Metapathway (edges simplified)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from metapathway_layers import MetapathwayEncoder, MetapathwayDecoder
from pathway_layers import PathwayEncoder, PathwayDecoder
from dense_layers import make_layers

class Encoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, dropout: float, negative_slope: float) -> None:
        super().__init__()

        self.layers = [pathway_mask.size(0), 1024, 512, 256]

        self.metapathway = MetapathwayEncoder(metapathway_mask.size(1), metapathway_mask.size(0), metapathway_mask, negative_slope=negative_slope)
        self.pathway = PathwayEncoder(pathway_mask.size(1), pathway_mask.size(0), pathway_mask, negative_slope=negative_slope)
        self.dense = make_layers(self.layers, p_drop=dropout, negative_slope=negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.metapathway(x)
        x = self.pathway(x)
        x = self.dense(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, dropout: float, negative_slope: float) -> None:
        super().__init__()

        self.layers = [256, 512, 1024, pathway_mask.size(1)]

        self.dense = make_layers(self.layers, p_drop=dropout, negative_slope=negative_slope)
        self.pathway = PathwayDecoder(pathway_mask.size(1), pathway_mask.size(0), pathway_mask, negative_slope=negative_slope)
        self.metapathway = MetapathwayDecoder(metapathway_mask.size(1), metapathway_mask.size(0), metapathway_mask, negative_slope=negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.pathway(x)
        x = self.metapathway(x)

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
