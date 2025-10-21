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
from dense_layers import DenseLayer

class Encoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, fc_dims: list[int], latent_dim: int, dropout: float, negative_slope: float) -> None:
        super().__init__()

        self.metapathway = MetapathwayEncoder(metapathway_mask.size(1), metapathway_mask.size(0), metapathway_mask, negative_slope=negative_slope)
        self.pathway = PathwayEncoder(pathway_mask.size(1), pathway_mask.size(0), pathway_mask, negative_slope=negative_slope)

        dims = [pathway_mask.size(0), *fc_dims, latent_dim]
        self.fcs = nn.ModuleList([DenseLayer(dims[i], dims[i + 1], dropout, negative_slope) for i in range(len(dims) - 1)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.metapathway(x)
        x = self.pathway(x)
        for block in self.fcs:
            x = block(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, metapathway_mask: torch.Tensor, pathway_mask: torch.Tensor, fc_dims: list[int], latent_dim: int,  dropout: float, negative_slope: float) -> None:
        super().__init__()

        dims = [latent_dim, *reversed(fc_dims), pathway_mask.size(1)]
        self.fcs = nn.ModuleList([DenseLayer(dims[i], dims[i + 1], dropout, negative_slope) for i in range(len(dims) - 1)])

        self.pathway = PathwayDecoder(pathway_mask.size(1), pathway_mask.size(0), pathway_mask, negative_slope=negative_slope)
        self.metapathway = MetapathwayDecoder(metapathway_mask.size(1), metapathway_mask.size(0), metapathway_mask, negative_slope=negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.fcs:
            x = block(x)
        x = self.pathway(x)
        x = self.metapathway(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self, *,
                 metapathway_mask: torch.Tensor,
                 pathway_mask: torch.Tensor,
                 fc_dims: list[int],
                 latent_dim: int,
                 dropout: float,
                 negative_slope: float) -> None:
        
        super().__init__()
        self.encoder = Encoder(metapathway_mask, pathway_mask, fc_dims, latent_dim, dropout, negative_slope)
        self.decoder = Decoder(metapathway_mask, pathway_mask.T, fc_dims, latent_dim, dropout, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x
