from __future__ import annotations

import torch
import torch.nn as nn

from encoder.first_layer import MetapathwayGraphEncoder
from encoder.second_layer import MetapathwayToPathwayEncoder

from decoder.first_layer import PathwayToMetapathwayDecoder
from decoder.second_layer import MetapathwayGraphDecoder

class Encoder(nn.Module):
    def __init__(self, graph_mask: torch.Tensor, latent_mask: torch.Tensor) -> None:
        super().__init__()
        self.first_layer = MetapathwayGraphEncoder(graph_mask.size(1), graph_mask.size(0), graph_mask)
        self.second_layer = MetapathwayToPathwayEncoder(latent_mask.size(1), latent_mask.size(0), latent_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, graph_mask: torch.Tensor, latent_mask: torch.Tensor) -> None:
        super().__init__()
        self.first_layer = PathwayToMetapathwayDecoder(latent_mask.size(1), latent_mask.size(0), latent_mask)
        self.second_layer = MetapathwayGraphDecoder(graph_mask.size(1), graph_mask.size(0), graph_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, graph_mask: torch.Tensor, latent_mask: torch.Tensor) -> None:
        super().__init__()
        self.encoder = Encoder(graph_mask, latent_mask)
        self.decoder = Decoder(graph_mask, latent_mask.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x