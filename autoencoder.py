from __future__ import annotations

import torch
import torch.nn as nn

from metapathway_layers import MetapathwayEncoder, MetapathwayDecoder
from pathway_layers import PathwayEncoder, PathwayDecoder
from dense_layers import DenseEncoder, DenseDecoder

from utils import NEGATIVE_SLOPE

class Encoder(nn.Module):
    def __init__(self, graph_mask: torch.Tensor, latent_mask: torch.Tensor, hidden=1024, bottleneck=512) -> None:
        super().__init__()
        self.first_layer = MetapathwayEncoder(graph_mask.size(1), graph_mask.size(0), graph_mask)
        self.second_layer = PathwayEncoder(latent_mask.size(1), latent_mask.size(0), latent_mask)
        self.d = DenseEncoder(latent_mask.size(0), hidden=hidden, bottleneck=bottleneck)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.d(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, graph_mask: torch.Tensor, latent_mask: torch.Tensor, hidden=1024, bottleneck=512) -> None:
        super().__init__()
        self.d = DenseDecoder(latent_mask.size(1), hidden=hidden, bottleneck=bottleneck)
        self.first_layer = PathwayDecoder(latent_mask.size(1), latent_mask.size(0), latent_mask)
        self.second_layer = MetapathwayDecoder(graph_mask.size(1), graph_mask.size(0), graph_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d(x)
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