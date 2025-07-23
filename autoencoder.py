import torch.nn as nn

from encoder.first_layer import MetapathwayGraphEncoder
from encoder.second_layer import MetapathwayToPathwayEncoder

from decoder.first_layer import PathwayToMetapathwayDecoder
from decoder.second_layer import MetapathwayGraphDecoder

class Encoder(nn.Module):
    def __init__(self, graph_m, latent_m):
        super().__init__()
        self.first_layer = MetapathwayGraphEncoder(graph_m.size(1), graph_m.size(0), graph_m)
        self.second_layer = MetapathwayToPathwayEncoder(latent_m.size(1), latent_m.size(0), latent_m)
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, graph_m, latent_m):
        super().__init__()
        self.first_layer = PathwayToMetapathwayDecoder(latent_m.size(1), latent_m.size(0), latent_m)
        self.second_layer = MetapathwayGraphDecoder(graph_m.size(1), graph_m.size(0), graph_m)
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, graph_m, latent_m):
        super().__init__()
        self.encoder = Encoder(graph_m, latent_m)
        self.decoder = Decoder(graph_m, latent_m.T)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x