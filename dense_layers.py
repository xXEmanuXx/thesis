import torch.nn as nn

class DenseEncoder(nn.Sequential):
    def __init__(self, pathway_dim: int, p_drop: float, negative_slope: float, mid_dim: int = 2048, bottleneck: int = 1024):
         super().__init__(
            nn.Linear(pathway_dim, mid_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p_drop),
            nn.Linear(mid_dim, bottleneck),
            nn.LeakyReLU(negative_slope)
        )

class DenseDecoder(nn.Sequential):
    def __init__(self, pathway_dim: int, p_drop: float, negative_slope: float, mid_dim: int = 2048, bottleneck: int = 1024):
        super().__init__(
            nn.Linear(bottleneck, mid_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p_drop),
            nn.Linear(mid_dim, pathway_dim),
            nn.LeakyReLU(negative_slope)
        )
