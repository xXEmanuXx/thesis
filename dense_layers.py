import torch.nn as nn
from utils import NEGATIVE_SLOPE

def _block(in_f: int, out_f: int, p_drop: float = 0.1):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
        nn.BatchNorm1d(out_f),
        nn.Dropout(p_drop),
    )

class DenseEncoder(nn.Sequential):
    """Pathway → bottleneck"""
    def __init__(self, pathway_dim: int, hidden: int = 1024, bottleneck: int = 256):
        super().__init__(
            _block(pathway_dim, hidden),
            _block(hidden, bottleneck, p_drop=0.05)   # bottleneck
        )
        self.out_features = bottleneck          # ci serve nel decoder

class DenseDecoder(nn.Sequential):
    """bottleneck → Pathway"""
    def __init__(self, pathway_dim: int, hidden: int = 1024, bottleneck: int = 256):
        super().__init__(
            _block(bottleneck, hidden),
            _block(hidden, pathway_dim, p_drop=0.0)
        )
