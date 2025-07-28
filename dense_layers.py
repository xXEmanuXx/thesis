import torch.nn as nn

import utils

def make_layers(sizes: list[int], *, p_drop: float, negative_slope: float):
    blocks = []
    for in_f, out_f in zip(sizes, sizes[1:]):
        blocks += [
            nn.Linear(in_f, out_f, device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p_drop),
        ]

    return nn.Sequential(*blocks)
