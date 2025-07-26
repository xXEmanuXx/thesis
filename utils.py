"""
Utility helpers

This module centralizes on:
    - global hyper-parameters and paths
    - tensor initialization helpers
    - serialization of model checkpoints

"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, Any
import pandas as pd
import torch
import zstandard as zstd

DEFAULT_DEVICE: torch.device = torch.device('cpu')
DEFAULT_DTYPE: torch.dtype = torch.float32

DATA_PATH: Path = Path('data')
CKPT_PATH: Path = Path('autoencoder.pt')

# Data
METAPATHWAY_NODES = 20507

# Hyper parameters
BATCH_SIZE = 222
NUM_EPOCH = 150
LR_INIT = 1e-3
LR_MAX = 1e-2
NEGATIVE_SLOPE = 0.04 # Leaky ReLU
DELTA_RAW = 700 # 57, 80, 41, 54, 700
DELTA_NORM = 20

def build_mask(src: pd.Series, tgt: pd.Series) -> torch.Tensor:
    """
    Return a Boolean mask (out_features, in_features) for sparse edges.

    Each unique value in **src**/**tgt** becomes in/out feature.
    The mask is `True` where an edge `(src[i], tgt[i])` exists.

    Parameters
    ----------
    src
        Series containing all id nodes that act as a source
    tgt
        Series containing all id nodes that act as a target

    Returns
    -------
    torch.Tensor
        tensor containing `True` values where an edge exists between two nodes

    Notes
    -----
    Returned tensor is placed on `DEFAULT_DEVICE`.
    """

    src_unique = src.drop_duplicates().tolist()
    tgt_unique = tgt.drop_duplicates().tolist()
    idx_src = {node_id: i for i, node_id in enumerate(src_unique)}
    idx_tgt = {node_id: i for i, node_id in enumerate(tgt_unique)}

    rows = [idx_tgt[t] for t in tgt] 
    cols = [idx_src[s] for s in src]

    mask = torch.zeros(len(tgt_unique), len(src_unique), dtype=torch.bool, device=DEFAULT_DEVICE)
    mask[rows, cols] = True

    return mask

def save_model(epoch: int, 
               epoch_loss: float, 
               model_state: dict[str, Any], 
               optimizer_state: dict[str, Any], 
               scheduler_state: dict[str, Any],
               *,
               ckpt_path: Path = CKPT_PATH) -> None:

    """
    Compresses and saves to disk an object which contains information regardin the model training

    Parameters
    ----------
    epoch
        Current training epoch
    epoch_loss
        loss of the current training epoch
    model_state
        dictionary which describes the model state
    optimizer_state
        dictionary which describes the optimizer state
    scheduler_state
        dictionary which describes the scheduler state
    ckpt_path
        Save path of the model
    """

    buf = io.BytesIO()
    torch.save({
        "epoch": epoch,
        "epoch_loss": epoch_loss,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state
    }, buf)

    compressed = zstd.ZstdCompressor().compress(buf.getvalue())

    ckpt_path.write_bytes(compressed)

def load_model(*, map_location: torch.device = DEFAULT_DEVICE, ckpt_path: Path = CKPT_PATH) -> dict[str, Any]:

    """
    Decompresses and loads the model for training

    Parameters
    ----------
    map_location
        location on where to map the model
    ckpt_path
        load path of the model
    """

    compressed = ckpt_path.read_bytes()
    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    buf = io.BytesIO(decompressed)
    ckpt = torch.load(buf, map_location=map_location)

    return ckpt
