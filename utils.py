"""
Utility helpers

This module centralizes on:
    - global constants and paths
    - tensor initialization helpers
    - serialization of model checkpoints
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import pandas as pd
import torch
import zstandard as zstd

DEFAULT_DEVICE: torch.device = torch.device("cuda")
DEFAULT_DTYPE: torch.dtype = torch.float32

DATA_DIR = Path("data")
RESULTS_DIR = Path("sweeps")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "results.jsonl"
RESULTS_FILE.touch(exist_ok=True)

SPLIT_FILE = RESULTS_DIR / "split_indices.json"
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15

METAPATHWAY_NODES = 20507

BATCH_SIZE = 200
NUM_EPOCH = 200
EARLY_STOP = 20

def build_mask(src: pd.Series, tgt: pd.Series) -> torch.Tensor:
    """
    Return a boolean mask (out_features, in_features) for sparse edges.

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
               *,
               ckpt_path: Path) -> None:

    """
    Compresses and saves to disk a file which contains information regarding the model training

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
    }, buf, _use_new_zipfile_serialization=False)

    compressed = zstd.ZstdCompressor().compress(buf.getvalue())
    ckpt_path.write_bytes(compressed)

def load_model(*, map_location: torch.device = DEFAULT_DEVICE, ckpt_path: Path) -> dict[str, Any]:
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
