"""
Data loading and preprocessing helpers

This modules deals with filtering an load the input data used in the autoencoder
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import functools
import json
import secrets

from pathlib import Path
from typing import Sequence, Tuple

import utils

def filter_input_df(df: pd.DataFrame, filter_df: pd.DataFrame, filter_id_col: str) -> pd.DataFrame:
    """
    Filters a dataframe keeping only rows whose *index* is present in `filter_df[filter_id_col]`

    Parameters
    ----------
    df 
        DataFrame to be filtered. Its index must contain node identifiers that match values from filter_df[filter_id_col]
    filter_df
        DataFrame providing the whitelist of node ids
    filter_id_col
        Column in *filter_df* that lists the node ids to keep

    Returns
    -------
    pd.DataFrame
        Filtered view of *df*
    """
    
    input_nodes = df.index.values.astype(str).tolist()
    filter_nodes = filter_df[filter_id_col].tolist()

    filtered_input_nodes = [int(node) for node in input_nodes if node in filter_nodes]

    return df.loc[filtered_input_nodes]

def build_index_tensor(subset_ids: Sequence[str | int], 
                       universe_ids: Sequence[str | int], 
                       *, 
                       dtype: torch.dtype, 
                       device: torch.device) -> torch.Tensor:
    
    """
    Map `subset_ids` to their integer positions inside `universe_ids`.

    This helper is the backbone for the four canonical index tensors used by the
    sparse auto-encoder layers.

    Parameters
    ----------
    subset_ids
        Identifiers to map (e.g. tumour sample ids).
    universe_ids
        Reference list containing **all** unique identifiers (e.g. meta-pathway nodes).
    device, dtype
        Standard tensor options.

    Returns
    -------
    torch.Tensor
        1-D tensor with shape `(len(subset_ids),)` whose *i-th* value is the
        position of `subset_ids[i]` inside `universe_ids`.
    """

    lookup = {str(id): i for i, id in enumerate(universe_ids)}
    indices = [lookup[str(id)] for id in subset_ids]

    return torch.tensor(indices, dtype=dtype, device=device)

@functools.lru_cache(maxsize=1)
def load_metapathway_tables(*, root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the four canonical TSV tables shipped with the project once and save them on cache for future access.

    Parameters
    ----------
    root
        Directory that contains the `.tsv` files.

    Returns
    -------
    tuple
        `(tumor_df, nodes_df, edges_df, pathway_df)` all as *pandas* dataframes.
    """

    return (
        pd.read_csv(root / "test_tumor_samples.tsv", sep="\t"),
        pd.read_csv(root / "metapathway_nodes_2025.tsv", sep="\t"),
        pd.read_csv(root / "metapathway_edges_simplified_2025.tsv", sep="\t"),
        pd.read_csv(root / "metapathway_nodes_to_pathways_2025.tsv", sep="\t"),
    )

@functools.lru_cache(maxsize=None)
def _prepare_default_tensors_cached(root: Path, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    """Internal cache keyed by *(root, str(device))* so CPU/GPU tensors co-exist."""

    tumor_df, nodes_df, edges_df, pathway_df = load_metapathway_tables(root=root)

    tumor_df = filter_input_df(tumor_df, nodes_df, "#Id")

    nodes_ids = nodes_df["#Id"].astype(str).tolist()
    tumor_ids = tumor_df.index.astype(str).tolist()
    nodes_source_ids = edges_df["#Source"].astype(str).drop_duplicates().tolist()
    nodes_target_ids = edges_df["Target"].astype(str).drop_duplicates().tolist()
    pathway_nodes_source_ids = pathway_df["NodeId"].astype(str).drop_duplicates().tolist()

    return {
        "idx_in": build_index_tensor(tumor_ids, nodes_ids, dtype=dtype, device=device),
        "idx_src": build_index_tensor(nodes_source_ids, nodes_ids, dtype=dtype, device=device),
        "idx_tgt": build_index_tensor(nodes_target_ids, nodes_ids, dtype=dtype, device=device),
        "idx_pathway": build_index_tensor(pathway_nodes_source_ids, nodes_ids, dtype=dtype, device=device),
    }

def prepare_default_tensors(*, root: Path, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    """Public wrapper around the cached implementation"""

    return _prepare_default_tensors_cached(root, dtype, device)

def load_idx(name: str, *, root: Path = utils.DATA_DIR, dtype: torch.dtype = torch.long, device: torch.device = utils.DEFAULT_DEVICE) -> torch.Tensor:
    """Returns one tensor of the cached ones"""

    return prepare_default_tensors(root=root, dtype=dtype, device=device)[name]

def load_input(*, root: Path = utils.DATA_DIR) -> np.ndarray:
    """Returns filtered input over metapathway nodes with shape (n_samples, n_features)"""

    tumor_df, nodes_df, _, _ = load_metapathway_tables(root=root)
    tumor_df = filter_input_df(tumor_df, nodes_df, "#Id")

    return tumor_df.T.values

def create_split_indices(save_file: Path, seed: int | None = None) -> None:
    """
    Generate and save to json indices for each sample and divide the dataset into three subsets: train, validation and test

    Parameters
    ----------
    save_file
        file path where the json is saved
    seed
        used for random shuffling of the subsets, if none it is generated
    """

    n_samples = load_input().shape[0]
    idx_all = np.arange(n_samples)

    if seed is None:
        seed = secrets.randbits(32)
    np.random.default_rng(seed).shuffle(idx_all)

    n_train = int(n_samples * utils.TRAIN_SPLIT)
    n_val   = int(n_samples * utils.VAL_SPLIT)
    splits = {
        "train": idx_all[:n_train].tolist(),
        "val":   idx_all[n_train : n_train + n_val].tolist(),
        "test":  idx_all[n_train + n_val :].tolist(),
    }

    save_file.write_text(json.dumps(splits, indent=2))
