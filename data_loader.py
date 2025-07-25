"""
Data loading and preprocessing helpers

This modules deals with filtering an load the input data used in the autoencoder
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import functools

from pathlib import Path
from typing import Sequence, Tuple

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE, DATA_PATH

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
                       device: torch.device = DEFAULT_DEVICE, 
                       dtype: torch.dtype = torch.long) -> torch.Tensor:
    
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
    
    root = Path(root)

    return (
        pd.read_csv(root / "test_tumor_samples.tsv", sep="\t"),
        pd.read_csv(root / "metapathway_nodes_2025.tsv", sep="\t"),
        pd.read_csv(root / "metapathway_edges_simplified_2025.tsv", sep="\t"),
        pd.read_csv(root / "metapathway_nodes_to_pathways_2025.tsv", sep="\t"),
    )

@functools.lru_cache(maxsize=None)
def _prepare_default_tensors_cached(root: Path, device: torch.device):
    """
    Internal cache keyed by *(root, str(device))* so CPU/GPU tensors co-exist.
    """
    tumor_df, nodes_df, edges_df, pathway_df = load_metapathway_tables(root=root)

    tumor_df = filter_input_df(tumor_df, nodes_df, '#Id')

    nodes_ids = nodes_df["#Id"].astype(str).tolist()
    tumor_ids = tumor_df.index.astype(str).tolist()
    nodes_source_ids = edges_df["#Source"].astype(str).drop_duplicates().tolist()
    nodes_target_ids = edges_df["Target"].astype(str).drop_duplicates().tolist()
    pathway_nodes_source_ids = pathway_df["NodeId"].astype(str).drop_duplicates().tolist()

    return {
        "idx_in": build_index_tensor(tumor_ids, nodes_ids, device=device),
        "idx_src": build_index_tensor(nodes_source_ids, nodes_ids, device=device),
        "idx_tgt": build_index_tensor(nodes_target_ids, nodes_ids, device=device),
        "idx_pathway": build_index_tensor(pathway_nodes_source_ids, nodes_ids, device=device),
    }

def prepare_default_tensors(*, root: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Public wrapper around the cached implementation
    """
    return _prepare_default_tensors_cached(Path(root), device)

def _single_tensor(name: str, *, root:Path, device: torch.device):
    return prepare_default_tensors(root=root, device=device)[name]


def load_idx_in(*, root: Path = DATA_PATH, device: torch.device = DEFAULT_DEVICE) -> torch.Tensor:
    """Return **idx_in** tensor (input → meta-pathway total)."""
    return _single_tensor("idx_in", root=root, device=device)


def load_idx_src(*, root: Path = DATA_PATH, device: torch.device = DEFAULT_DEVICE) -> torch.Tensor:
    """Return **idx_src** tensor (meta-pathway total → source nodes)."""
    return _single_tensor("idx_src", root=root, device=device)


def load_idx_tgt(*, root: Path = DATA_PATH, device: torch.device = DEFAULT_DEVICE) -> torch.Tensor:
    """Return **idx_tgt** tensor (meta-pathway target → total)."""
    return _single_tensor("idx_tgt", root=root, device=device)


def load_idx_pathway(*, root: Path = DATA_PATH, device: torch.device = DEFAULT_DEVICE) -> torch.Tensor:
    """Return **idx_pathway** tensor (meta-pathway total → pathway source)."""
    return _single_tensor("idx_pathway", root=root, device=device)

def load_input(*, root: Path = DATA_PATH) -> np.ndarray:
    tumor_df, nodes_df, _, _ = load_metapathway_tables(root=root)
    tumor_df = filter_input_df(tumor_df, nodes_df, '#Id')
    return tumor_df.T.values
