"""
Model factory utilities.

This module centralises the logic to build an :class:`~autoencoder.AutoEncoder`
instance with the correct sparse masks already mapped to the desired
``device``.  Having a single entry-point avoids repeating the mask building
logic across training / evaluation scripts.

"""
import torch

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE, DATA_PATH, build_mask    
from autoencoder import AutoEncoder
from data_loader import load_metapathway_tables

def create_model(dropout: float, negative_slope: float, device: torch.device = DEFAULT_DEVICE, dtype: torch.dtype = DEFAULT_DTYPE):
    """
    Instantiate an :class:`~autoencoder.AutoEncoder` ready for training.

    Parameters
    ----------
    device
        Target device (e.g. ``"cpu"``, ``"cuda:0"``) where both masks and model
        parameters will reside. Defaults to :data:`utils.DEFAULT_DEVICE`.
    dtype
        Desired floating-point precision for model parameters. Masks remain
        boolean so *dtype* is ignored for them. Default is
        :data:`utils.DEFAULT_DTYPE`.

    Returns
    -------
    AutoEncoder
        A fresh model located on `device`.
    """

    tumor_df, nodes_df, edges_df, pathway_df = load_metapathway_tables(root=DATA_PATH)

    metapathway_mask = build_mask(edges_df["#Source"], edges_df["Target"])
    pathway_mask = build_mask(pathway_df["NodeId"], pathway_df["#PathwayId"])

    model = AutoEncoder(metapathway_mask=metapathway_mask, pathway_mask=pathway_mask, dropout=dropout, negative_slope=negative_slope)
    model = model.to(device=device, dtype=dtype)

    return model
