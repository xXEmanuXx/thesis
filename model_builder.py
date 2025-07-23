import torch

from utils import build_mask
from autoencoder import AutoEncoder
from data_loader import nodes_source_ids, nodes_target_ids, pathway_nodes_source_ids, pathway_nodes_target_ids

def create_model(device: str):
    # Weights and masks
    graph_m = build_mask(nodes_source_ids, nodes_target_ids)
    latent_m = build_mask(pathway_nodes_source_ids, pathway_nodes_target_ids)

    tensors = [graph_m, latent_m]
    for t in tensors:
        if t.device != torch.device(device):
            t.data = t.to(device)

    model = AutoEncoder(
        graph_m=graph_m,
        latent_m=latent_m,
    )

    return model.to(device=device)