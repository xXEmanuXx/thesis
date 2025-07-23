import torch
import pandas as pd
import io
import zstandard as zstd

# Device and checkpoints
DEVICE = 'cpu'
CKPT_PATH = 'neg004.pt'

# Data
METAPATHWAY_NODES = 20507

# Hyper parameters
BATCH_SIZE = 222
NUM_EPOCH = 200
LR_INIT = 1e-3
LR_MAX = 1e-2
NEGATIVE_SLOPE = 0.04 # Leaky ReLU
DELTA_RAW = 700 # 57, 80, 41, 54, 700
DELTA_NORM = 20

def build_weight(weight: torch.Tensor, mask: torch.Tensor, init: str = 'kaiming'):
    n_out = weight.size(0)
    n_in = weight.size(1)

    rows, cols = mask.nonzero(as_tuple=True)
    rows_list = rows.tolist()
    cols_list = cols.tolist()

    fan_in = torch.bincount(torch.tensor(rows_list), minlength=n_out).clamp(min=1)
    if init == "kaiming":
        bound = torch.sqrt(torch.tensor(6.0) / fan_in.float())
    elif init == "xavier":
        fan_out = torch.bincount(torch.tensor(cols_list), minlength=n_in).clamp(min=1)
        fan_avg = (fan_in.unsqueeze(1) + fan_out.unsqueeze(0)) / 2
        bound = torch.sqrt(torch.tensor(6.0) / fan_avg.float())
    else:
        raise ValueError("init deve essere 'kaiming' o 'xavier'")

    # genera pesi solo dove mask=True
    rand = torch.empty(int(mask.sum().item())).uniform_(-1.0, 1.0)
    per_edge_bound = bound[torch.tensor(rows_list)]
    weight[mask] = rand * per_edge_bound

    return weight

def build_mask(src: pd.Series, tgt: pd.Series):
    src_unique = src.drop_duplicates().tolist()
    tgt_unique = tgt.drop_duplicates().tolist()
    idx_src = {node_id: i for i, node_id in enumerate(src_unique)}
    idx_tgt = {node_id: i for i, node_id in enumerate(tgt_unique)}

    rows, cols = [], []

    for src_id, tgt_id in zip(src, tgt):
        rows.append(idx_tgt[tgt_id])
        cols.append(idx_src[src_id])

    n_out, n_in = len(tgt_unique), len(src_unique)
    mask = torch.zeros((n_out, n_in), dtype=torch.bool)
    mask[rows, cols] = True

    return mask

def save_model(epoch: int, epoch_loss: float, model_state, optimizer_state, scheduler_state):
    buf = io.BytesIO()
    ckpt = {
        "epoch": epoch,
        "epoch_loss": epoch_loss,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state
    }

    torch.save(ckpt, buf)
    data = buf.getvalue()

    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(data)
    with open(CKPT_PATH, "wb") as f:
        f.write(compressed)

def load_model(map_location: str = 'cpu'):
    with open(CKPT_PATH, "rb") as f:
        compressed = f.read()
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)
    buf = io.BytesIO(decompressed)

    ckpt = torch.load(buf, map_location=map_location)

    model_state = ckpt["model_state"]
    optimizer_state = ckpt["optimizer_state"]
    scheduler_state = ckpt["scheduler_state"]
    start_epoch = ckpt["epoch"] + 1
    epoch_loss = ckpt["epoch_loss"]

    return model_state, optimizer_state, scheduler_state, start_epoch, epoch_loss
