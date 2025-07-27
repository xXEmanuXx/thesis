# sweep.py — Grid‑search launcher (CPU/GPU‑agnostic)
# --------------------------------------------------
# Lancia training multipli (grid‑search) passando iper‑parametri a train.py
# e colleziona metriche in sweeps/results.jsonl.
# Ora supporta anche l’esecuzione **solo‑CPU** (no CUDA).
# --------------------------------------------------

from __future__ import annotations
import argparse
import itertools
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List
import torch

import utils

# --- 1. Definisci lo spazio degli iper‑parametri ------------------------------
PARAM_GRID: Dict[str, List] = {
    "lr_init": [1e-3],
    "lr_max": [1e-2],
    "weight_decay": [0],
    "dropout": [0.1],
    "negative_slope": [0.03, 0.04],
    "grad_clip": [1.0]

}

# -----------------------------------------------------------------------------

def truncate_results():
    utils.RESULTS_FILE.write_text("")

def dict_product(param_grid: Dict[str, List]):
    """Restituisce tutte le combinazioni di iper-parametri (dict)."""
    keys, vals = list(param_grid.keys()), list(param_grid.values())
    for comb in itertools.product(*vals):
        yield dict(zip(keys, comb))

def run_single(cfg: Dict, run_id: str, device: str):
    """Lancia `train.py` con gli iper-parametri indicati."""
    env = os.environ.copy()
    # Se device è GPU imposta la visibility; se cpu rimuovi la variabile
    if device.lower() != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = device
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    hp_args = list(itertools.chain.from_iterable((f"--{k}", str(v)) for k, v in cfg.items()))

    cmd = ["python", "train.py", "--run_id", run_id, *hp_args,]

    print("Launch:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    metrics_path = utils.RESULTS_DIR / run_id / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}

    with open(utils.RESULTS_FILE, "a") as f:
        f.write(json.dumps({"run_id": run_id, **cfg, **metrics}) + "\n")

# -----------------------------------------------------------------------------

def main(devices: List[str], reset: bool):
    if reset:
        truncate_results()
    all_cfgs = list(dict_product(PARAM_GRID))
    for i, cfg in enumerate(all_cfgs):
        run_id = f"sweep_{i:04d}"
        device = devices[i % len(devices)].strip()
        print(f"[{datetime.now().isoformat(timespec="seconds")}] Run {run_id} on {device} -> {cfg}")
        run_single(cfg, run_id, device)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid-search per l'autoencoder - auto CPU/GPU")
    default_dev = "cpu" if utils.DEFAULT_DEVICE == torch.device("cpu") else "0"
    parser.add_argument(
        "--devices",
        default=default_dev,
        help="Lista di device da usare: 'cpu' oppure id GPU separati da virgola (es. '0,1').",
    )
    args = parser.parse_args()
    device_list = args.devices.split(",") if args.devices else [default_dev]
    main(device_list, utils.RESET_RESULTS_ON_START)
