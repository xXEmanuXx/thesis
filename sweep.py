"""
Grid-search launcher

Executes multiple trainings passing hyper-parameters to `train.py`
and saves metrics in `sweeps/results.jsonl`
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List

import utils
import data_loader

PARAM_GRID: Dict[str, List] = {
    "lr_init": [1e-3],
    "lr_max": [1e-2],
    "weight_decay": [0],
    "dropout": [0.05],
    "negative_slope": [0.03],
    "grad_clip": [1.0]
}

def dict_product(param_grid: Dict[str, List]):
    """Returns all combinations of hyper-parameters"""
    keys, vals = list(param_grid.keys()), list(param_grid.values())
    for comb in itertools.product(*vals):
        yield dict(zip(keys, comb))

def run_single(cfg: Dict, run_id: str):
    """Executes `train.py` with one combination of hyper-parameters"""
    env = os.environ.copy()
    hp_args = list(itertools.chain.from_iterable((f"--{k}", str(v)) for k, v in cfg.items()))

    cmd = ["python", "train.py", "--run_id", run_id, *hp_args,]

    print("Launch:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    metrics_path = utils.RESULTS_DIR / run_id / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}

    with open(utils.RESULTS_FILE, "a") as f:
        f.write(json.dumps({"run_id": run_id, **cfg, **metrics}) + "\n")

if __name__ == "__main__":
    # Commentare queste due righe per evitare reset di 'results.jsonl' e di 'split_indices.json'
    utils.RESULTS_FILE.write_text("")
    data_loader.create_split_indices(utils.SPLIT_FILE)
    # Decommentare se si vuole un seed fissato per ogni grid search
    #data_loader.create_split_indices(utils.SPLIT_FILE, seed=12345)
        
    all_cfgs = list(dict_product(PARAM_GRID))
    for i, cfg in enumerate(all_cfgs):
        run_id = f"sweep_{i:04d}"
        print(f"[{datetime.now().isoformat(timespec="seconds")}] Run {run_id} -> {cfg}")
        run_single(cfg, run_id)
