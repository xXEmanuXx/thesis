from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import data_loader
import model_builder
import utils

def load_run_cfg(run_id: str):
    results_path = utils.RESULTS_FILE

    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found. have you executed sweep?")
    
    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("run_id") == run_id:
                return data
    
    raise ValueError(f"run_id {run_id} not found in {results_path}")

def main(args: argparse.Namespace) -> None:
    run_id = args.run_id

    cfg = load_run_cfg(run_id)
    dropout = cfg.get("dropout")
    negative_slope = cfg.get("negative_slope")

    print(f"Run {run_id}: dropout={dropout} negative_slope={negative_slope}")

    model = model_builder.create_model(dropout, negative_slope)

    ckpt = utils.load_model(ckpt_path=utils.RESULTS_DIR / run_id / "best.pt")

    model.load_state_dict(ckpt["model_state"])
    scaler = joblib.load(utils.RESULTS_DIR / run_id / "scaler.pkl")

    input_raw = data_loader.load_input()
    input_norm = scaler.transform(input_raw)

    input = torch.tensor(input_norm, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(input)

    output_raw = scaler.inverse_transform(output.cpu().numpy())

    mse_norm = F.mse_loss(output, input).item()
    mse_raw = F.mse_loss(torch.tensor(output_raw, dtype=utils.DEFAULT_DTYPE, ), torch.tensor(input_raw, dtype=utils.DEFAULT_DTYPE)).item()
    #mse_norm = F.huber_loss(output, input, delta=DELTA_NORM).item()
    #mse_raw = F.huber_loss(torch.tensor(output_raw, dtype=utils.DEFAULT_DTYPE), torch.tensor(input_raw, dtype=utils.DEFAULT_DTYPE), delta=DELTA_RAW).item()
    
    print(f"MSE norm: {mse_norm:.4f} MSE raw: {mse_raw:.4f}")

    df_input = pd.DataFrame(input_raw, columns=[f"Feature_{i}" for i in range(input_raw.shape[1])])
    df_output = pd.DataFrame(output_raw, columns=[f"Feature_{i}" for i in range(input_raw.shape[1])])

    pd.set_option('display.max_columns', 60)

    print("Input valori raw")
    print(df_input.head(10))
    print("Output valori raw, applicata la inverse_transform() sull'output del modello (norm)")
    print(df_output.head(10))

    df_diff = df_input - df_output
    abs_all  = df_diff.abs().values.ravel().astype(float)
    delta_90 = np.percentile(abs_all, 90)   # q = 90
    print(f"delta (90° percentile) ≈ {delta_90:.2f}")

    print("media per feature")
    print(df_input.mean(axis=0, numeric_only=True).head(30))

    # 1) distribuzione output vs input (norm)
    print("mean in/out", input.mean().item(), output.mean().item())
    print("std  in/out", input.std().item(),  output.std().item())

    # 2) MSE per-feature (norm)
    feat_mse = ((output - input)**2).mean(dim=0)
    top_bad  = torch.topk(feat_mse, 10)
    print("Peggiori 10 feature:", top_bad)

    for name, p in model.named_parameters():
        if "weight" in name and p.ndim > 1:
            print(name, p.detach().abs().mean().item())
        if "bias" in name and p.ndim == 1:
            print(name, p.detach())

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Valuta un run best.pt della sweep")

    p.add_argument("--run_id", required=True, help="ID del run (es. sweep_0042)")
    
    main(p.parse_args())