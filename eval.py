"""
This module is used to evaluate and make inference on a trained model

This module must be executed after training with `sweep.py` and requires an argument '--run_id'
which is the directory containing the model and scaler state for a specific training default to `sweeps/sweep_####'
"""

from __future__ import annotations

import argparse
import json
from textwrap import indent

import joblib
import pandas as pd
import torch
import torch.nn.functional as F

import data_loader
import model_builder
import utils

def load_run_cfg(run_id: str):
    results_file = utils.RESULTS_FILE

    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found. have you executed sweep?")
    
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("run_id") == run_id:
                return data
    
    raise ValueError(f"run_id {run_id} not found in {results_file}")

def section(title: str) -> None:
    """Print a section header."""
    line = "=" * len(title)
    print(f"\n{title}\n{line}")

def print_summary(metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        print(f"{k:<30}: {v:.6f}")


def _select_columns(columns: list[str], n_features: int) -> list[str]:
    """Return *n_features* columns: half from the start and half from the end."""
    if n_features >= len(columns):
        return list(columns)
    first_half = n_features // 2
    last_half = n_features - first_half

    return list(columns[:first_half]) + list(columns[-last_half:])

def preview_dataframe(df_input: pd.DataFrame, df_output: pd.DataFrame, n_samples: int, n_features: int) -> None:
    """Show *n* sample rows of input vs output side-by-side."""
    cols = _select_columns(df_input.columns.tolist(), n_features)
    df_preview = pd.concat({"input": df_input[cols].head(n_samples), "output": df_output[cols].head(n_samples)}, axis=1)
    print(df_preview)

def preview_mean(df_input: pd.DataFrame, n_features: int) -> None:
    cols = _select_columns(df_input.columns.tolist(), n_features)
    print(df_input[cols].mean(axis=0))

def evaluate(run_id: str, n_samples: int, n_features: int) -> None:
    section(f"Evaluating run: {run_id}")

    cfg = load_run_cfg(run_id)
    print("Hyper-parameters:")
    print(indent(json.dumps(cfg, indent=2), "  "))

    ckpt = utils.load_model(ckpt_path=utils.RESULTS_DIR / run_id / "best.pt")
    scaler = joblib.load(utils.RESULTS_DIR / run_id / "scaler.pkl")
    splits = json.loads(utils.SPLIT_FILE.read_text())
    
    model = model_builder.create_model(
        cfg.get("dropout"),
        cfg.get("negative_slope"),
        latent_dim=cfg.get("latent_dim"),
        fc_dims=list(map(int, cfg.get("fc_dims").split(","))),
    )
    model.load_state_dict(ckpt["model_state"])

    input_raw = data_loader.load_input()[splits["test"], :]
    input_norm = scaler.transform(input_raw)
    input_tensor = torch.tensor(input_norm, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_raw = scaler.inverse_transform(output_tensor.cpu().numpy())

    mse_norm = F.mse_loss(output_tensor, input_tensor).item()
    mse_raw = F.mse_loss(
        torch.tensor(output_raw, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE),
        torch.tensor(input_raw, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)
    ).item()

    df_input = pd.DataFrame(input_raw, columns=[f"F{i}" for i in range(input_raw.shape[1])])
    df_output = pd.DataFrame(output_raw, columns=df_input.columns)

    section("Reconstruction metrics")
    print_summary({
        "MSE (normalised)": mse_norm,
        "MSE (raw)": mse_raw,
    })

    section(f"Preview of first {n_samples} test samples (raw)")
    preview_dataframe(df_input, df_output, n_samples, n_features)

    section("Mean of the input features displayed above")
    preview_mean(df_input, n_features)

    section("Distribution check (normalised space)")
    mu_in, mu_out = input_tensor.mean().item(), output_tensor.mean().item()
    sd_in, sd_out = input_tensor.std().item(), output_tensor.std().item()
    print(
        f"Mean input / output : {mu_in:.4f} / {mu_out:.4f}\n"
        f"Std  input / output : {sd_in:.4f} / {sd_out:.4f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a run from sweep results")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. sweep_0042")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples to preview")
    parser.add_argument("--n_features", type=int, default=10, help="Features to preview")
    args = parser.parse_args()

    evaluate(args.run_id, args.n_samples, args.n_features)
