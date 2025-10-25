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

# === Utilities ===

def load_run_cfg(run_id: str):
    results_file = utils.RESULTS_FILE
    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found.")
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("run_id") == run_id:
                return data
    raise ValueError(f"run_id {run_id} not found in {results_file}")

def section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")

def print_summary(metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.6f}")

def _select_columns(columns: list[str], n_features: int) -> list[str]:
    """Return *n_features* columns: half from the start and half from the end."""
    if n_features >= len(columns):
        return list(columns)
    first_half = n_features // 2
    last_half = n_features - first_half
    return list(columns[:first_half]) + list(columns[-last_half:])

# === Preview helpers ===

def preview_dataframe(df_input: pd.DataFrame, df_output: pd.DataFrame, n_samples: int, n_features: int):
    """
    Print a side-by-side preview of input vs output.
    Ensures columns are aligned and uses a temporary pandas option context so
    the console prints full tables (no '...') where possible.
    """
    cols = _select_columns(df_input.columns.tolist(), n_features)

    df_input_sel = df_input[cols].head(n_samples).reset_index(drop=True)
    df_output_sel = df_output[cols].head(n_samples).reset_index(drop=True)

    df_preview = pd.concat({"input": df_input_sel, "output": df_output_sel}, axis=1)

    print(df_preview)

def preview_stats(df_input: pd.DataFrame, df_output: pd.DataFrame, n_features: int, title: str | None = None):
    """
    Print mean & std comparison for raw-space DataFrames.
    Useful for the normalized-trained model after inverse-transform.
    """
    if title:
        section(title)

    cols = _select_columns(df_input.columns.tolist(), n_features)
    means_in = df_input[cols].mean()
    means_out = df_output[cols].mean()
    std_in = df_input[cols].std()
    std_out = df_output[cols].std()

    df_stats = pd.DataFrame({
        "input_mean": means_in,
        "output_mean": means_out,
        "mean_diff": (means_out - means_in),
        "input_std": std_in,
        "output_std": std_out,
        "std_diff": (std_out - std_in),
    })
    print(df_stats)

# === Model loading & evaluation ===

def load_model_and_raw_input(run_id: str):
    """Load model checkpoint and raw input test split (raw space)."""
    cfg = load_run_cfg(run_id)
    print("Hyper-parameters:")
    print(indent(json.dumps(cfg, indent=2), "  "))

    ckpt = utils.load_model(ckpt_path=utils.RESULTS_DIR / run_id / "best.pt")
    model = model_builder.create_model(
        cfg.get("dropout"),
        cfg.get("negative_slope"),
        latent_dim=cfg.get("latent_dim"),
        fc_dims=list(map(int, cfg.get("fc_dims").split(","))),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    splits = json.loads(utils.SPLIT_FILE.read_text())
    input_raw = data_loader.load_input()[splits["test"], :]

    return model, input_raw, cfg

def evaluate_raw_model(run_id: str, n_samples: int, n_features: int):
    """
    Evaluate a model that was trained on RAW data (no scaler).
    Returns metrics and DataFrames df_in_raw, df_out_raw.
    """
    model, input_raw, _cfg = load_model_and_raw_input(run_id)

    # input tensor is raw
    input_tensor = torch.tensor(input_raw, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_raw = output_tensor.cpu().numpy()

    mse_raw = F.mse_loss(torch.tensor(output_raw), torch.tensor(input_raw)).item()
    mae_raw = F.l1_loss(torch.tensor(output_raw), torch.tensor(input_raw)).item()

    metrics = {"MSE_raw": mse_raw, "MAE_raw": mae_raw}

    df_in_raw = pd.DataFrame(input_raw, columns=[f"F{i}" for i in range(input_raw.shape[1])])
    df_out_raw = pd.DataFrame(output_raw, columns=df_in_raw.columns)

    # Previews
    section("Reconstruction metrics (RAW-trained model)")
    print_summary(metrics)
    section(f"Preview of first {n_samples} test samples (raw)")
    preview_dataframe(df_in_raw, df_out_raw, n_samples, n_features)
    section("Mean & std comparison (raw model)")
    preview_stats(df_in_raw, df_out_raw, n_features)

    return metrics, df_in_raw, df_out_raw

def evaluate_norm_model(run_id: str, n_samples: int, n_features: int):
    """
    Evaluate a model that was trained on NORMALIZED data.
    Returns:
      - metrics (dict)
      - df_in_norm, df_out_norm  (normalized-space DataFrames)
      - df_in_raw_from_norm, df_out_raw_from_norm (raw-space DataFrames obtained by inverse transform)
    """
    model, input_raw, cfg = load_model_and_raw_input(run_id)

    # load scaler for this normalized training run
    scaler = joblib.load(utils.RESULTS_DIR / run_id / "scaler.pkl")

    # normalized input (what the model actually sees)
    input_norm = scaler.transform(input_raw)
    input_tensor = torch.tensor(input_norm, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_norm = output_tensor.cpu().numpy()
    # inverse-transform outputs to raw space
    output_raw_from_norm = scaler.inverse_transform(output_norm)

    # metrics: raw-space comparison plus normalized-space MSE
    mse_raw = F.mse_loss(torch.tensor(output_raw_from_norm), torch.tensor(input_raw)).item()
    mae_raw = F.l1_loss(torch.tensor(output_raw_from_norm), torch.tensor(input_raw)).item()
    mse_norm = F.mse_loss(output_tensor, input_tensor).item()

    metrics = {"MSE_raw": mse_raw, "MAE_raw": mae_raw, "MSE_norm": mse_norm}

    # Build DataFrames
    df_in_norm = pd.DataFrame(input_norm, columns=[f"F{i}" for i in range(input_norm.shape[1])])
    df_out_norm = pd.DataFrame(output_norm, columns=df_in_norm.columns)

    df_in_norm_raw = pd.DataFrame(input_raw, columns=[f"F{i}" for i in range(input_raw.shape[1])])
    df_out_norm_raw = pd.DataFrame(output_raw_from_norm, columns=df_in_norm_raw.columns)

    # Previews: normalized-space and raw-space
    section("Reconstruction metrics (Normalized-trained model)")
    print_summary(metrics)

    section(f"Preview of first {n_samples} test samples (normalized space)")
    preview_dataframe(df_in_norm, df_out_norm, n_samples, n_features)

    section(f"Preview of first {n_samples} test samples (raw space) — inverse transformed")
    preview_dataframe(df_in_norm_raw, df_out_norm_raw, n_samples, n_features)

    section("Mean & std comparison (normalized space)")
    preview_stats(df_in_norm, df_out_norm, n_features, title=None)

    section("Mean & std comparison (raw space) — inverse transformed outputs")
    preview_stats(df_in_norm_raw, df_out_norm_raw, n_features, title=None)

    return metrics, df_in_norm, df_out_norm, df_in_norm_raw, df_out_norm_raw

# === Orchestration ===

def single_model(type: str, run_id: str, n_samples: int, n_features: int):
    if (type == 'raw'):
        section("=== RAW-TRAINED MODEL ===")
        metrics_raw, df_in_raw, df_out_raw = evaluate_raw_model(run_id, n_samples, n_features)
        
        section("SUMMARY COMPARISON (raw scale)")
        print(f"MSE_raw: {metrics_raw['MSE_raw']:.6f}")
        print(f"MAE_raw: {metrics_raw['MAE_raw']:.6f}")

        return {
            "raw": {"metrics": metrics_raw, "df_in_raw": df_in_raw, "df_out_raw": df_out_raw},
        }

    elif (type == 'norm'):
        section("=== NORMALIZED-TRAINED MODEL ===")
        metrics_norm, df_in_norm, df_out_norm, df_in_norm_raw, df_out_norm_raw = evaluate_norm_model(run_id, n_samples, n_features)
        
        section("SUMMARY COMPARISON (raw scale)")
        print(f"MSE_raw: (inv-transformed): {metrics_norm['MSE_raw']:.6f}")
        print(f"MAE_raw: (inv-transformed): {metrics_norm['MAE_raw']:.6f}")
        print(f"MSE_norm (normalized space): {metrics_norm.get('MSE_norm', float('nan')):.6f}")

        # If you want to keep DataFrames for further programmatic use, return them:
        return {
            "norm": {
                "metrics": metrics_norm,
                "df_in_norm": df_in_norm,
                "df_out_norm": df_out_norm,
                "df_in_norm_raw": df_in_norm_raw,
                "df_out_norm_raw": df_out_norm_raw
            }
        }
    else:
        print("Wrong type specified, use 'raw' or 'norm'")
        return
    
def compare_models(run_raw: str, run_norm: str, n_samples: int, n_features: int):
    # Raw model (unchanged behavior)
    section("=== RAW-TRAINED MODEL ===")
    metrics_raw, df_in_raw, df_out_raw = evaluate_raw_model(run_raw, n_samples, n_features)

    # Normalized model (we'll return both normalized-space and raw-space DataFrames)
    section("=== NORMALIZED-TRAINED MODEL ===")
    metrics_norm, df_in_norm, df_out_norm, df_in_norm_raw, df_out_norm_raw = evaluate_norm_model(run_norm, n_samples, n_features)

    # Final summary comparison (raw-space metrics)
    section("SUMMARY COMPARISON (raw scale)")
    print(f"MSE_raw  → Raw model: {metrics_raw['MSE_raw']:.6f} | Normalized model (inv-transformed): {metrics_norm['MSE_raw']:.6f}")
    print(f"MAE_raw  → Raw model: {metrics_raw['MAE_raw']:.6f} | Normalized model (inv-transformed): {metrics_norm['MAE_raw']:.6f}")
    print(f"MSE_norm → Normalized model (normalized space): {metrics_norm.get('MSE_norm', float('nan')):.6f}")

    if metrics_raw["MSE_raw"] < metrics_norm["MSE_raw"]:
        print("\n→ Raw-trained model performs better on raw reconstruction (lower MSE_raw).")
    else:
        print("\n→ Normalized-trained model performs better on raw reconstruction (lower MSE_raw).")

    # If you want to keep DataFrames for further programmatic use, return them:
    return {
        "raw": {"metrics": metrics_raw, "df_in_raw": df_in_raw, "df_out_raw": df_out_raw},
        "norm": {
            "metrics": metrics_norm,
            "df_in_norm": df_in_norm,
            "df_out_norm": df_out_norm,
            "df_in_norm_raw": df_in_norm_raw,
            "df_out_norm_raw": df_out_norm_raw
        }
    }

if __name__ == "__main__":
    mode_parser = argparse.ArgumentParser()

    subparsers = mode_parser.add_subparsers(help='Choose a mode', dest='command')

    single_parser = subparsers.add_parser('single', description='Single model evaluation')
    single_parser.add_argument("--type", required=True, help="Data type of the trained model ('raw' or 'norm')")
    single_parser.add_argument("--run_id", required=True, help='Run ID of the trained model')

    compare_parser = subparsers.add_parser('compare', description='Compare two models against each other')
    compare_parser.add_argument("--run_raw", required=True, help="Run ID for raw-trained model (e.g. sweep_0025)")
    compare_parser.add_argument("--run_norm", required=True, help="Run ID for normalized-trained model (e.g. sweep_0037)")
    
    mode_parser.add_argument("--n_samples", type=int, default=10, help="Samples to preview")
    mode_parser.add_argument("--n_features", type=int, default=10, help="Features to preview")
    args = mode_parser.parse_args()

    pd.option_context("display.max_columns", 10)

    if (args.command == 'single'):
        results = single_model(args.type, args.run_id, args.n_samples, args.n_features)
    elif (args.command == 'compare'):
        results = compare_models(args.run_raw, args.run_norm, args.n_samples, args.n_features)
    else:
        print("Wrong command specified. Use 'single' or 'compare'")
