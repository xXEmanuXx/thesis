"""
eval_compare.py

Compare raw-trained vs normalized-trained models.
- Raw model: evaluated on raw inputs (unchanged behavior).
- Normalized model: evaluated in normalized space but additionally
  inverse-transformed to raw space. Both normalized and raw-space
  DataFrames are returned so you can preview means in either space.

Usage:
python eval_compare.py --run_raw sweep_0025 --run_norm sweep_0037 --n_samples 10 --n_features 10
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

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(results):
    """Create comparison plots from eval_compare results dictionary."""

    raw = results["raw"]
    norm = results["norm"]

    # Extract metrics
    metrics_raw = raw["metrics"]
    metrics_norm = norm["metrics"]

    # ---- 1. BAR PLOT: overall metrics ----
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["MSE_raw", "MAE_raw"]
    raw_vals = [metrics_raw["MSE_raw"], metrics_raw["MAE_raw"]]
    norm_vals = [metrics_norm["MSE_raw"], metrics_norm["MAE_raw"]]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, raw_vals, width, label="Raw-trained")
    ax.bar(x + width/2, norm_vals, width, label="Normalized-trained")

    ax.set_title("Overall Reconstruction Error (Raw Scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error value")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---- 2. FEATURE-WISE: Mean & Std Difference (raw space) ----
    df_raw = raw["df_in_raw"].copy()
    df_raw_out = raw["df_out_raw"].copy()
    df_norm_raw = norm["df_in_norm_raw"].copy()
    df_norm_out_raw = norm["df_out_norm_raw"].copy()

    # Compute per-feature mean and std diffs
    features = df_raw.columns
    mean_diff_raw = (df_raw_out.mean() - df_raw.mean()).values
    mean_diff_norm = (df_norm_out_raw.mean() - df_norm_raw.mean()).values
    std_diff_raw = (df_raw_out.std() - df_raw.std()).values
    std_diff_norm = (df_norm_out_raw.std() - df_norm_raw.std()).values

    # Select subset (first & last few features)
    n_feats = 5
    idxs = list(range(n_feats)) + list(range(-n_feats, 0))
    feats_to_plot = [features[i] for i in idxs]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axs[0].bar(np.arange(len(feats_to_plot)), mean_diff_raw[idxs], alpha=0.7, label="Raw-trained")
    axs[0].bar(np.arange(len(feats_to_plot)), mean_diff_norm[idxs], alpha=0.7, label="Normalized-trained")
    axs[0].set_title("Feature-wise Mean Difference (Output - Input)")
    axs[0].set_xticks(np.arange(len(feats_to_plot)))
    axs[0].set_xticklabels(feats_to_plot, rotation=45)
    axs[0].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axs[0].legend()

    axs[1].bar(np.arange(len(feats_to_plot)), std_diff_raw[idxs], alpha=0.7, label="Raw-trained")
    axs[1].bar(np.arange(len(feats_to_plot)), std_diff_norm[idxs], alpha=0.7, label="Normalized-trained")
    axs[1].set_title("Feature-wise Std Difference (Output - Input)")
    axs[1].set_xticks(np.arange(len(feats_to_plot)))
    axs[1].set_xticklabels(feats_to_plot, rotation=45)
    axs[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axs[1].legend()

    fig.suptitle("Comparison of Feature Statistics Between Models", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ---- 3. OPTIONAL: Scatter plot of input vs output (raw space) ----
    feat_idx = 0  # change to inspect a specific feature
    plt.figure(figsize=(5,5))
    plt.scatter(df_raw[features[feat_idx]], df_raw_out[features[feat_idx]], alpha=0.5, label="Raw-trained", s=15)
    plt.scatter(df_norm_raw[features[feat_idx]], df_norm_out_raw[features[feat_idx]], alpha=0.5, label="Normalized-trained", s=15)
    plt.plot([df_raw[features[feat_idx]].min(), df_raw[features[feat_idx]].max()],
             [df_raw[features[feat_idx]].min(), df_raw[features[feat_idx]].max()],
             'k--', linewidth=0.8)
    plt.xlabel("Input (raw space)")
    plt.ylabel("Reconstructed Output")
    plt.title(f"Input vs Output: Feature {features[feat_idx]}")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    # temporarily expand pandas printing
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
    parser = argparse.ArgumentParser(description="Compare raw vs normalized models")
    parser.add_argument("--run_raw", required=True, help="Run ID for raw-trained model (e.g. sweep_0025)")
    parser.add_argument("--run_norm", required=True, help="Run ID for normalized-trained model (e.g. sweep_0037)")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples to preview")
    parser.add_argument("--n_features", type=int, default=10, help="Features to preview")
    args = parser.parse_args()

    pd.option_context("display.max_columns", 10)
    results = compare_models(args.run_raw, args.run_norm, args.n_samples, args.n_features)
    plot_comparison(results)
