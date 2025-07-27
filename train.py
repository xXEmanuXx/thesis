"""
train.py - Training script per l'auto-encoder metapathway
---------------------------------------------------------
Esegue l'addestramento completo con supporto a:
- argomenti CLI per hyper-parameter sweep (lr, weight_decay, dropout, hidden, bottleneck …)
- early stopping e salvataggio del *best checkpoint*
- One-Cycle LR scheduler di default (disattivabile via flag)
- serializzazione metriche in sweeps/<run_id>/metrics.json compatibile con sweep.py
"""

from __future__ import annotations
import argparse
import json
import math
import joblib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from sklearn.preprocessing import StandardScaler

import data_loader
import model_builder
import utils

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Metapathway Auto-Encoder")

    p.add_argument("--run_id", required=True, help="ID univoco del run / sweep")
    p.add_argument("--lr_init", type=float, default=1e-3)
    p.add_argument("--lr_max", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--negative_slope", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)

    return p

@dataclass
class RunState:
    best_val: float = math.inf
    best_epoch: int = -1
    epochs_no_improve: int = 0

# ---------------------------------------------------------------------------
# Loop di training / validazione
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, scheduler, grad_clip: float):
    model.train()
    running = 0.0
    for xb, in loader:
        print("inizio batch")
        xb = xb.to(utils.DEFAULT_DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = criterion(preds, xb)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        running += loss.item()
        print("batch completato")

    return running / len(loader)


def validate(model: nn.Module, loader: DataLoader, criterion):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, in loader:
            print("inizio val")
            xb = xb.to(utils.DEFAULT_DEVICE)
            preds = model(xb)
            loss = criterion(preds, xb)
            running += loss.item()
            print("fine val")

    return running / len(loader)

# ---------------------------------------------------------------------------
# Entry‑point principale
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    # 1) Caricamento dati ----------------------------------------------------
    scaler = StandardScaler()
    input_data_norm = scaler.fit_transform(data_loader.load_input())

    inputs = torch.tensor(input_data_norm, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)

    dataset = TensorDataset(inputs)
    val_len = int(len(dataset) * utils.VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=utils.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=utils.BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2) Model ---------------------------------------------------------------
    model = model_builder.create_model(args.dropout, args.negative_slope)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

    div_factor = args.lr_max / args.lr_init
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        epochs=utils.NUM_EPOCH,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=div_factor,
        final_div_factor=100.0,
    )

    # 3) Training loop -------------------------------------------------------
    state = RunState()
    run_dir = utils.RESULTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, run_dir / "scaler.pkl")

    stopped_reason = "max_epochs"

    for epoch in range(1, utils.NUM_EPOCH + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, args.grad_clip)
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch}/{utils.NUM_EPOCH}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        # Early‑stopping -----------------------------------------------------
        if val_loss + 1e-8 < state.best_val:
            state.best_val = val_loss
            state.best_epoch = epoch
            state.epochs_no_improve = 0

            if epoch % 1 == 0:
                utils.save_model(epoch, val_loss, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), ckpt_path=run_dir / "best.pt")
                print(f"Nuovo BEST (val_loss={val_loss:.4f}) salvato")
        else:
            state.epochs_no_improve += 1
            if state.epochs_no_improve >= utils.EARLY_STOP:
                print(f"Early-stopping: nessun miglioramento da {utils.EARLY_STOP} epoche")
                stopped_reason = "early_stop"
                break

    # 4) Metriche in JSON ----------------------------------------------------
    metrics = {"val_loss": state.best_val, "epoch": state.best_epoch, "stopped_reason": stopped_reason}
    with open(run_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Best val-loss {state.best_val:.4f} @ epoch {state.best_epoch}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    cfg = parser.parse_args()
    main(cfg)
