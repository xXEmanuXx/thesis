"""
Training script for auto-encoder metapathway

Executes complete training with
    - cli arguments for hyper-parameters grid search
    - early stopping and best checkpoint save
    - metrics saving for later evaluation
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
from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.preprocessing import StandardScaler

import data_loader
import model_builder
import utils

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Metapathway Auto-Encoder")

    p.add_argument("--run_id", required=True, help="unique run id")
    p.add_argument("--mode", required=True)
    p.add_argument("--lr", default="1e-3,1e-2")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--negative_slope", type=float, default=0.03)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--fc_dims", default="2048,1024")

    return p

@dataclass
class RunState:
    best_val: float = math.inf
    best_epoch: int = -1
    epochs_no_improve: int = 0

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, scheduler, grad_clip: float = 1.0):
    model.train()
    running = 0.0
    for xb, in loader:
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

    return running / len(loader)


def validate(model: nn.Module, loader: DataLoader, criterion):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, in loader:
            xb = xb.to(utils.DEFAULT_DEVICE)
            preds = model(xb)
            loss = criterion(preds, xb)
            running += loss.item()

    return running / len(loader)

def main(args: argparse.Namespace):
    run_dir = utils.RESULTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "norm":
        scaler = StandardScaler()
        input_data = scaler.fit_transform(data_loader.load_input())
        joblib.dump(scaler, run_dir / "scaler.pkl")
    else:
        input_data = data_loader.load_input()

    input_tensor = torch.tensor(input_data, dtype=utils.DEFAULT_DTYPE, device=utils.DEFAULT_DEVICE)

    splits = json.loads(utils.SPLIT_FILE.read_text())

    dataset = TensorDataset(input_tensor)
    train_set = Subset(dataset, splits["train"])
    val_set = Subset(dataset, splits["val"])
    train_loader = DataLoader(train_set, batch_size=utils.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=utils.BATCH_SIZE, shuffle=False)

    model = model_builder.create_model(args.dropout, args.negative_slope, latent_dim=args.latent_dim, fc_dims=list(map(int, args.fc_dims.split(","))))

    criterion = nn.MSELoss()

    lr_init, lr_max = map(float, args.lr.split(","))
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=args.weight_decay)

    div_factor = lr_max / lr_init
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_max,
        epochs=utils.NUM_EPOCH,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=div_factor,
        final_div_factor=100.0,
    )

    state = RunState()

    stopped_reason = "max_epochs"
    for epoch in range(1, utils.NUM_EPOCH + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch}/{utils.NUM_EPOCH}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss + 1e-8 < state.best_val:
            state.best_val = val_loss
            state.best_epoch = epoch
            state.epochs_no_improve = 0

            utils.save_model(epoch, val_loss, model.state_dict(), ckpt_path=run_dir / "best.pt")
            print(f"new best (val_loss={val_loss:.4f}) saved")
        else:
            state.epochs_no_improve += 1
            if state.epochs_no_improve >= utils.EARLY_STOP:
                stopped_reason = "early_stop"
                print(f"Early-stopping: no improvements since {utils.EARLY_STOP} epochs")
                break

    metrics = {"val_loss": state.best_val, "epoch": state.best_epoch, "stopped_reason": stopped_reason}
    with open(run_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Best val-loss {state.best_val:.4f} @ epoch {state.best_epoch}")

if __name__ == "__main__":
    main(build_parser().parse_args())
