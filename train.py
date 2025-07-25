import torch
import torch.nn as nn
import os
import joblib

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE, BATCH_SIZE, LR_INIT, LR_MAX, NUM_EPOCH, CKPT_PATH
from utils import save_model, load_model
from data_loader import load_input
from model_builder import create_model

scaler = StandardScaler()
input_data_norm = scaler.fit_transform(load_input())
joblib.dump(scaler, "scaler.pkl")

inputs = torch.tensor(input_data_norm, dtype=DEFAULT_DTYPE)
dataset = TensorDataset(inputs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = create_model(DEFAULT_DEVICE)

criterion = nn.MSELoss()
#criterion = nn.HuberLoss(delta=DELTA_NORM)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=NUM_EPOCH, steps_per_epoch=len(dataloader), pct_start=0.3, div_factor=LR_MAX/LR_INIT, final_div_factor=100)

start_epoch = 1
if os.path.exists(CKPT_PATH):
    print(f"carico checkpoint {CKPT_PATH}")

    ckpt = load_model(map_location=DEFAULT_DEVICE, ckpt_path=CKPT_PATH)
    
    start_epoch = ckpt['epoch'] + 1
    epoch_loss = ckpt['epoch_loss']

    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])

    print(f"riprendo da epoch {start_epoch}, epoch_loss={epoch_loss:.4f}")

model.train()
for epoch in range(start_epoch, NUM_EPOCH + 1):
    epoch_loss = 0
    for (x,) in dataloader:
        print("inizio batch")
        x = x.to(DEFAULT_DEVICE)

        y_hat = model(x)

        with torch.no_grad():
            print("mean_in ", x.mean().item(), "std_in ", x.std().item(), "mean_out", y_hat.mean().item(), "std_out", y_hat.std().item())            

        loss = criterion(y_hat, x)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        print("batch completato")

    epoch_loss /= len(dataloader)

    print(f"Epoch {epoch}/{NUM_EPOCH}, epoch_loss={epoch_loss:.4f}")

    if epoch % 30 == 0:
        save_model(epoch, epoch_loss, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
        print(f"Nuovo BEST (epoch_loss={epoch_loss:.4f}) salvato")

save_model(epoch, epoch_loss, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
print("Salvataggio finale")
