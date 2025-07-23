import torch
import torch.nn as nn
import os
import joblib

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils import DEVICE, BATCH_SIZE, LR_INIT, LR_MAX, NUM_EPOCH, CKPT_PATH, DELTA_NORM
from utils import save_model, load_model
from data_loader import input_data
from model_builder import create_model

scaler = StandardScaler()
input_data_norm = scaler.fit_transform(input_data)
joblib.dump(scaler, "scaler.pkl")

inputs = torch.tensor(input_data_norm, dtype=torch.float32)
dataset = TensorDataset(inputs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = create_model(DEVICE)

criterion = nn.MSELoss()
#criterion = nn.HuberLoss(delta=DELTA_NORM)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=NUM_EPOCH, steps_per_epoch=len(dataloader), pct_start=0.3, div_factor=LR_MAX/LR_INIT, final_div_factor=100)

start_epoch = 0
if os.path.exists(CKPT_PATH):
    print(f"carico checkpoint {CKPT_PATH}")

    model_state, optimizer_state, scheduler_state, start_epoch, epoch_loss = load_model()
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    print(f"riprendo da epoch {start_epoch}, epoch_loss={epoch_loss:.4f}")

model.train()
for epoch in range(start_epoch, NUM_EPOCH):
    epoch_loss = 0
    for (x,) in dataloader:
        print("inizio batch")
        x = x.to(DEVICE)

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

    print(f"Epoch {epoch+1}/{NUM_EPOCH}, epoch_loss={epoch_loss:.4f}")

    if (epoch + 1) % 30 == 0:
        save_model(epoch, epoch_loss, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
        print(f"Nuovo BEST (epoch_loss={epoch_loss:.4f}) salvato")

save_model(epoch, epoch_loss, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
print("Salvataggio finale")
