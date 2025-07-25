import torch
import torch.nn.functional as F
import pandas as pd
import joblib
import numpy as np

from utils import DEFAULT_DEVICE, DELTA_NORM, DELTA_RAW
from utils import load_model
from data_loader import load_input
from model_builder import create_model

model = create_model(DEFAULT_DEVICE)

ckpt = load_model()
model.load_state_dict(ckpt['model_state'])

scaler = joblib.load("scaler.pkl")
input_data = load_input()
input_data_norm = scaler.transform(input_data)

inputs = torch.tensor(input_data_norm, dtype=torch.float32)

model.eval()
with torch.no_grad():
    outputs = model(inputs)
    outputs_np = outputs.cpu().numpy()

    output_raw = scaler.inverse_transform(outputs_np)

    mse_norm = F.mse_loss(outputs, inputs).item()
    mse_raw = F.mse_loss(torch.tensor(output_raw, dtype=torch.float32), torch.tensor(input_data, dtype=torch.float32)).item()
    #mse_norm = F.huber_loss(outputs, inputs, delta=DELTA_NORM).item()
    #mse_raw = F.huber_loss(torch.tensor(output_raw, dtype=torch.float32), torch.tensor(input_data, dtype=torch.float32), delta=DELTA_RAW).item()
    print("MSE norm:", mse_norm, "MSE raw:", mse_raw)

df_input = pd.DataFrame(input_data, columns=[f"Feature_{i}" for i in range(input_data.shape[1])])
df_output = pd.DataFrame(output_raw, columns=[f"Feature_{i}" for i in range(input_data.shape[1])])

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
print("mean in/out", inputs.mean().item(), outputs.mean().item())
print("std  in/out", inputs.std().item(),  outputs.std().item())

# 2) MSE per-feature (norm)
feat_mse = ((outputs - inputs)**2).mean(dim=0)
top_bad  = torch.topk(feat_mse, 10)
print("Peggiori 10 feature:", top_bad)

for name, p in model.named_parameters():
    if "weight" in name and p.ndim > 1:
        print(name, p.detach().abs().mean().item())
    if "bias" in name and p.ndim == 1:
        print(name, p.detach())