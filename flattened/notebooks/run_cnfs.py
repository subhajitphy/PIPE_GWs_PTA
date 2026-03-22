#!/usr/bin/env python
# coding: utf-8

import os, math, csv, sys, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, random_split
from IPython.display import clear_output
from scipy.signal import hilbert

pkg_dir = "/scratch/projects/CFP03/CFP03-CF-051/packages/Phy_PE/pred_ph_enc/pred_ph_snr_enc/"
sys.path.append(pkg_dir)
from ea_model_hy import PhaseProvider  # updated version with use_snrenc, etc.
from phase_pred import load_phase_model, predict_phase

# ---- NEW: import model module ----
from model_cnfs import (
    PosteriorNet,
    prepare_batch,
    CTX_DIM,
    WEIGHTING_MODE,
    ALPHA_POS_INIT,
    ALPHA_PHASE_INIT,
)

# ===================== Device & Config =====================

DATA_PATH = "/scratch/projects/CFP03/CFP03-CF-051/projects/gen_data/sample_ts_1k"

device = torch.device(
    "mps" if torch.backends.mps.is_available() and platform.system()=="Darwin"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
AMP_ENABLED = (device.type == "cuda")
print("Train Device:", device, "| AMP:", AMP_ENABLED)

VAL_SPLIT  = 0.15
BATCH_SIZE = 128
EPOCHS     = 60
LR         = 2e-4
WD         = 1e-4
GRAD_CLIP  = 1.0

SAVE_PATH  = "best_posterior_flow_cnfs.pt"

target_names = ["log10_n","e0","log10_Mc","log10_A"]

# ===================== Data Prep =====================

data = np.load(f"{DATA_PATH}/signals_with_params_B.npz", allow_pickle=True)
pul_par      = pd.DataFrame.from_records(data["pul_par"])
X_raw_1      = data["signals_B"]
phi_true_np  = data["phase_B"]

# center each timeseries
X_centered = X_raw_1 - X_raw_1.mean(axis=1, keepdims=True)

def downsample_timeseries(X: np.ndarray, new_len: int) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Input X must be 2D, got {X.shape}")
    old_len = X.shape[1]
    idx = np.linspace(0, old_len - 1, new_len).astype(int)
    return X[:, idx]

X_d           = downsample_timeseries(X_centered, 400)
phi_true_np_d = downsample_timeseries(phi_true_np, 400)

# SNR + additive Gaussian noise
snrs            = np.random.choice(np.arange(20,30), size=len(X_d))
s_x_i_2         = np.sqrt(np.sum(X_d**2, axis=1))
noise_strengths = (s_x_i_2 / snrs)[:, None]
X_noisy         = X_d + np.random.normal(0.0, noise_strengths, size=X_d.shape)

# derived columns
pul_par["eta"] = pul_par["q"] / (1.0 + pul_par["q"])**2
M   = np.power(10.0, pul_par["log10_M"])
Mc  = M * np.power(pul_par["eta"], 3.0/5.0)
pul_par["log10_Mc"] = np.log10(Mc)

# phase noise via Hilbert amplitude
N, L = X_d.shape
rng = np.random.default_rng(42)
# z_clean = hilbert(X_d, axis=1)
# A_inst  = np.maximum(np.abs(z_clean), 1e-12)
# sigma_h = np.broadcast_to(noise_strengths, (N, L))
# sigma_phi = sigma_h / A_inst
# delta_phi = rng.normal(0.0, sigma_phi)
# phi_noisy_unwrapped = phi_true_np_d + delta_phi

# phi_noisy_pt = torch.from_numpy(
#     np.angle(np.exp(1j * phi_noisy_unwrapped))
# ).float()

# standardize y and X (global)
y_raw  = torch.tensor(pul_par[target_names].to_numpy(), dtype=torch.float32)
y_mean = y_raw.mean(dim=0, keepdim=True)
y_std  = y_raw.std(dim=0, keepdim=True)
y      = (y_raw - y_mean) / (y_std + 1e-12)

X_raw  = torch.as_tensor(X_noisy, dtype=torch.float32)
X_mean = X_raw.mean(dim=0, keepdim=True)
X_std  = X_raw.std(dim=0, keepdim=True, unbiased=False)
X      = (X_raw - X_mean) / (X_std + 1e-12)

# dataset / loaders
# Option A (with phase): ds = TensorDataset(X, phi_noisy_pt, y)
# Option B (no phase in loader): ds = TensorDataset(X, y)
# ds = TensorDataset(X, phi_noisy_pt, y)
phi_true_pt = torch.as_tensor(phi_true_np_d, dtype=torch.float32)
ds = TensorDataset(X, y)

n_val = int(VAL_SPLIT * len(ds))
train_ds, val_ds = random_split(
    ds,
    [len(ds) - n_val, n_val],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ===================== Optional PhaseProvider =====================

USE_PHASE_PROVIDER = True

phase_provider = None
if USE_PHASE_PROVIDER:
    phase_model, mu_x_phase, std_x_phase, T_NEW_PHASE, DS_STRIDE_PHASE = load_phase_model(
        f"{pkg_dir}/best_fast.pt",
        device=device,
        use_fast=True
    )

    # Instantiate PhaseProvider
    phase_provider = PhaseProvider(
        phase_model=phase_model,
        mu_x=mu_x_phase,
        std_x=std_x_phase,
        predict_fn=predict_phase,
        device=device,
        unwrap=True,
        x_mean=X_mean,
        x_std=X_std
    )

    print("PhaseProvider loaded and ready.")
else:
    print("PhaseProvider disabled.")




# ======================= Training / evaluation ===============================

def train_epoch(model, loader, opt, scaler=None):
    model.train()
    total, nobs = 0.0, 0
    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)

        opt.zero_grad(set_to_none=True)

        if scaler is not None and AMP_ENABLED:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                nll = -model.log_prob(yb, xb, phib).mean()
            scaler.scale(nll).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            nll = -model.log_prob(yb, xb, phib).mean()
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        bs = xb.size(0)
        total += nll.item() * bs
        nobs  += bs
    return total / max(nobs, 1)


def eval_epoch(model, loader, max_batches=None):
    """
    No outer no_grad: CNF uses autograd inside forward; we just avoid backward.
    """
    model.eval()
    total, nobs = 0.0, 0
    for bi, batch in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break
        xb, phib, yb = prepare_batch(batch, device)
        nll = -model.log_prob(yb, xb, phib).mean()
        bs  = xb.size(0)
        total += float(nll.item()) * bs
        nobs  += bs
        del nll
    return total / max(nobs, 1)


# ---- Build model ----
# Modes:
# 1) True phase only: PosteriorNet(use_phase=True, phase_provider=None)
# 2) True-or-pred (fallback): PosteriorNet(use_phase=True, phase_provider=phase_provider)
# 3) Predicted phase only: use TensorDataset(X, y) and use_phase=True, phase_provider=phase_provider
# 4) No phase at all: PosteriorNet(use_phase=False, phase_provider=None)

theta_dim = y.shape[1]

model = PosteriorNet(
    theta_dim=theta_dim,
    ctx_dim=CTX_DIM,
    use_phase=False,
    phase_provider=None,
    weighting_mode=WEIGHTING_MODE,
    alpha_pos=ALPHA_POS_INIT,
    alpha_phase=ALPHA_PHASE_INIT,
    atol=1e-3,
    rtol=1e-3,
    method="rk4",
    step_size=0.1,
).to(device)

if AMP_ENABLED:
    scaler = torch.cuda.amp.GradScaler(enabled=True)
else:
    scaler = None

opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
best = float("inf")

# ======================= Logging & Training Loop =======================

save_dir = "loss_plots_cnfs"
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "loss_log_cnfs.csv")

header = ["epoch", "train_total", "val_total"] + [f"val_{t}" for t in target_names]
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

train_losses, val_losses = [], []

for ep in range(1, EPOCHS + 1):
    tr = train_epoch(model, train_loader, opt, scaler)
    va = eval_epoch(model, val_loader, max_batches=None)
    train_losses.append(tr)
    val_losses.append(va)

    # per-target proxy (in base space)
    model.eval()
    per_target_loss = torch.zeros(len(target_names))
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            xb, phib, yb = prepare_batch(batch, device)
            phase_eff = model._maybe_get_phase(xb, phib)
            h  = model.cond(xb, phase_eff)
            z0 = model.flow.to_base(yb.to(device), h)    # (B,D)
            mse = (z0**2).mean(dim=0)
            per_target_loss += mse.cpu()
            n_batches += 1
    per_target_loss /= max(n_batches, 1)

    # learned / manual weights
    w_pos_val   = float(model.cond.backbone.w_pos)   if model.cond.backbone.w_pos is not None else 0.0
    w_phase_val = float(model.cond.backbone.w_phase) if model.cond.backbone.w_phase is not None else 0.0

    row = [ep, tr, va] + [x.item() for x in per_target_loss]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(
        f"[{ep:03d}/{EPOCHS}] train NLL {tr:.4f} | val NLL {va:.4f} | "
        f"best {best:.4f} | w_pos={w_pos_val:.3f} | w_phase={w_phase_val:.3f}"
    )

    if va < best:
        best = va
        torch.save(model.state_dict(), SAVE_PATH)

    if (ep % 5 == 0) or (ep == EPOCHS):
        clear_output(wait=True)
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label="Train NLL", lw=2)
        plt.plot(val_losses,   label="Val NLL",   lw=2)
        plt.xlabel("Epoch"); plt.ylabel("Loss (NLL)")
        plt.title(f"CNF: Train vs Val Loss (up to epoch {ep})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"loss_curve_cnfs_ep{ep:03d}.png"), dpi=150)
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

print("Saved best CNF model to:", SAVE_PATH)
print(f"CNF loss CSV and plots saved in: {save_dir}")

