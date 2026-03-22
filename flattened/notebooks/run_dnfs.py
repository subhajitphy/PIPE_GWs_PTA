#!/usr/bin/env python
# coding: utf-8

# ===================== Imports ======================
import os, sys, csv, random, math, copy, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from scipy.signal import hilbert


pkg_dir = "/scratch/projects/CFP03/CFP03-CF-051/packages/Phy_PE/pred_ph_enc/pred_ph_snr_enc/"
sys.path.append(pkg_dir)
from ea_model_hy import PhaseProvider  # updated version with use_snrenc, etc.
from phase_pred import load_phase_model, predict_phase

# ---- NEW: import model module ----
import model_dnfs as mdl
from model_dnfs import PosteriorNet

# ===================== Device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device (data prep):", device)

# ===================== Data prep (your block) ======================
DATA_PATH = "/scratch/projects/CFP03/CFP03-CF-051/projects/gen_data/sample_ts_1k"

target_names = ["log10_n", "e0", "log10_Mc", "log10_A"]
eps = 1e-8

# ---------------- Load data ----------------
data = np.load(f"{DATA_PATH}/signals_with_params_B.npz", allow_pickle=True)
pul_par = pd.DataFrame.from_records(data["pul_par"])
X_raw_1 = data["signals_B"]         # shape: (N, L)
phi_true_np = data["phase_B"]

# Center timeseries per series
X_centered = X_raw_1 - X_raw_1.mean(axis=1, keepdims=True)

def downsample_timeseries(X: np.ndarray, new_len: int) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("Input X must be 2D, got shape {}".format(X.shape))
    old_len = X.shape[1]
    idx = np.linspace(0, old_len - 1, new_len).astype(int)
    return X[:, idx]

# Downsample to 400
X_d = downsample_timeseries(X_centered, 400)
phi_true_np_d = downsample_timeseries(phi_true_np, 400)

# Add noise with chosen SNRs
snrs = np.random.choice(np.arange(20,30), size=len(X_d))   # one SNR per series
s_x_i_2 = np.sqrt(np.sum(X_d**2, axis=1))                  # L2 norm per series
noise_strengths = (s_x_i_2 / snrs)[:, None]                # σ_h per series
X_noisy = X_d + np.random.normal(0.0, noise_strengths, size=X_d.shape)

# augment derived columns
pul_par["eta"] = pul_par["q"] / (1.0 + pul_par["q"])**2
M  = np.power(10.0, pul_par["log10_M"])
Mc = M * np.power(pul_par["eta"], 3.0/5.0)
pul_par["log10_Mc"] = np.log10(Mc)

'''
# ----- Phase noise injection via amplitude -----
N, L = X_d.shape
rng = np.random.default_rng(42)

# Instantaneous amplitude from the analytic signal of the CLEAN waveform
z_clean  = hilbert(X_d, axis=1)     # analytic signal per row
A_inst   = np.abs(z_clean)          # (N, L)
A_inst   = np.maximum(A_inst, 1e-12)

# Broadcast per-series σ_h to (N, L)
sigma_h = np.broadcast_to(noise_strengths, (N, L))

# Phase perturbation std: σ_φ(t) = σ_h / A(t)
sigma_phi = sigma_h / A_inst

# Sample phase noise and add
delta_phi = rng.normal(0.0, sigma_phi)
phi_noisy_unwrapped = phi_true_np_d + delta_phi

# quick plot/save (optional)
idx = 53
save_dir_preview = "./"
os.makedirs(save_dir_preview, exist_ok=True)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(X_noisy[idx], lw=1); plt.title(f"Noisy Signal (idx={idx})"); plt.xlabel("Time"); plt.ylabel("Amp")
plt.subplot(1, 2, 2)
plt.plot(phi_noisy_unwrapped[idx], lw=1); plt.title(f"Noisy Phase (idx={idx})"); plt.xlabel("Time"); plt.ylabel("Phase [rad]")
plt.tight_layout()
save_path = os.path.join(save_dir_preview, f"noisy_signal_phase_{idx}.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved figure to: {save_path}")

# Wrapped noisy phase in (-pi, pi]
phi_noisy_pt = torch.from_numpy(np.angle(np.exp(1j * phi_noisy_unwrapped))).float()  # (N, L)
'''

VAL_SPLIT  = 0.15
BATCH_SIZE = 128
EPOCHS     = 60

# ----- Standardize y and X -----
y_raw = torch.tensor(pul_par[target_names].to_numpy(), dtype=torch.float32)  # (N, 8)
y_mean = torch.mean(y_raw, dim=0, keepdim=True)
y_std  = torch.std(y_raw, dim=0, keepdim=True)
y = (y_raw - y_mean) / (y_std + 1e-12)

if isinstance(X_noisy, torch.Tensor):
    X_raw = X_noisy.clone().detach()
else:
    X_raw = torch.as_tensor(X_noisy, dtype=torch.float32)

X_mean = X_raw.mean(dim=0, keepdim=True)
X_std  = X_raw.std(dim=0, keepdim=True, unbiased=False)
X = (X_raw - X_mean) / (X_std + 1e-12)

# Dataset WITH phase (can also build one WITHOUT phase: TensorDataset(X, y))
# ds = TensorDataset(X, phi_noisy_pt, y)
ds = TensorDataset(X,  y)
n_val = int(VAL_SPLIT * len(ds))
train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ======================= Posterior model (NF + Transformer + PhaseSinusoidalPE) =============

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() and platform.system()=="Darwin"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
AMP_ENABLED = (DEVICE.type == "cuda")
print("Train Device:", DEVICE, "| AMP:", AMP_ENABLED)

# ------------- Hyperparams -------------
SEQ_LEN     = X.shape[1]        # 400 after your downsample
THETA_DIM   = y.shape[1]        # 8
CTX_DIM     = 256
D_MODEL     = 128
NHEAD       = 4
DEPTH       = 4
FLOW_LAYERS = 8
FLOW_HIDDEN = 256
LR          = 2e-4
WD          = 1e-4
GRAD_CLIP   = 1.0
SAVE_PATH   = "best_posterior_flow.pt"

# ---- Sync THETA_DIM into the model module (so flow dimension matches y) ----
mdl.THETA_DIM = THETA_DIM

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

def _unpack_batch(batch):
    """
    Supports:
      - (xb, phib, yb)
      - (xb, yb)
    Returns: xb, phib_or_None, yb
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            xb, phib, yb = batch
            return xb, phib, yb
        elif len(batch) == 2:
            xb, yb = batch
            return xb, None, yb
    raise ValueError("Batch must be (xb, phib, yb) or (xb, yb)")

def prepare_batch(batch):
    xb, phib, yb = _unpack_batch(batch)
    xb = xb.to(DEVICE, non_blocking=True)   # global standardized X
    yb = yb.to(DEVICE, non_blocking=True)
    if phib is not None:
        phib = phib.to(DEVICE, non_blocking=True)
    return xb, phib, yb

def train_epoch(model, loader, opt, scaler=None):
    model.train()
    total, nobs = 0.0, 0
    for batch in loader:
        xb, phib, yb = prepare_batch(batch)

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
    return total / max(nobs,1)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, nobs = 0.0, 0
    for batch in loader:
        xb, phib, yb = prepare_batch(batch)
        nll = -model.log_prob(yb, xb, phib).mean()
        bs = xb.size(0)
        total += nll.item() * bs
        nobs  += bs
    return total / max(nobs,1)

# ---- Train ----
model = PosteriorNet(use_phase=True, phase_provider=phase_provider).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

best = float("inf")

save_dir = "loss_plots"
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "loss_log.csv")

# ---- Initialize CSV header ----
header = ["epoch", "train_total", "val_total"] + [f"val_{t}" for t in target_names]
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(header)

train_losses, val_losses = [], []

for ep in range(1, EPOCHS + 1):
    tr = train_epoch(model, train_loader, opt, scaler if AMP_ENABLED else None)
    va = eval_epoch(model, val_loader)
    train_losses.append(tr)
    val_losses.append(va)

    # --- Per-target proxy (based on z variance of true y under flow condition) ---
    model.eval()
    per_target_loss = torch.zeros(len(target_names))
    nobs = 0
    with torch.no_grad():
        for batch in val_loader:
            xb, phib, yb = prepare_batch(batch)
            # use same phase logic as in forward
            phase_eff = model._maybe_get_phase(xb, phib)
            h = model.cond(xb, phase_eff)
            z, _ = model.flow.fwd_to_z(yb.to(DEVICE), h)
            mse = (z**2).mean(dim=0)
            per_target_loss += mse.cpu()
            nobs += 1
    per_target_loss /= max(nobs, 1)

    # ---- log epoch + PE / phase-PE weights ----
    backbone = model.cond.backbone
    w_pos_val   = float(backbone.w_pos)   if getattr(backbone, "w_pos", None)   is not None else 0.0
    w_phase_val = float(backbone.w_phase) if getattr(backbone, "w_phase", None) is not None else 0.0

    row = [ep, tr, va] + [x.item() for x in per_target_loss]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    if va < best:
        best = va
        torch.save(model.state_dict(), SAVE_PATH)

    print(
        f"[{ep:03d}/{EPOCHS}] "
        f"train NLL {tr:.4f} | val NLL {va:.4f} | best {best:.4f} | "
        f"w_pos={w_pos_val:.3f} | w_phase={w_phase_val:.3f}"
    )

    # ---- PNGs EVERY EPOCH ----
    # 1) Per-epoch file: loss_curve_epXXX.png
    # 2) Latest file (overwritten): loss_curve_latest.png
    fig = plt.figure(figsize=(6, 4))
    epochs_axis = np.arange(1, len(train_losses) + 1)

    plt.plot(epochs_axis, train_losses, label="Train NLL", lw=2)
    plt.plot(epochs_axis, val_losses,   label="Val NLL",   lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (NLL)")
    plt.title("Train vs Val NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Per-epoch snapshot
    per_epoch_path = os.path.join(save_dir, f"loss_curve_ep{ep:03d}.png")
    fig.savefig(per_epoch_path, dpi=150)

    # Latest snapshot
    latest_path = os.path.join(save_dir, "loss_curve_latest.png")
    fig.savefig(latest_path, dpi=150)

    plt.close(fig)

print("Saved best model to:", SAVE_PATH)
print(f"Loss CSV and plots saved in: {save_dir}")
