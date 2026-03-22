#!/usr/bin/env python
# coding: utf-8

# ==========================================================
# SECTION 1 — Imports & Global Config
# ==========================================================

import os, sys, math, time, warnings, random, platform, csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import nullcontext

# ----------------------- Global seeds -----------------------
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================================
# Flexible sequence length
# ==========================================================
T_NEW = 400                # <--- Change this anytime (e.g. 512, 1000, 300)
DS_STRIDE = 4              # Fast model stride (ensure T_NEW % DS_STRIDE == 0)
assert T_NEW % DS_STRIDE == 0, "T_NEW must be divisible by DS_STRIDE"

# ==========================================================
# Paths / Hyperparameters
# ==========================================================
DATA_PATH   = "/scratch/projects/CFP03/CFP03-CF-051/projects/gen_data/sample_ts_1k"
SAVE_DIR    = "./checkpoints_phase_tx_mixed_snr_film"
VAL_FRAC    = 0.10
BATCH_SIZE  = 128

EPOCHS      = 20
LR          = 2e-3
WEIGHT_DECAY= 1e-4

USE_FAST    = True    # True = Fast conv-downsampled Transformer
RESUME      = True    # Resume from last checkpoint

GRAD_CLIP   = 1.0

# Loss weights
KAPPA    = 8.0
L_SMOOTH = 0.10
L_SPECT  = 0.05
L_SNR    = 0.05   # weight for SNR regression loss

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# Device & AMP
# ==========================================================
if torch.cuda.is_available():
    device = torch.device('cuda'); amp_device_type = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps'); amp_device_type = 'mps'
else:
    device = torch.device('cpu'); amp_device_type = 'cpu'

print("Using device:", device)
amp_enabled = (amp_device_type in ('cuda', 'mps'))

scaler = torch.cuda.amp.GradScaler(
    enabled=(amp_device_type == 'cuda')
)

# ==========================================================
# Load Data
# ==========================================================
print("Loading:", DATA_PATH)
data        = np.load(f"{DATA_PATH}/signals_with_params_B.npz", allow_pickle=True)
pul_par     = pd.DataFrame.from_records(data["pul_par"])
X_raw_1     = data["signals_B"]          # shape [N, T_orig]
phi_true_np = data["phase_B"]

# Center each row
X_centered = X_raw_1 - X_raw_1.mean(axis=1, keepdims=True)

# Downsample flexibly
def downsample_timeseries(X, new_len):
    N, old_len = X.shape
    idx = np.linspace(0, old_len - 1, new_len).astype(int)
    return X[:, idx]

X_d           = downsample_timeseries(X_centered, T_NEW)
phi_true_np_d = downsample_timeseries(phi_true_np, T_NEW)

N0 = len(X_d)
print(f"Base dataset size: {N0}, T = {T_NEW}")

# ==========================================================
# 2× Low-SNR + 1× High-SNR dataset expansion
# ==========================================================
# 1) low-SNR copies (2N samples, SNR ∈ [10,30])
N_low = 2 * N0
log10_low_min,  log10_low_max = math.log10(10.0), math.log10(30.0)
log10_snrs_low  = np.random.uniform(log10_low_min, log10_low_max, size=N_low)
snrs_low        = 10.0**log10_snrs_low  # shape [2N]

# 2) high-SNR copies (N samples, SNR ∈ [30,100])
N_high = N0
log10_high_min, log10_high_max = math.log10(30.0), math.log10(100.0)
log10_snrs_high = np.random.uniform(log10_high_min, log10_high_max, size=N_high)
snrs_high       = 10.0**log10_snrs_high

# 3) Build repeated/expanded clean signals
X_rep_low  = np.repeat(X_d, 2, axis=0)  # shape [2N0, T_NEW]
X_rep_high = X_d.copy()                 # shape [N0, T_NEW]

s_x_low  = np.sqrt(np.sum(X_rep_low**2,  axis=1))
s_x_high = np.sqrt(np.sum(X_rep_high**2, axis=1))

noise_low  = (s_x_low  / snrs_low )[:, None]
noise_high = (s_x_high / snrs_high)[:, None]

X_noisy_low  = X_rep_low  + np.random.normal(0.0, noise_low,  size=X_rep_low.shape)
X_noisy_high = X_rep_high + np.random.normal(0.0, noise_high, size=X_rep_high.shape)

# 4) Concatenate noisy signals
X_noisy = np.concatenate([X_noisy_low, X_noisy_high], axis=0)
phi_rep = np.concatenate([np.repeat(phi_true_np_d, 2, axis=0), phi_true_np_d], axis=0)
snrs_all= np.concatenate([snrs_low, snrs_high], axis=0)

N_total = len(X_noisy)
print(f"Expanded dataset size: {N_total} (2× low SNR + 1× high SNR)")

# ==========================================================
# SECTION 2 — Split, standardization, dataset, dataloaders
# ==========================================================

# Shuffle
rng = np.random.default_rng(SEED)
perm = rng.permutation(N_total)

X_mix    = X_noisy[perm]
PHI_mix  = phi_rep[perm]
SNR_mix  = snrs_all[perm]

# Split
n_val     = int(N_total * VAL_FRAC)
idx_split = N_total - n_val

X_tr_np,  X_va_np  = X_mix[:idx_split],  X_mix[idx_split:]
PHI_tr_np,PHI_va_np= PHI_mix[:idx_split],PHI_mix[idx_split:]
snr_tr_np,snr_va_np= SNR_mix[:idx_split],SNR_mix[idx_split:]

# Convert to tensors
X_tr   = torch.from_numpy(np.ascontiguousarray(X_tr_np)).float()
X_va   = torch.from_numpy(np.ascontiguousarray(X_va_np)).float()

PHI_tr = torch.from_numpy(np.ascontiguousarray(PHI_tr_np)).float()
PHI_va = torch.from_numpy(np.ascontiguousarray(PHI_va_np)).float()

snr_tr = torch.from_numpy(snr_tr_np.astype(np.float32))
snr_va = torch.from_numpy(snr_va_np.astype(np.float32))

# Standardize input using TRAIN stats only (do NOT scale SNR)
mu_x  = X_tr.mean()
std_x = X_tr.std().clamp_min(1e-6)

X_tr = (X_tr - mu_x) / std_x
X_va = (X_va - mu_x) / std_x

# Convert phases to (cos φ, sin φ)
Y_tr = torch.stack([torch.cos(PHI_tr), torch.sin(PHI_tr)], dim=-1)
Y_va = torch.stack([torch.cos(PHI_va), torch.sin(PHI_va)], dim=-1)

# Dataset Class
class SeriesDataset(Dataset):
    def __init__(self, X, Y, snr):
        self.X = X
        self.Y = Y
        self.snr = snr

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        # return: X [T,1], Y [T,2], snr [1] (linear SNR)
        return self.X[i].unsqueeze(-1), self.Y[i], self.snr[i:i+1]

pin = (device.type != 'cpu')

train_loader = DataLoader(
    SeriesDataset(X_tr, Y_tr, snr_tr),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin
)
val_loader = DataLoader(
    SeriesDataset(X_va, Y_va, snr_va),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin
)

# ==========================================================
# SECTION 3 — PE + Slow Transformer + Fast Transformer
# ==========================================================

# -------------------------
# Sinusoidal Positional Encoding
# -------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# -------------------------
# Slow Transformer (no downsample, also predicts SNR)
# -------------------------
class PhaseTransformer(nn.Module):
    def __init__(self, seq_len, d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pe = SinusoidalPE(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=d_ff, dropout=p_drop,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Phase head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

        # SNR head (log10 SNR from pooled tokens)
        self.snr_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [B,T,1]
        z = self.in_proj(x)        # [B,T,D]
        z = self.pe(z)             # add PE
        z = self.encoder(z)        # [B,T,D]

        y = self.head(z)           # [B,T,2]

        z_pool = z.mean(dim=1)     # [B,D]
        snr_log10_pred = self.snr_head(z_pool).squeeze(-1)  # [B]

        return y, snr_log10_pred

# -------------------------
# Fast Transformer (downsample + upsample, predicts SNR)
# -------------------------
class PhaseTransformerFast(nn.Module):
    def __init__(self, seq_len, d_model=128, depth=4, heads=4,
                 d_ff=512, p_drop=0.1, ds_stride=4):
        super().__init__()
        assert seq_len % ds_stride == 0
        self.T  = seq_len
        self.S  = ds_stride
        self.Ts = seq_len // ds_stride

        self.ds_conv = nn.Sequential(
            nn.Conv1d(1, d_model//2, kernel_size=7, stride=ds_stride,
                      padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.post_ds_norm = nn.LayerNorm(d_model)
        self.pe = SinusoidalPE(self.Ts, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=d_ff, dropout=p_drop,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)

        self.head_lowrate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

        self.upsampler = nn.Upsample(size=self.T, mode="linear", align_corners=False)

        # SNR head (log10 SNR from pooled tokens in downsampled space)
        self.snr_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [B,T,1]
        z = self.ds_conv(x.transpose(1, 2))   # [B,D,Ts]
        z = z.transpose(1, 2)                 # [B,Ts,D]
        z = self.post_ds_norm(z)
        z = self.pe(z)

        z = self.encoder(z)                   # [B,Ts,D]

        # Phase prediction (low rate -> upsample)
        y_low = self.head_lowrate(z)          # [B,Ts,2]
        y     = self.upsampler(y_low.transpose(1, 2)).transpose(1, 2)  # [B,T,2]

        # SNR prediction from pooled features
        z_pool = z.mean(dim=1)                # [B,D]
        snr_log10_pred = self.snr_head(z_pool).squeeze(-1)  # [B]

        return y, snr_log10_pred

# ==========================================================
# SECTION 4 — Losses, metrics, optimizer, scheduler
# ==========================================================

model = (
    PhaseTransformerFast(T_NEW, d_model=128, depth=4, heads=4,
                         d_ff=512, p_drop=0.1, ds_stride=DS_STRIDE)
    if USE_FAST else
    PhaseTransformer(T_NEW, d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1)
)
model = model.to(device)
print("Model params: %.3fM" %
      (sum(p.numel() for p in model.parameters())/1e6))

# -------------------------
# Losses
# -------------------------
def _normalize_unit(y, eps=1e-7):
    return y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=eps)

def von_mises_nll(y_pred, y_true, kappa=KAPPA):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    cosd = torch.sum(yp*yt, dim=-1)
    return torch.mean(-kappa * cosd)

def circular_smoothness_loss(y):
    y  = _normalize_unit(y)
    a  = torch.atan2(y[...,1], y[...,0])
    d  = a[:,1:] - a[:,:-1]
    d  = (d + math.pi) % (2*math.pi) - math.pi
    return torch.mean(d**2)

def spectral_unit_loss(y_pred, y_true):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    zp = torch.complex(yp[...,0], yp[...,1])
    zt = torch.complex(yt[...,0], yt[...,1])
    # Disable AMP inside FFT for numerical stability
    with torch.autocast(device_type=amp_device_type, enabled=False):
        P  = torch.fft.fft(zp, dim=-1)
        Tt = torch.fft.fft(zt, dim=-1)
        Pm = P.abs(); Tm = Tt.abs()
        Pm = Pm / (Pm.norm(dim=-1, keepdim=True)+1e-8)
        Tm = Tm / (Tm.norm(dim=-1, keepdim=True)+1e-8)
        return F.mse_loss(Pm, Tm)

def total_loss(y_pred, y_true, snr_log10_pred=None, snr_true=None):
    # Phase-related terms
    lv = von_mises_nll(y_pred, y_true)
    ls = circular_smoothness_loss(y_pred)
    lp = spectral_unit_loss(y_pred, y_true)

    # Phase-only loss (for plotting)
    phase_loss = lv + L_SMOOTH*ls + L_SPECT*lp

    # Optional SNR loss (on log10 SNR)
    if (snr_log10_pred is not None) and (snr_true is not None):
        snr_true_log10 = torch.log10(torch.clamp(snr_true, min=1e-3))
        lsnr = F.mse_loss(snr_log10_pred, snr_true_log10.squeeze(-1))
    else:
        lsnr = torch.zeros((), device=y_pred.device)

    # Weighted total
    total = phase_loss + L_SNR * lsnr

    return total, dict(
        total=total.item(),
        phase=phase_loss.item(),
        snr=lsnr.item(),
        von_mises=lv.item(),
        smooth=ls.item(),
        spectral=lp.item()
    )

# -------------------------
# Optimizer + Scheduler
# -------------------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def angular_mae_deg(y_pred, y_true):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    ap = torch.atan2(yp[...,1], yp[...,0])
    at = torch.atan2(yt[...,1], yt[...,0])
    d  = (ap - at + math.pi) % (2*math.pi) - math.pi
    return d.abs().mean().mul(180/math.pi).item()

def run_epoch(loader, train=True):
    model.train(train)
    tot_loss = 0.0; count = 0
    logs = {
        'total':0.0,'phase':0.0,'snr':0.0,
        'von_mises':0.0,'smooth':0.0,'spectral':0.0
    }

    for xb, yb, snb in loader:
        xb  = xb.to(device)
        yb  = yb.to(device)
        snb = snb.to(device)   # true SNR (linear)

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=amp_device_type, dtype=torch.float16,
                            enabled=amp_enabled):
            # Model returns (phase, log10 SNR prediction)
            y_pred, snr_log10_pred = model(xb)
            loss, parts = total_loss(y_pred, yb, snr_log10_pred, snb)

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

        bs = xb.size(0)
        tot_loss += parts["total"] * bs
        count    += bs
        for k in logs:
            logs[k] += parts[k]*bs

    for k in logs:
        logs[k] /= max(1,count)

    return logs["total"], logs

# ==========================================================
# SECTION 5 — Training loop with resume, CSV, plots, ckpts
# ==========================================================

best_path = os.path.join(
    SAVE_DIR,
    "best_fast.pt" if USE_FAST else "best_slow.pt"
)
last_path = os.path.join(
    SAVE_DIR,
    "last_fast.pt" if USE_FAST else "last_slow.pt"
)
csv_path  = os.path.join(SAVE_DIR, "loss_log.csv")
png_path  = os.path.join(SAVE_DIR, "loss_curves.png")

# ------------------------------------------------------
# Resume logic
# ------------------------------------------------------
epochs_hist = []
train_hist  = []
val_hist    = []

if RESUME and os.path.exists(csv_path):
    print("[RESUME] Loading CSV history...")
    with open(csv_path,"r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs_hist.append(int(row["epoch"]))
            train_hist.append(float(row["train_total"]))
            val_hist.append(float(row["val_total"]))
    print("[RESUME] Loaded history:", len(epochs_hist),"epochs")

if RESUME and os.path.exists(last_path):
    print("[RESUME] Loading last checkpoint")
    ckpt = torch.load(last_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    sched.load_state_dict(ckpt["scheduler_state_dict"])
    last_epoch_done = ckpt.get("epoch", 0)
    start_epoch = last_epoch_done + 1
    best_val    = ckpt.get("best_val", float("inf"))
else:
    start_epoch = 1
    best_val    = float("inf")

end_epoch = start_epoch + EPOCHS - 1
print(f"Training epochs: {start_epoch} → {end_epoch}")

# If not resuming, write CSV header
if not RESUME or not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_total","train_phase","train_snr",
            "train_vm","train_sm","train_sp",
            "val_total","val_phase","val_snr",
            "val_vm","val_sm","val_sp",
            "val_angMAE_deg",
            "lr","epoch_time_s"
        ])

# ------------------------------------------------------
# Helper: save loss plot (total only)
# ------------------------------------------------------
def save_loss_plot():
    plt.figure(figsize=(6,4))
    plt.plot(epochs_hist, train_hist, label="train", marker="o")
    plt.plot(epochs_hist, val_hist,   label="val",   marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

# ------------------------------------------------------
# Helper: save multi-component loss plot (total/phase/snr)
# ------------------------------------------------------
def save_multi_loss_plot():
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    # --- Total loss ---
    axes[0].plot(df["epoch"], df["train_total"], label="train")
    axes[0].plot(df["epoch"], df["val_total"], label="val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    # --- Phase loss ---
    axes[1].plot(df["epoch"], df["train_phase"], label="train")
    axes[1].plot(df["epoch"], df["val_phase"], label="val")
    axes[1].set_title("Phase Loss (vonM + smooth + spectral)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    # --- SNR loss ---
    axes[2].plot(df["epoch"], df["train_snr"], label="train")
    axes[2].plot(df["epoch"], df["val_snr"], label="val")
    axes[2].set_title("SNR Loss (MSE on log10 SNR)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_components.png"), dpi=150)
    plt.close()

# ------------------------------------------------------
# Training Loop
# ------------------------------------------------------
for epoch in range(start_epoch, end_epoch+1):
    t0 = time.time()

    tr, tr_parts = run_epoch(train_loader, train=True)
    va, va_parts = run_epoch(val_loader,   train=False)
    sched.step()

    with torch.no_grad():
        xb, yb, snb = next(iter(val_loader))
        xb  = xb.to(device)
        yb  = yb.to(device)
        snb = snb.to(device)
        y_pred, snr_log10_pred = model(xb)
        ang_mae = angular_mae_deg(y_pred, yb)

    dt = time.time() - t0
    lr = sched.get_last_lr()[0]

    print(
        f"[{epoch:03d}/{end_epoch:03d}] "
        f"train {tr:.6f} | val {va:.6f} | "
        f"angMAE {ang_mae:.3f} deg | lr {lr:.3e} | {dt:.1f}s"
    )

    epochs_hist.append(epoch)
    train_hist.append(tr)
    val_hist.append(va)
    save_loss_plot()

    # Append row to CSV
    with open(csv_path,"a",newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            tr_parts["total"], tr_parts["phase"], tr_parts["snr"],
            tr_parts["von_mises"], tr_parts["smooth"], tr_parts["spectral"],
            va_parts["total"], va_parts["phase"], va_parts["snr"],
            va_parts["von_mises"], va_parts["smooth"], va_parts["spectral"],
            ang_mae, lr, dt
        ])

    # Save multi-component loss plot (uses CSV)
    save_multi_loss_plot()

    # Best checkpoint
    if va < best_val:
        best_val = va
        torch.save({
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "mu_x": mu_x,
            "std_x": std_x,
            "T_NEW": T_NEW,
            "DS_STRIDE": DS_STRIDE
        }, best_path)

    # Last checkpoint
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "mu_x": mu_x,
        "std_x": std_x,
        "T_NEW": T_NEW,
        "DS_STRIDE": DS_STRIDE
    }, last_path)

print("Training complete.")
print("Best checkpoint:", best_path)
print("Last checkpoint:", last_path)
print("Loss curve PNG:", png_path)
print("Loss components PNG:", os.path.join(SAVE_DIR, "loss_components.png"))
