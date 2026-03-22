#!/usr/bin/env python
# coding: utf-8
"""
run_ph_pred_realisation.py

Phase prediction in *realisation mode* for the NEW-APPROACH dataset:
  - Input sample = one realisation containing all pulsars: X[r] has shape (P, L)
  - Target       = shared phase for that realisation: phase_B[r] has shape (L,)
  - Model outputs a single phase series (cos φ, sin φ) for the realisation: (L, 2)

Key idea:
  At each time step t, we have a P-dimensional "snapshot" across pulsars.
  We embed that P-vector into d_model tokens, run a Transformer over time,
  and predict (cos φ, sin φ) per time step.
"""

# ==========================================================
# SECTION 1 — Imports & Global Config
# ==========================================================
import os, sys, math, time, random, platform, csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL REPRODUCIBILITY
# ==========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {SEED}")

# ==========================================================
# Device & AMP
# ==========================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() and platform.system() == "Darwin"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
AMP_ENABLED = (device.type == "cuda")  # AMP on CUDA only
print("Device:", device, "| AMP:", AMP_ENABLED)

scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

# ==========================================================
# Config (NEW APPROACH NPZ)
# ==========================================================
DATA_PATH  = "/scratch/projects/CFP03/CFP03-CF-051/projects/gen_data/new_approach/con/fix/real_min/"


NPZ_NAME   = "lr_signals_with_params_E_B_phase_base.npz"

SAVE_DIR   = "./checkpoints_phase_tx_realisation"
os.makedirs(SAVE_DIR, exist_ok=True)

VAL_SPLIT  = 0.15
BATCH_SIZE = 128
EPOCHS     = 60

# Optional noise (per-realisation SNR)
ADD_NOISE = True
SNR_LO, SNR_HI = 20, 30

# Optim / training
LR   = 2e-4
WD   = 1e-4
CLIP = 1.0

# Loss weights (same spirit as your older run_ph_pred.py)
KAPPA    = 8.0
L_SMOOTH = 0.10
L_SPECT  = 0.05

# Model
USE_FAST   = True        # Fast conv-downsampled Transformer (recommended)
DS_STRIDE  = 4           # must divide L
D_MODEL    = 128
DEPTH      = 4
HEADS      = 4
D_FF       = 512
P_DROP     = 0.1

# ==========================================================
# Helpers
# ==========================================================
def split_realizations(R, val_split=0.15, seed=42):
    rg = np.random.default_rng(seed)
    idx = np.arange(R)
    rg.shuffle(idx)
    n_val = int(round(val_split * R))
    val_r = idx[:n_val]
    trn_r = idx[n_val:]
    return np.sort(trn_r), np.sort(val_r)

def add_noise_snr_flat(X_flat, snr_lo=20, snr_hi=30, seed=0):
    """
    X_flat: (N, D) flattened per-sample vector.
    Adds per-sample Gaussian noise with SNR ~ UniformInteger[snr_lo, snr_hi).
    """
    rg = np.random.default_rng(seed)
    snrs = rg.integers(snr_lo, snr_hi, size=X_flat.shape[0])
    s_x  = np.sqrt(np.sum(X_flat**2, axis=1))
    ns   = (s_x / snrs)[:, None]
    return X_flat + rg.normal(0.0, ns, size=X_flat.shape)

def save_pred_plot(epoch, phi_true, y_pred_unit, out_dir, tag="val_batch0"):
    """
    Save a quick diagnostic plot of phase vs time (angle) for the first sample in a batch.
    """
    # y_pred_unit: (L,2) unit vectors
    ang_pred = np.arctan2(y_pred_unit[:,1], y_pred_unit[:,0])
    ang_true = phi_true

    # unwrap for nicer visual continuity
    ang_pred_u = np.unwrap(ang_pred)
    ang_true_u = np.unwrap(ang_true)

    plt.figure(figsize=(10,4))
    plt.plot(ang_true_u, label="true φ (unwrap)")
    plt.plot(ang_pred_u, label="pred φ (unwrap)", alpha=0.85)
    plt.legend()
    plt.xlabel("t index")
    plt.ylabel("φ (rad, unwrapped)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"phi_epoch_{epoch:04d}_{tag}.png"), dpi=150)
    plt.close()

# ==========================================================
# SECTION 2 — Load & prepare data (REALISATION MODE)
# ==========================================================
npz_path = os.path.join(DATA_PATH, NPZ_NAME)
print("Loading data from:", npz_path)
data = np.load(npz_path, allow_pickle=True)

X_B      = data["X_B"]         # (P,R,L)
phase_B  = data["phase_B"]     # (R,L)

P, R, L = X_B.shape
print("Shapes -> X_B:", X_B.shape, "phase_B:", phase_B.shape)
assert phase_B.shape == (R, L)

if USE_FAST:
    assert (L % DS_STRIDE) == 0, "L must be divisible by DS_STRIDE for USE_FAST=True"

# Center each time series along time axis (per pulsar, per realisation)
X_centered = X_B - X_B.mean(axis=2, keepdims=True)

# Split on realisations
train_r, val_r = split_realizations(R, VAL_SPLIT, seed=SEED)
print(f"Realisations: total={R}, train={len(train_r)}, val={len(val_r)}")

# Build X: (R,P,L) ; y: (R,L)
X_tr = np.transpose(X_centered[:, train_r, :], (1, 0, 2)).astype(np.float32)  # (Rtr,P,L)
X_va = np.transpose(X_centered[:, val_r,   :], (1, 0, 2)).astype(np.float32)  # (Rva,P,L)
phi_tr = phase_B[train_r].astype(np.float32)                                   # (Rtr,L)
phi_va = phase_B[val_r].astype(np.float32)                                     # (Rva,L)

# Optional noise (per realisation, shared across pulsars)
if ADD_NOISE:
    Rtr, Pn, Ln = X_tr.shape
    Rva, _, _   = X_va.shape
    X_tr_flat = X_tr.reshape(Rtr, Pn * Ln)
    X_va_flat = X_va.reshape(Rva, Pn * Ln)
    X_tr_flat = add_noise_snr_flat(X_tr_flat, SNR_LO, SNR_HI, seed=123)
    X_va_flat = add_noise_snr_flat(X_va_flat, SNR_LO, SNR_HI, seed=456)
    X_tr = X_tr_flat.reshape(Rtr, Pn, Ln)
    X_va = X_va_flat.reshape(Rva, Pn, Ln)

print("Final samples -> train:", X_tr.shape, "val:", X_va.shape, "| P:", P, "| L:", L)

# ==========================================================
# SECTION 3 — Standardize inputs (TRAIN stats only)
# ==========================================================
# Standardize on flattened vectors to keep one mean/std per (pulsar,time) feature.
X_tr_flat = torch.from_numpy(np.ascontiguousarray(X_tr.reshape(X_tr.shape[0], -1))).float()
X_va_flat = torch.from_numpy(np.ascontiguousarray(X_va.reshape(X_va.shape[0], -1))).float()

mu_x  = X_tr_flat.mean(dim=0, keepdim=True)
std_x = X_tr_flat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

X_tr_t = ((X_tr_flat - mu_x) / std_x).reshape(X_tr.shape[0], P, L)  # (Rtr,P,L)
X_va_t = ((X_va_flat - mu_x) / std_x).reshape(X_va.shape[0], P, L)

# Targets: convert φ to (cos φ, sin φ)
PHI_tr = torch.from_numpy(np.ascontiguousarray(phi_tr)).float()  # (Rtr,L)
PHI_va = torch.from_numpy(np.ascontiguousarray(phi_va)).float()

Y_tr = torch.stack([torch.cos(PHI_tr), torch.sin(PHI_tr)], dim=-1)  # (Rtr,L,2)
Y_va = torch.stack([torch.cos(PHI_va), torch.sin(PHI_va)], dim=-1)


# ==========================================================
# Save SMALL validation-only Torch file
# ==========================================================

save_dict = {
    "X_va_t": X_va_t.cpu(),        # (R_va, P, L)
    "PHI_va": PHI_va.cpu(),        # (R_va, L)
    "mu_x": mu_x.cpu(),            # (1, P*L)
    "std_x": std_x.cpu(),          # (1, P*L)
}

SAVE_PATH = "val_phase_data_small.pt"
torch.save(save_dict, SAVE_PATH)

print(f"[OK] Saved validation phase dataset to: {SAVE_PATH}")
for k, v in save_dict.items():
    print(f"  {k}: {tuple(v.shape)}")


# ==========================================================
# SECTION 4 — Dataset / Loaders
# ==========================================================
class RealisationDataset(Dataset):
    def __init__(self, X_rpl, Y_phase, PHI_phase):
        self.X = X_rpl
        self.Y = Y_phase
        self.PHI = PHI_phase

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        # X: (P,L), Y: (L,2), PHI: (L,)
        return self.X[i], self.Y[i], self.PHI[i]

pin = (device.type == "cuda")
gen = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    RealisationDataset(X_tr_t, Y_tr, PHI_tr),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, generator=gen, num_workers=0
)
val_loader = DataLoader(
    RealisationDataset(X_va_t, Y_va, PHI_va),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=0
)

print("Train batches:", len(train_loader), "| Val batches:", len(val_loader))

# ==========================================================
# SECTION 5 — Model: Transformer over time with pulsar pooling/embedding
# ==========================================================
class SinusoidalPE(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,T,D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def _normalize_unit(y, eps=1e-7):
    return y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=eps)

class PhaseTransformerRealisation(nn.Module):
    """
    Input:  x  (B,P,L) or (B,P,L,1)
    Output: y  (B,L,2)  [cos φ, sin φ]
    """
    def __init__(self, P, L, d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1):
        super().__init__()
        self.P = P
        self.L = L

        # At each time, embed the P-vector across pulsars -> d_model token
        self.in_proj = nn.Linear(P, d_model)
        self.pe = SinusoidalPE(L, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=d_ff, dropout=p_drop,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, x):
        # x: (B,P,L) or (B,P,L,1)
        if x.dim() == 4:
            x = x.squeeze(-1)
        # (B,P,L) -> (B,L,P)
        x = x.permute(0, 2, 1).contiguous()
        z = self.in_proj(x)   # (B,L,D)
        z = self.pe(z)
        z = self.encoder(z)
        y = self.head(z)      # (B,L,2)
        return y

class PhaseTransformerRealisationFast(nn.Module):
    """
    Same as above, but conv-downsample along time before transformer.
    """
    def __init__(self, P, L, d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1, ds_stride=4):
        super().__init__()
        assert L % ds_stride == 0
        self.P = P
        self.L = L
        self.S = ds_stride
        self.Ls = L // ds_stride

        self.in_proj = nn.Linear(P, d_model)

        # Downsample along time on token channels
        self.ds_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=ds_stride, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.post_ds_norm = nn.LayerNorm(d_model)
        self.pe = SinusoidalPE(self.Ls, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=d_ff, dropout=p_drop,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head_lowrate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )
        self.upsampler = nn.Upsample(size=L, mode="linear", align_corners=False)

    def forward(self, x):
        # x: (B,P,L) or (B,P,L,1)
        if x.dim() == 4:
            x = x.squeeze(-1)
        x = x.permute(0, 2, 1).contiguous()  # (B,L,P)
        z = self.in_proj(x)                  # (B,L,D)

        # conv expects (B,D,L)
        z = z.transpose(1, 2)                # (B,D,L)
        z = self.ds_conv(z)                  # (B,D,Ls)
        z = z.transpose(1, 2)                # (B,Ls,D)
        z = self.post_ds_norm(z)
        z = self.pe(z)

        z = self.encoder(z)                  # (B,Ls,D)
        y_low = self.head_lowrate(z)         # (B,Ls,2)

        # upsample back to L
        y = self.upsampler(y_low.transpose(1, 2)).transpose(1, 2)  # (B,L,2)
        return y

model = (
    PhaseTransformerRealisationFast(P, L, d_model=D_MODEL, depth=DEPTH, heads=HEADS,
                                    d_ff=D_FF, p_drop=P_DROP, ds_stride=DS_STRIDE)
    if USE_FAST else
    PhaseTransformerRealisation(P, L, d_model=D_MODEL, depth=DEPTH, heads=HEADS,
                                d_ff=D_FF, p_drop=P_DROP)
).to(device)

print("Model params: %.3fM" % (sum(p.numel() for p in model.parameters())/1e6))



# ==========================================================
# SECTION 6 — Losses / Metrics
# ==========================================================
def von_mises_nll(y_pred, y_true, kappa=KAPPA):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    cosd = torch.sum(yp * yt, dim=-1)
    return torch.mean(-kappa * cosd)

def circular_smoothness_loss(y):
    y  = _normalize_unit(y)
    a  = torch.atan2(y[..., 1], y[..., 0])  # (B,L)
    d  = a[:, 1:] - a[:, :-1]
    d  = (d + math.pi) % (2 * math.pi) - math.pi
    return torch.mean(d**2)

def spectral_unit_loss(y_pred, y_true):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    zp = torch.complex(yp[...,0], yp[...,1])
    zt = torch.complex(yt[...,0], yt[...,1])
    # Disable AMP in FFT for numerical stability
    with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"), enabled=False):
        Pp = torch.fft.fft(zp, dim=-1)
        Pt = torch.fft.fft(zt, dim=-1)
        Pm = Pp.abs(); Tm = Pt.abs()
        Pm = Pm / (Pm.norm(dim=-1, keepdim=True) + 1e-8)
        Tm = Tm / (Tm.norm(dim=-1, keepdim=True) + 1e-8)
        return F.mse_loss(Pm, Tm)

def total_loss(y_pred, y_true):
    lv = von_mises_nll(y_pred, y_true)
    ls = circular_smoothness_loss(y_pred)
    lp = spectral_unit_loss(y_pred, y_true)
    tot = lv + L_SMOOTH * ls + L_SPECT * lp
    return tot, dict(total=tot.item(), von_mises=lv.item(), smooth=ls.item(), spectral=lp.item())

@torch.no_grad()
def angular_mae_deg(y_pred, y_true):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    ap = torch.atan2(yp[...,1], yp[...,0])
    at = torch.atan2(yt[...,1], yt[...,0])
    d  = (ap - at + math.pi) % (2*math.pi) - math.pi
    return d.abs().mean().mul(180/math.pi).item()

# ==========================================================
# SECTION 7 — Optim / Sched
# ==========================================================
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)


# ==========================================================
# SECTION 8 — Train / Eval loops + logging
# ==========================================================
best_path = os.path.join(SAVE_DIR, "best_fast.pt" if USE_FAST else "best_slow.pt")
last_path = os.path.join(SAVE_DIR, "last_fast.pt" if USE_FAST else "last_slow.pt")
csv_path  = os.path.join(SAVE_DIR, "loss_log.csv")
png_path  = os.path.join(SAVE_DIR, "loss_curves.png")
PLOT_DIR  = os.path.join(SAVE_DIR, "epoch_phase_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# CSV header
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_total","train_vm","train_sm","train_sp",
            "val_total","val_vm","val_sm","val_sp",
            "val_angMAE_deg",
            "lr","epoch_time_s"
        ])

epochs_hist, train_hist, val_hist = [], [], []
best_val = float("inf")

def save_loss_plot():
    if len(epochs_hist) == 0:
        return
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

def run_epoch(loader, train=True):
    model.train(train)
    tot = 0.0; n = 0
    parts_sum = {"total":0.0, "von_mises":0.0, "smooth":0.0, "spectral":0.0}

    for xb, yb, phib in loader:
        xb = xb.to(device)     # (B,P,L)
        yb = yb.to(device)     # (B,L,2)

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                            dtype=torch.float16, enabled=AMP_ENABLED):
            y_pred = model(xb)
            loss, parts = total_loss(y_pred, yb)

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()

        bs = xb.size(0)
        tot += parts["total"] * bs
        n   += bs
        for k in parts_sum:
            parts_sum[k] += parts[k] * bs

    for k in parts_sum:
        parts_sum[k] /= max(1, n)
    return parts_sum["total"], parts_sum

# ==========================================================
# SECTION 9 — Training loop (saves per-epoch phase plot)
# ==========================================================
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr, trp = run_epoch(train_loader, train=True)
    va, vap = run_epoch(val_loader, train=False)
    sched.step()
    lr = sched.get_last_lr()[0]

    # quick val batch diagnostic + angMAE
    with torch.no_grad():
        xb, yb, phib = next(iter(val_loader))
        xb = xb.to(device)
        yb = yb.to(device)
        y_pred = model(xb)
        ang_mae = angular_mae_deg(y_pred, yb)

        # save first sample plot for this epoch
        y0 = _normalize_unit(y_pred[0]).detach().cpu().numpy()
        phi0 = phib[0].numpy()
        save_pred_plot(epoch, phi0, y0, PLOT_DIR, tag="val0")

    dt = time.time() - t0

    print(
        f"[{epoch:03d}/{EPOCHS:03d}] "
        f"train {tr:.6f} | val {va:.6f} | angMAE {ang_mae:.3f} deg | lr {lr:.3e} | {dt:.1f}s"
    )

    epochs_hist.append(epoch)
    train_hist.append(tr)
    val_hist.append(va)
    save_loss_plot()

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            trp["total"], trp["von_mises"], trp["smooth"], trp["spectral"],
            vap["total"], vap["von_mises"], vap["smooth"], vap["spectral"],
            ang_mae, lr, dt
        ])

    # Best + last ckpt
    if va < best_val:
        best_val = va
        torch.save({
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "mu_x": mu_x.cpu(),
            "std_x": std_x.cpu(),
            "P": P,
            "L": L,
            "USE_FAST": USE_FAST,
            "DS_STRIDE": DS_STRIDE,
            "D_MODEL": D_MODEL,
            "DEPTH": DEPTH,
            "HEADS": HEADS,
            "D_FF": D_FF
        }, best_path)

    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "mu_x": mu_x.cpu(),
        "std_x": std_x.cpu(),
        "P": P,
        "L": L,
        "USE_FAST": USE_FAST,
        "DS_STRIDE": DS_STRIDE,
        "D_MODEL": D_MODEL,
        "DEPTH": DEPTH,
        "HEADS": HEADS,
        "D_FF": D_FF
    }, last_path)

print("Training complete.")
print("Best checkpoint:", best_path)
print("Last checkpoint:", last_path)
print("Loss curve PNG:", png_path)
print("Per-epoch phase plots:", PLOT_DIR)



# ==========================================================
# SECTION 10 — After training: pick ONE val sample and plot φ_pred vs φ_true
# 

import numpy as np
import matplotlib.pyplot as plt
import torch

# choose which sample from val_loader to plot
VAL_PICK = 30   # change to 5, 10, etc.

model.eval()
with torch.no_grad():
    # get a deterministic batch (first batch from val_loader)
    xb, yb, phib = next(iter(val_loader))   # xb:(B,P,L), yb:(B,L,2), phib:(B,L)
    xb = xb.to(device)
    yb = yb.to(device)

    y_pred = model(xb)                      # (B,L,2)
    y0 = _normalize_unit(y_pred[VAL_PICK]).detach().cpu().numpy()  # (L,2)

    # predicted phase from (cos,sin) or (x,y) unit vector
    phi_pred = np.arctan2(y0[:, 1], y0[:, 0])     # (L,)

    # true phase from loader
    phi_true = phib[VAL_PICK].detach().cpu().numpy().astype(np.float64)  # (L,)

# unwrap for meaningful comparison
phi_pred_u = np.unwrap(phi_pred)
phi_true_u = np.unwrap(phi_true)
dphi = phi_pred_u - phi_true_u

rmse = float(np.sqrt(np.mean(dphi**2)))
mae  = float(np.mean(np.abs(dphi)))

# --------------------------------------------------
# Output directory & filenames
# --------------------------------------------------
OUTDIR = "plots_phase"
os.makedirs(OUTDIR, exist_ok=True)

fname_main = os.path.join(
    OUTDIR, f"phase_unwrapped_VAL{VAL_PICK:04d}.png"
)
fname_res  = os.path.join(
    OUTDIR, f"phase_residual_unwrapped_VAL{VAL_PICK:04d}.png"
)

# --------------------------------------------------
# Main phase comparison plot
# --------------------------------------------------
t = np.arange(phi_true.shape[0])

plt.figure(figsize=(10, 4.5))
plt.plot(t, phi_true_u, lw=2, label="true φ (unwrapped)")
plt.plot(t, phi_pred_u, lw=1.7, alpha=0.9, label="pred φ (unwrapped)")
plt.grid(ls="--", alpha=0.35)
plt.xlabel("t index")
plt.ylabel("phase")
plt.title(f"VAL sample={VAL_PICK} | RMSE={rmse:.3e}, MAE={mae:.3e}")
plt.legend()
plt.tight_layout()
plt.savefig(fname_main, dpi=200, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# Residual plot
# --------------------------------------------------
plt.figure(figsize=(10, 3))
plt.plot(t, dphi, lw=1.5)
plt.axhline(0.0, color="k", lw=1)
plt.grid(ls="--", alpha=0.35)
plt.xlabel("t index")
plt.ylabel("Δφ = pred − true (unwrapped)")
plt.tight_layout()
plt.savefig(fname_res, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:")
print(" ", fname_main)
print(" ", fname_res)


