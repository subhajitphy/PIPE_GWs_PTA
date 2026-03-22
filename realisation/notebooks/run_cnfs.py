#!/usr/bin/env python
# coding: utf-8

"""
run_dnfs.py  (now trains CNF from model_dnfs.py)

This is your DNFs training script updated to:
- use CNF-based PosteriorNet (same import: model_dnfs.PosteriorNet)
- keep EAUnifiedPE conditioner, realization split, standardization, etc.
- replace z2 diagnostic: RealNVP fwd_to_z -> CNF to_base

Important:
- CNF + Hutchinson trace is usually unstable with AMP/fp16 -> AMP disabled by default.
"""

import os, sys, random, platform, csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
# PATHS
# ==========================================================
PKG_PATH = "/scratch/projects/CFP03/CFP03-CF-051/projects/mis/REALISATION/pkg/"
sys.path.append(PKG_PATH)

from phase_pred import PhaseProvider      # your PhaseProvider
import model_cnfs as mdl                  # now contains CNF implementation
from model_cnfs import PosteriorNet

# ==========================================================
# GLOBAL REPRODUCIBILITY
# ==========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {SEED}")

# ==========================================================
# Device
# ==========================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() and platform.system() == "Darwin"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
# CNF: keep fp32 for stability
AMP_ENABLED = False
print("Device:", device, "| AMP:", AMP_ENABLED)

# ==========================================================
# Config (NPZ)
# ==========================================================
DATA_PATH  = "/scratch/projects/CFP03/CFP03-CF-051/projects/gen_data/new_approach/con/fix/real_min/"
NPZ_NAME   = "lr_signals_with_params_E_B_phase_base.npz"

VAL_SPLIT  = 0.10
BATCH_SIZE = 256
EPOCHS     = 80

# Optim
LR   = 2e-4
WD   = 1e-4
CLIP = 1.0

# Targets
target_names = ["log10_n", "e0", "log10_Mc", "log10_A", "q"]

# REALISATION mode only
SAMPLE_MODE = "realization"

# Pred-phase encoding (NO true phase tensor in loader)
USE_TRUE_PHASE     = False
USE_PHASE_PROVIDER = True
USE_PHASE          = bool(USE_TRUE_PHASE or USE_PHASE_PROVIDER)

# noise: fixed SNR range (ENTIRE dataset first)
ADD_NOISE = True
SNR_LO, SNR_HI = 20, 30

SAVE_DIR = "cnfs_pred_phase_realisation"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "best_posterior_flow_cnfs_pred_phase.pt")
CSV_PATH  = os.path.join(SAVE_DIR, "loss_log_cnfs_pred_phase.csv")

# ==========================================================
# Helpers
# ==========================================================
def split_realizations(R: int, val_split: float = 0.10, seed: int = 42):
    """Deterministic split on realizations."""
    rg = np.random.default_rng(seed)
    idx = np.arange(R)
    rg.shuffle(idx)
    n_val = int(round(val_split * R))
    val_r = idx[:n_val]
    trn_r = idx[n_val:]
    return np.sort(trn_r), np.sort(val_r)

def add_noise_snr_flat(X: np.ndarray, snr_lo: int, snr_hi: int, seed: int):
    """
    X: (N, D)
    Adds Gaussian noise per sample with SNR ~ UniformInt[snr_lo, snr_hi] (deterministic).
    Noise sigma per sample: ||x|| / snr
    """
    rg = np.random.default_rng(seed)
    snrs = rg.integers(snr_lo, snr_hi + 1, size=X.shape[0])  # inclusive hi
    s_x  = np.sqrt(np.sum(X**2, axis=1)).clip(min=1e-12)
    ns   = (s_x / snrs)[:, None]
    noise = rg.normal(0.0, ns, size=X.shape)
    return X + noise

def standardize_realisation_X(X_tr: np.ndarray, X_va: np.ndarray):
    """
    X_tr, X_va: (R, P, L)
    Standardize using TRAIN stats computed over flattened dimension (P*L).
    Returns torch tensors: X_tr_t, X_va_t, plus (X_mean, X_std) as torch tensors.
    """
    Rtr, P, L = X_tr.shape
    Rva, _, _ = X_va.shape

    X_tr_flat = torch.as_tensor(X_tr.reshape(Rtr, -1), dtype=torch.float32)
    X_va_flat = torch.as_tensor(X_va.reshape(Rva, -1), dtype=torch.float32)

    X_mean = X_tr_flat.mean(dim=0, keepdim=True)
    X_std  = X_tr_flat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)

    X_tr_t = ((X_tr_flat - X_mean) / X_std).reshape(Rtr, P, L)
    X_va_t = ((X_va_flat - X_mean) / X_std).reshape(Rva, P, L)

    return X_tr_t, X_va_t, X_mean, X_std

def standardize_y(y_tr: np.ndarray, y_va: np.ndarray):
    """
    y_tr, y_va: (R, D)
    Standardize using TRAIN stats.
    """
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
    y_va_t = torch.as_tensor(y_va, dtype=torch.float32)

    y_mean = y_tr_t.mean(dim=0, keepdim=True)
    y_std  = y_tr_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)

    y_tr_t = (y_tr_t - y_mean) / y_std
    y_va_t = (y_va_t - y_mean) / y_std

    return y_tr_t, y_va_t, y_mean, y_std

def _unpack_batch(batch):
    """
    Supports:
      - (xb, yb)
      - (xb, phib, yb)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            xb, yb = batch
            return xb, None, yb
        if len(batch) == 3:
            xb, phib, yb = batch
            return xb, phib, yb
    raise ValueError("Batch must be (xb, yb) or (xb, phib, yb)")

def prepare_batch(batch, device):
    xb, phib, yb = _unpack_batch(batch)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    if phib is not None:
        phib = phib.to(device, non_blocking=True)
    return xb, phib, yb

# ==========================================================
# SECTION 1 — Load NPZ
# ==========================================================
npz_path = os.path.join(DATA_PATH, NPZ_NAME)
print("Loading:", npz_path)
data = np.load(npz_path, allow_pickle=True)

X_B        = data["X_B"]           # (P,R,L)
Y_by_psr   = data["Y_by_pulsar"]   # (P,R,nparam)
phase_B    = data["phase_B"]       # (R,L)
param_cols = list(data["param_cols"])

P, R, L = X_B.shape
print("Shapes:")
print("  X_B      :", X_B.shape)
print("  Y_by_psr :", Y_by_psr.shape)
print("  phase_B  :", phase_B.shape)
print("  P,R,L    :", P, R, L)

# ==========================================================
# SECTION 2 — Ensure log10_Mc exists
# ==========================================================
if "log10_Mc" not in param_cols:
    if ("log10_M" not in param_cols) or ("q" not in param_cols):
        raise KeyError("Need 'log10_M' and 'q' in param_cols to derive log10_Mc.")
    iM = param_cols.index("log10_M")
    iq = param_cols.index("q")
    log10_M = Y_by_psr[:, :, iM]
    q       = Y_by_psr[:, :, iq]
    eta     = q / (1.0 + q)**2
    Mc      = (10.0**log10_M) * (eta**(3.0/5.0))
    log10_Mc = np.log10(Mc)
    Y_by_psr = np.concatenate([Y_by_psr, log10_Mc[..., None]], axis=2)
    param_cols.append("log10_Mc")
    print("Derived and appended log10_Mc. New nparam =", len(param_cols))

name_to_idx = {n: i for i, n in enumerate(param_cols)}
for n in target_names:
    if n not in name_to_idx:
        raise KeyError(f"target '{n}' not in param_cols={param_cols}")

tidx = [name_to_idx[n] for n in target_names]
THETA_DIM = len(tidx)
mdl.THETA_DIM = THETA_DIM  # IMPORTANT: sets CNF theta dimension inside model_dnfs.py
print("THETA_DIM =", THETA_DIM)

# ==========================================================
# SECTION 3 — Center X per pulsar (same as training)
# ==========================================================
X_centered = X_B - X_B.mean(axis=2, keepdims=True)  # (P,R,L)

# ==========================================================
# SECTION 4 — Add noise to ENTIRE dataset first (deterministic)
# ==========================================================
if ADD_NOISE:
    print(f"Adding noise to ENTIRE dataset with SNR in [{SNR_LO}, {SNR_HI}] (inclusive).")
    X_all = np.transpose(X_centered, (1, 0, 2))   # (R,P,L)
    X_all_flat = X_all.reshape(R, P * L)          # (R, P*L)
    X_all_flat = add_noise_snr_flat(X_all_flat, SNR_LO, SNR_HI, seed=SEED + 12345)
    X_all = X_all_flat.reshape(R, P, L)           # (R,P,L)
else:
    X_all = np.transpose(X_centered, (1, 0, 2))   # (R,P,L)

# ==========================================================
# SECTION 5 — Deterministic split by realizations
# ==========================================================
train_r, val_r = split_realizations(R, val_split=VAL_SPLIT, seed=SEED)
print(f"Split realizations: total={R} | train={len(train_r)} | val={len(val_r)}")

X_tr = X_all[train_r]                     # (Rtr,P,L)
X_va = X_all[val_r]                       # (Rva,P,L)

# global params per realization: pulsar 0 (shared)
y_tr = Y_by_psr[0, train_r, :][:, tidx]   # (Rtr,D)
y_va = Y_by_psr[0, val_r,   :][:, tidx]   # (Rva,D)

# True phase tensors if you want them
if USE_TRUE_PHASE:
    phi_tr = np.repeat(phase_B[train_r][:, None, :], P, axis=1)  # (Rtr,P,L)
    phi_va = np.repeat(phase_B[val_r][:, None, :],   P, axis=1)  # (Rva,P,L)
else:
    phi_tr = phi_va = None

print("Final arrays:")
print("  X_tr:", X_tr.shape, "X_va:", X_va.shape)
print("  y_tr:", y_tr.shape, "y_va:", y_va.shape)

# ==========================================================
# SECTION 6 — Standardization (TRAIN stats only)
# ==========================================================
X_tr_t, X_va_t, X_mean, X_std = standardize_realisation_X(X_tr, X_va)
y_tr_t, y_va_t, y_mean, y_std = standardize_y(y_tr, y_va)

if USE_TRUE_PHASE:
    phi_tr_t = torch.as_tensor(phi_tr, dtype=torch.float32)
    phi_va_t = torch.as_tensor(phi_va, dtype=torch.float32)

print("Torch tensors:")
print("  X_tr_t:", tuple(X_tr_t.shape), "X_va_t:", tuple(X_va_t.shape))
print("  y_tr_t:", tuple(y_tr_t.shape), "y_va_t:", tuple(y_va_t.shape))
print("  X_mean/std:", tuple(X_mean.shape), tuple(X_std.shape))

# ==========================================================
# SECTION 7 — DataLoaders
# ==========================================================
pin = (device.type == "cuda")
loader_gen = torch.Generator().manual_seed(SEED)

if USE_TRUE_PHASE:
    train_ds = TensorDataset(X_tr_t, phi_tr_t, y_tr_t)
    val_ds   = TensorDataset(X_va_t, phi_va_t, y_va_t)
else:
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds   = TensorDataset(X_va_t, y_va_t)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin,
    generator=loader_gen,
    num_workers=0,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin,
    num_workers=0,
)

print("Batches:", "train =", len(train_loader), "| val =", len(val_loader))

# ==========================================================
# SECTION 8 — PhaseProvider (predicted φ)
# ==========================================================
# ==========================================================
# SECTION 8 — PhaseProvider (predicted φ)
# ==========================================================
phase_provider = None
if USE_PHASE_PROVIDER:
    PHASE_CKPT = os.path.join(PKG_PATH, "best_fast.pt")  # adjust if needed

    phase_provider = PhaseProvider(
        phase_ckpt_path=PHASE_CKPT,
        device=device,
        base_len=L,
    )

    # ---- FIX: PhaseProvider needs x_mean/x_std to unstandardize EA inputs ----
    # X_mean/X_std are computed in standardize_realisation_X() on flattened (P*L) features.
    # Ensure they are on the correct device/dtype.
    phase_provider.x_mean = X_mean.to(device=device, dtype=torch.float32)
    phase_provider.x_std  = X_std.to(device=device, dtype=torch.float32)

    print("PhaseProvider ready:", PHASE_CKPT)
    print("PhaseProvider x_mean/x_std set:", tuple(phase_provider.x_mean.shape), tuple(phase_provider.x_std.shape))
else:
    print("PhaseProvider disabled.")


# ==========================================================
# SECTION 9 — Build CNF posterior model
# ==========================================================
model = PosteriorNet(
    seq_len=L,
    use_phase=USE_PHASE,
    phase_provider=phase_provider,
    x_mean=X_mean,
    x_std=X_std,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
best_val = float("inf")

# ==========================================================
# TRAIN/EVAL
# ==========================================================
header = ["epoch", "train_nll", "val_nll"] + [f"val_z2_{t}" for t in target_names]
with open(CSV_PATH, "w", newline="") as f:
    csv.writer(f).writerow(header)

def train_epoch(model, loader):
    model.train()
    total, nobs = 0.0, 0
    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)
        opt.zero_grad(set_to_none=True)

        # CNF: fp32 path
        nll = -model.log_prob(yb, xb, phib).mean()
        nll.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()

        bs = xb.size(0)
        total += float(nll.item()) * bs
        nobs  += bs
    return total / max(nobs, 1)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, nobs = 0.0, 0
    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)
        nll = -model.log_prob(yb, xb, phib).mean()
        bs = xb.size(0)
        total += float(nll.item()) * bs
        nobs  += bs
    return total / max(nobs, 1)

@torch.no_grad()
def per_target_z2(model, loader):
    """
    Diagnostic: E[z^2] per target dim on val.

    For CNF:
      h = model.cond(xb, phase_eff)
      z0 = model.flow.to_base(yb, h)
      return E[z0^2]
    """
    model.eval()
    acc = torch.zeros(len(target_names))
    nb = 0
    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)
        # use PosteriorNet's phase logic for consistency
        phase_eff = phib if model.use_phase else None
        if model.use_phase and (phase_eff is None) and (getattr(model, "phase_provider", None) is not None):
            phase_eff = model.phase_provider(xb)

        h  = model.cond(xb, phase_eff)
        z0 = model.flow.to_base(yb, h)              # (B, D)
        acc += (z0**2).mean(dim=0).detach().cpu()
        nb += 1
    return (acc / max(nb, 1))

# ==========================================================
# LIVE PLOTS
# ==========================================================
def _save_live_plots(train_losses, val_losses, z2_hist, save_dir, epoch,
                     snap_every=1, show_every=0):
    # Loss plot
    fig = plt.figure(figsize=(6.5, 4.2))
    plt.plot(train_losses, label="train_nll")
    plt.plot(val_losses, label="val_nll")
    plt.xlabel("epoch")
    plt.ylabel("NLL")
    plt.grid(ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "live_loss.png"), dpi=160)
    if snap_every and (epoch % snap_every == 0):
        fig.savefig(os.path.join(save_dir, f"live_loss_ep{epoch:04d}.png"), dpi=160)
    if show_every and (epoch % show_every == 0):
        plt.show()
    plt.close(fig)

    if z2_hist is not None and len(z2_hist) > 0:
        z2_arr = np.stack(z2_hist, axis=0)  # (E, D)
        fig2 = plt.figure(figsize=(7.5, 4.2))
        for j, name in enumerate(target_names):
            plt.plot(z2_arr[:, j], label=f"z2_{name}")
        plt.xlabel("epoch")
        plt.ylabel("E[z^2] on val")
        plt.grid(ls="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        fig2.savefig(os.path.join(save_dir, "live_z2.png"), dpi=160)
        if snap_every and (epoch % snap_every == 0):
            fig2.savefig(os.path.join(save_dir, f"live_z2_ep{epoch:04d}.png"), dpi=160)
        if show_every and (epoch % show_every == 0):
            plt.show()
        plt.close(fig2)

# ==========================================================
# TRAIN LOOP
# ==========================================================
train_losses, val_losses = [], []
z2_hist = []
SNAP_EVERY = 1
SHOW_EVERY = 0

for ep in range(1, EPOCHS + 1):
    tr = train_epoch(model, train_loader)
    va = eval_epoch(model, val_loader)
    z2 = per_target_z2(model, val_loader).numpy()

    train_losses.append(tr)
    val_losses.append(va)
    z2_hist.append(z2)

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([ep, tr, va] + [float(x) for x in z2])

    if va < best_val:
        best_val = va
        torch.save({"model_state": model.state_dict()}, SAVE_PATH)

    _save_live_plots(train_losses, val_losses, z2_hist, SAVE_DIR, ep,
                     snap_every=SNAP_EVERY, show_every=SHOW_EVERY)

    print(f"[{ep:03d}] train_nll={tr:.6f}  val_nll={va:.6f}  best={best_val:.6f}")

print("Done. Best saved to:", SAVE_PATH)

# final curve
final_loss_path = os.path.join(SAVE_DIR, "loss_curve_final.png")
plt.figure(figsize=(6.5, 4.2))
plt.plot(train_losses, label="train_nll")
plt.plot(val_losses, label="val_nll")
plt.xlabel("epoch")
plt.ylabel("NLL")
plt.grid(ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(final_loss_path, dpi=160)
plt.show()
