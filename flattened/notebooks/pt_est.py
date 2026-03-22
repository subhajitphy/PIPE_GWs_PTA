#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL REPRODUCIBILITY
# ==========================================================
SEED = 42

# Python & OS
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)                 # Python RNG

# NumPy
np.random.seed(SEED)              # legacy API (if any np.random.* is used)
rng = np.random.default_rng(SEED) # use THIS for all new NumPy randomness

# PyTorch
torch.manual_seed(SEED)           # CPU RNG

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)      # CUDA RNG
    torch.cuda.manual_seed_all(SEED)  # All GPUs

# CuDNN determinism (recommended for reproducibility)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {SEED}")

# ==========================================================
# Basic config
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_PATH  = "/scratch/projects/CFP03/CFP03-CF-051/projects/gen_data/sample_ts_1k"
VAL_SPLIT  = 0.15
BATCH_SIZE = 128
EPOCHS     = 60

target_names = ["log10_n", "e0", "log10_Mc", "log10_A", "q"]
eps = 1e-8

# ==========================================================
# SECTION 1 — Load & prepare data
# ==========================================================
print("Loading data from:", DATA_PATH)
data = np.load(f"{DATA_PATH}/signals_with_params_B.npz", allow_pickle=True)
pul_par      = pd.DataFrame.from_records(data["pul_par"])
X_raw_1      = data["signals_B"]   # shape: (N, L_orig)
phi_true_np  = data["phase_B"]

# Center each timeseries
X_centered = X_raw_1 - X_raw_1.mean(axis=1, keepdims=True)

def downsample_timeseries(X: np.ndarray, new_len: int) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Input X must be 2D, got shape {X.shape}")
    old_len = X.shape[1]
    idx = np.linspace(0, old_len - 1, new_len).astype(int)
    return X[:, idx]

# Downsample to 400
T_NEW = 400
X_d = downsample_timeseries(X_centered, T_NEW)
phi_true_np_d = downsample_timeseries(phi_true_np, T_NEW)

# (old single-realisation noise code left commented-out for reference)
# # Inject noise with random SNR in [20, 30]
# snrs = np.random.choice(np.arange(20, 30), size=len(X_d))   # (N,)
# s_x_i_2 = np.sqrt(np.sum(X_d**2, axis=1))                   # L2 norm per series
# noise_strengths = (s_x_i_2 / snrs)[:, None]                 # σ per series
# X_noisy = X_d + np.random.normal(0.0, noise_strengths, size=X_d.shape)

# -------------------------------------------------
# Config for multi-realisation noisy dataset
# -------------------------------------------------
N_REAL   = 5           # number of noise realisations
snr_min  = 20.0
snr_max  = 30.0

N, T_NEW = X_d.shape
log10_A  = pul_par["log10_A"].values.astype(np.float64)  # (N,)

# -------------------------------------------------
# 1) Draw all random n_val at once: shape (R, N)
#    (using seeded rng -> reproducible)
# -------------------------------------------------
n_val = rng.uniform(-1.0, 1.0, size=(N_REAL, N))   # (R, N)

# noise_strengths[r, i] = 10^(log10_A[i] - n_val[r, i])
noise_strengths = 10.0 ** (log10_A[None, :] - n_val)   # (R, N)

# Expand for broadcasting over time dimension
noise_strengths_exp = noise_strengths[..., None]       # (R, N, 1)

# -------------------------------------------------
# 2) Generate all noise + noisy signals in one go
# -------------------------------------------------
# noise[r, i, t] ~ N(0, sigma[r, i])
noise = rng.normal(0.0, noise_strengths_exp, size=(N_REAL, N, T_NEW))  # (R, N, T)
X_noisy_all = X_d[None, :, :] + noise                                   # (R, N, T)

# -------------------------------------------------
# 3) Compute SNR for each (realisation, sample)
# -------------------------------------------------
s_x_i_2 = np.sqrt(np.sum(X_d**2, axis=1))        # (N,)
snrs_all = s_x_i_2[None, :] / noise_strengths   # (R, N)

# SNR mask per (realisation, sample)
mask_all = (snrs_all >= snr_min) & (snrs_all <= snr_max)   # (R, N)

# -------------------------------------------------
# 4) Flatten masked indices and gather
# -------------------------------------------------
real_idx, samp_idx = np.nonzero(mask_all)  # arrays of length K

# Clean signal (same underlying X_d for each sample index)
X_clean_mask = X_d[samp_idx]                       # (K, T_NEW)

# Noisy signal: depends on realisation r
X_noisy_mask = X_noisy_all[real_idx, samp_idx, :]  # (K, T_NEW)

# Phase: same as clean (depends only on sample index)
phase_mask   = phi_true_np_d[samp_idx]            # (K, T_NEW)

# SNR values
snrs_mask    = snrs_all[real_idx, samp_idx]       # (K,)

# pul_par rows (replicated if same sample appears in multiple realisations)
pul_par_mask = pul_par.iloc[samp_idx].reset_index(drop=True)

# Optional: track which realisation each row came from
realisation_id = real_idx  # shape (K,)

print("Total kept samples across all realisations:", X_clean_mask.shape[0])
print(f"Global SNR range: [{snrs_mask.min():.2f}, {snrs_mask.max():.2f}]")

# -------------------------------------------------
# 5) Use these as your new dataset
# -------------------------------------------------
pul_par       = pul_par_mask
X_d           = X_clean_mask
X_noisy       = X_noisy_mask
phi_true_np_d = phase_mask
snrs          = snrs_mask

#######################################################################
# Use masked data for training
pul_par["eta"] = pul_par["q"] / (1.0 + pul_par["q"])**2
M  = np.power(10.0, pul_par["log10_M"])
Mc = M * np.power(pul_par["eta"], 3.0/5.0)
pul_par["log10_Mc"] = np.log10(Mc)
pul_par["log10_Ac"] = np.log10(np.sqrt(np.mean(X_d**2, axis=1)))

# ==========================================================
# SECTION 2 — Targets and standardization
# ==========================================================
y_raw = torch.tensor(pul_par[target_names].to_numpy(), dtype=torch.float32)  # (N, 5)

# Standardize targets
y_mean = torch.mean(y_raw, dim=0, keepdim=True)
y_std  = torch.std(y_raw, dim=0, keepdim=True)
y = (y_raw - y_mean) / y_std

# Standardize inputs
if isinstance(X_noisy, torch.Tensor):
    X_raw = X_noisy.clone().detach()
else:
    X_raw = torch.as_tensor(X_noisy, dtype=torch.float32)

X_mean = X_raw.mean(dim=0, keepdim=True)
X_std  = X_raw.std(dim=0, keepdim=True, unbiased=False)
X = (X_raw - X_mean) / X_std   # (N, L)

# ==========================================================
# SECTION 3 — Datasets & Loaders (deterministic split & shuffle)
# ==========================================================
ds = TensorDataset(X, y)
n_val = int(VAL_SPLIT * len(ds))

split_gen = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(
    ds,
    [len(ds) - n_val, n_val],
    generator=split_gen
)

loader_gen = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    generator=loader_gen,
    num_workers=0,      # easiest for strict determinism
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
)

# ==========================================================
# SECTION 4 — Import EA model (with phase+SNR encodings)
# ==========================================================
pkg_dir = "/scratch/projects/CFP03/CFP03-CF-051/packages/Phy_PE/pred_ph_enc/pred_ph_snr_enc/"
sys.path.append(pkg_dir)
from ea_model_hy import EAUnifiedPE, PhaseProvider  # updated version with use_snrenc, etc.
from phase_pred import load_phase_model, predict_phase

# ==========================================================
# SECTION 5 — Phase model + PhaseProvider (phase + SNR)
# ==========================================================
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

# ==========================================================
# SECTION 6 — Build EAUnifiedPE model (with phase + SNR enc)
# ==========================================================
model = EAUnifiedPE(
    seq_len=X.shape[1],
    out_dim=y.shape[1],
    patch=20,
    heads=8,
    d_model=128,
    depth=4,
    dim_ff=256,
    mem_size=40,
    weighting="learned",
    alpha_pos=1.0,
    alpha_phase=1.0,
    alpha_snr=1.0,         # initial SNR encoding weight
    use_posenc=True,
    use_phasepe=True,
    use_snrenc=True,       # <--- enable SNR encoding
    cnn_stem=False,
    phase_provider=phase_provider,
    x_mean=X_mean,
    x_std=X_std
).to(device)

print(model)

# ==========================================================
# SECTION 7 — Train
# ==========================================================
from train_plot_eval import train_with_display_and_save

model, curves = train_with_display_and_save(
    model, train_loader, val_loader,
    epochs=50,
    lr=2e-4,
    wd=1e-4,
    clip=1.0,
    device=device,
    target_names=target_names,
    save_dir="epoch_plots_live",
    display_epochs=5,           # show curve every 5 epochs
    save_epochs=1,              # save curve every epoch
    eval_every=5,               # run full-val prediction & plot every 5 epochs
    pred_idxs=tuple(range(len(target_names))),
    y_mean=y_mean,
    y_std=y_std
)
print("Training complete.")

