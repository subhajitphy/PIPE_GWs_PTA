#!/usr/bin/env python
# coding: utf-8

"""
plot_cnfs.py

CNF equivalent of plot_dnfs.py:
- Loads saved validation snapshot: validation_set.npz  (VAL_SNAP_PATH unchanged)
- Loads trained CNF PosteriorNet checkpoint from SAVE_DIR/SAVE_PATH
- Samples posterior for one validation index
- Saves posterior_validxXXXX.npz + corner plot

Usage:
  python plot_cnfs.py 200
"""

import os, sys, random, platform
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

# ==========================================================
# PATHS
# ==========================================================
PKG_PATH = "/scratch/projects/CFP03/CFP03-CF-051/projects/mis/REALISATION/pkg/"
sys.path.append(PKG_PATH)

from phase_pred import PhaseProvider
import model_cnfs as mdl
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

# CNF inference is fine on CUDA; keep fp32 for stability.
AMP_ENABLED = False
print("Device:", device, "| AMP:", AMP_ENABLED)

# ==========================================================
# CONFIG (match training)
# ==========================================================
SAVE_DIR  = "cnfs_pred_phase_realisation"
SAVE_PATH = os.path.join(SAVE_DIR, "best_posterior_flow_cnfs_pred_phase.pt")
#DATA_PATH="/scratch/projects/CFP03/CFP03-CF-051/projects/mis/REALISATION/SBI/SIMPLE/SBI_NO_ENC/Earth/"
DATA_PATH="/scratch/projects/CFP03/CFP03-CF-051/projects/mis/REALISATION/SBI/SIMPLE/SBI_TRUE_ENC/Earth/"
# saved validation snapshot (created once from training pipeline)
VAL_SNAP_PATH = DATA_PATH+"validation_set.npz"

# ==========================================================
# CLI
# ==========================================================
import argparse
parser = argparse.ArgumentParser(description="Run CNF posterior inference for a validation sample")
parser.add_argument("val_index", type=int, help="Validation index (e.g. 200)")
args = parser.parse_args()
VAL_INDEX = args.val_index

NSAMPLES  = 5000
POST_SEED = SEED + 999  # fixed seed => reproducible posterior draws

# phase encoding flags (MUST match training!)
USE_TRUE_PHASE     = False
USE_PHASE_PROVIDER = True   # <-- set this to match what you trained with
USE_PHASE          = bool(USE_TRUE_PHASE or USE_PHASE_PROVIDER)

# output dir (single)
SAVE_POST_DIR = "save_posterior"
os.makedirs(SAVE_POST_DIR, exist_ok=True)

# ==========================================================
# LOAD SAVED VALIDATION SNAPSHOT
# ==========================================================
print("Loading saved validation snapshot:", VAL_SNAP_PATH)
snap = np.load(VAL_SNAP_PATH, allow_pickle=True)

# --- standardized tensors (READY for inference) ---
if "X_va_std" in snap.files:
    X_va_std = snap["X_va_std"]  # (Rva,P,L)
else:
    X_va_std = snap["X_va"]      # legacy key

if "y_va_std" in snap.files:
    y_va_std = snap["y_va_std"]  # (Rva,D)
else:
    y_va_std = snap["y_va"]      # legacy key

# --- unstandardized snapshots ---
X_va_clean = snap["X_va_clean"]    # (Rva,P,L) centered, no noise
X_va_noisy = snap["X_va_noisy"]    # (Rva,P,L) centered + noise (pre-std)

# full params in physical units for validation realizations
y_va_full = snap["y_va_full"]      # (Rva, nparam) physical (unstandardized)

# meta
param_cols   = list(snap["param_cols"]) if "param_cols" in snap.files else None
target_names = list(snap["target_names"])
tidx         = snap["tidx"] if "tidx" in snap.files else None  # indices of target_names in param_cols

# stats
X_mean = torch.as_tensor(snap["X_mean"], dtype=torch.float32)  # (1, P*L)
X_std  = torch.as_tensor(snap["X_std"],  dtype=torch.float32)
y_mean = torch.as_tensor(snap["y_mean"], dtype=torch.float32)  # (1, D)
y_std  = torch.as_tensor(snap["y_std"],  dtype=torch.float32)

val_r = snap["val_r"]  # (Rva,) original realization ids

# --- noise metadata (preferred keys) ---
snr_va = None
sigma_va = None
log10_sigma_va = None
if "snr_va" in snap.files and "sigma_va" in snap.files and "log10_sigma_va" in snap.files:
    snr_va = snap["snr_va"]
    sigma_va = snap["sigma_va"]
    log10_sigma_va = snap["log10_sigma_va"]
elif "snr" in snap.files and "sigma" in snap.files and "log10_sigma" in snap.files:
    snr_va = snap["snr"]
    sigma_va = snap["sigma"]
    log10_sigma_va = snap["log10_sigma"]

Rva, P, L = X_va_std.shape
THETA_DIM = y_va_std.shape[1]
mdl.THETA_DIM = THETA_DIM  # ensure model uses correct theta dim

print("Loaded val snapshot:")
print("  X_va_std:", X_va_std.shape, "y_va_std:", y_va_std.shape)
print("  X_va_clean:", X_va_clean.shape, "X_va_noisy:", X_va_noisy.shape)
print("  y_va_full:", y_va_full.shape)
print("  P,L:", P, L, "| THETA_DIM:", THETA_DIM)
print("  Stats:",
      "X_mean/std:", tuple(X_mean.shape), tuple(X_std.shape),
      "| y_mean/std:", tuple(y_mean.shape), tuple(y_std.shape))

if snr_va is not None:
    print("  Noise metadata present: snr_va/sigma_va/log10_sigma_va:", snr_va.shape, sigma_va.shape, log10_sigma_va.shape)
else:
    print("  Noise metadata NOT present in validation snapshot (snr_va/sigma_va/log10_sigma_va missing).")

# bounds check
if VAL_INDEX < 0 or VAL_INDEX >= Rva:
    raise IndexError(f"VAL_INDEX={VAL_INDEX} out of range [0, {Rva-1}]")

# torch tensors
X_va_t = torch.as_tensor(X_va_std, dtype=torch.float32)
y_va_t = torch.as_tensor(y_va_std, dtype=torch.float32)

# dataset (two-mode compatibility)
if USE_TRUE_PHASE:
    raise NotImplementedError("Snapshot does not store phi_va. Keep USE_TRUE_PHASE=False.")
else:
    val_ds = TensorDataset(X_va_t, y_va_t)  # (xb, yb)

# ==========================================================
# PHASE PROVIDER (only if you trained with predicted-phase encoding)
# ==========================================================
phase_provider = None
if USE_PHASE_PROVIDER:
    PHASE_CKPT = os.path.join(PKG_PATH, "best_fast.pt")  # adjust if needed
    phase_provider = PhaseProvider(
        phase_ckpt_path=PHASE_CKPT,
        device=device,
        base_len=L,
    )

    # ---- CRITICAL FIX: PhaseProvider needs x_mean/x_std for unstandardizing EA inputs ----
    # Snapshot stores X_mean/X_std as (1, P*L) which matches flattened EA inputs.
    phase_provider.x_mean = X_mean.to(device=device, dtype=torch.float32)
    phase_provider.x_std  = X_std.to(device=device, dtype=torch.float32)

    print("PhaseProvider ready:", PHASE_CKPT)
    print("PhaseProvider x_mean/x_std set:", tuple(phase_provider.x_mean.shape), tuple(phase_provider.x_std.shape))
else:
    print("PhaseProvider disabled.")

# ==========================================================
# BUILD MODEL (must match training)
# ==========================================================
model = PosteriorNet(
    seq_len=L,
    use_phase=USE_PHASE,
    phase_provider=phase_provider,
    x_mean=X_mean,
    x_std=X_std,
).to(device)

# ==========================================================
# LOAD CHECKPOINT
# ==========================================================
ckpt = torch.load(SAVE_PATH, weights_only=False, map_location=device)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()
print("Loaded posterior model weights:", SAVE_PATH)

# ==========================================================
# PICK ONE VALIDATION EXAMPLE
# ==========================================================
xb_std, yb_std = val_ds[VAL_INDEX]
phib_std = None

xb_std = xb_std.unsqueeze(0).to(device)  # (1,P,L) standardized
yb_std = yb_std.unsqueeze(0).to(device)  # (1,D)   standardized

real_id = int(val_r[VAL_INDEX])
print(f"VAL_INDEX={VAL_INDEX} -> realization_id={real_id}")

# truth for targets in physical units
y_true = (yb_std * y_std.to(device) + y_mean.to(device)).squeeze(0)  # (D,)
y_true_np = y_true.detach().cpu().numpy()

# clean/noisy residuals for this validation sample (physical / unstandardized)
x_clean_np = X_va_clean[VAL_INDEX]  # (P,L)
x_noisy_np = X_va_noisy[VAL_INDEX]  # (P,L)

# full parameter vector for this realization in physical units
y_full_np = y_va_full[VAL_INDEX]    # (nparam,)

# noise metadata for this realization (if present)
if snr_va is not None:
    snr_i = int(snr_va[VAL_INDEX])
    sigma_i = float(sigma_va[VAL_INDEX])
    log10_sigma_i = float(log10_sigma_va[VAL_INDEX])
else:
    snr_i = None
    sigma_i = None
    log10_sigma_i = None

# ==========================================================
# SAMPLE POSTERIOR (REPRODUCIBLE)
# ==========================================================
torch.manual_seed(POST_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(POST_SEED)
np.random.seed(POST_SEED)

@torch.no_grad()
def sample_posterior(model, xb, phib, nsamples):
    # Preferred API (CNF PosteriorNet): model.sample(nsamples, xb, phase)
    if hasattr(model, "sample"):
        try:
            return model.sample(nsamples, xb, phib)
        except TypeError:
            # alternate keyword style
            return model.sample(n=nsamples, x=xb, phase=phib)

    # Fallback: h = cond(x), then flow.sample(n, h)
    phase_eff = None
    if getattr(model, "use_phase", False):
        if phib is not None:
            phase_eff = phib
        elif getattr(model, "phase_provider", None) is not None:
            phase_eff = model.phase_provider(xb)

    h = model.cond(xb, phase_eff)     # (1, ctx)
    return model.flow.sample(nsamples, h)

y_samp_std = sample_posterior(model, xb_std, phib_std, NSAMPLES).detach()

# ensure shape (NSAMPLES, D)
if y_samp_std.dim() == 3 and y_samp_std.size(0) == 1:
    y_samp_std = y_samp_std.squeeze(0)
if y_samp_std.shape[0] != NSAMPLES and y_samp_std.shape[1] == NSAMPLES:
    y_samp_std = y_samp_std.transpose(0, 1)
if tuple(y_samp_std.shape) != (NSAMPLES, THETA_DIM):
    raise RuntimeError(
        f"Unexpected y_samp_std shape {tuple(y_samp_std.shape)}; expected ({NSAMPLES}, {THETA_DIM})."
    )

# unstandardize posterior samples to physical units (targets only)
y_samp = y_samp_std * y_std.to(device) + y_mean.to(device)  # (NSAMPLES, D)
y_samp_np = y_samp.detach().cpu().numpy()

# ==========================================================
# SAVE POSTERIOR (NPZ)
# ==========================================================
npz_path = os.path.join(SAVE_POST_DIR, f"posterior_validx{VAL_INDEX:04d}.npz")

save_dict = dict(
    # posterior over TARGET parameters only (physical units)
    samples=y_samp_np,                 # (NSAMPLES, D)
    truth_targets=y_true_np,           # (D,)
    target_names=np.array(target_names),

    # picked validation sample residuals (physical / unstandardized)
    x_clean=x_clean_np,                # (P, L)
    x_noisy=x_noisy_np,                # (P, L)

    # full params for that realization (physical / unstandardized)
    y_full=y_full_np,                  # (nparam,)
    param_cols=np.array(param_cols) if param_cols is not None else None,

    # noise metadata for this realization
    snr=snr_i,
    sigma=sigma_i,
    log10_sigma=log10_sigma_i,

    # bookkeeping
    val_index=VAL_INDEX,
    realization_id=real_id,
    nsamples=NSAMPLES,
    seed=POST_SEED,
)

np.savez_compressed(npz_path, **save_dict)
print("Saved posterior:", npz_path)

# ==========================================================
# CORNER PLOT (targets only)
# ==========================================================
import corner

corner_path = os.path.join(SAVE_POST_DIR, f"corner_validx{VAL_INDEX:04d}.png")

fig = corner.corner(
    y_samp_np,
    labels=target_names,
    truths=y_true_np,
    show_titles=True,
    title_fmt=".3g",
    quantiles=[0.16, 0.5, 0.84],
)

title = f"CNF Posterior (val_idx={VAL_INDEX}, real_id={real_id}, seed={POST_SEED})"
if snr_i is not None:
    title += f" | SNR={snr_i} | sigma={sigma_i:.3g} | log10_sigma={log10_sigma_i:.3g}"
fig.suptitle(title, y=1.02)

fig.savefig(corner_path, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved corner:", corner_path)
print("Done. Outputs in:", SAVE_POST_DIR)
