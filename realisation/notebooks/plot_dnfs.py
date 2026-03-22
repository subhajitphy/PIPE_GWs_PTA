import os, sys, random, platform
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

import model_dnfs as mdl
from model_dnfs import PosteriorNet

# If you need phase encoding at inference (only if model was trained with it):
from phase_pred import PhaseProvider  # safe import even if disabled

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
AMP_ENABLED = (device.type == "cuda")
print("Device:", device, "| AMP:", AMP_ENABLED)

# ==========================================================
# CONFIG
# ==========================================================
SAVE_DIR = "dnfs_pred_phase_realisation"
SAVE_PATH = os.path.join(SAVE_DIR, "best_posterior_flow_dnfs_pred_phase.pt")

# saved validation snapshot (created once from your training pipeline)
#VAL_SNAP_PATH = "validation_set.npz"
VAL_SNAP_PATH = "/scratch/projects/CFP03/CFP03-CF-051/projects/mis/REALISATION/SBI/SBI_NO_ENC/validation_set.npz"
# inference target


import argparse

parser = argparse.ArgumentParser(description="Run posterior inference for a validation sample")
parser.add_argument(
    "val_index",
    type=int,
    help="Validation index (e.g. 200)"
)
args = parser.parse_args()

VAL_INDEX = args.val_index

NSAMPLES  = 5000
POST_SEED = SEED + 999  # fixed seed => reproducible posterior draws

# phase encoding flags (MUST match training!)
USE_TRUE_PHASE     = False
USE_PHASE_PROVIDER = True
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
# NOTE: in your updated training script these keys are X_va_std / y_va_std
# but we also keep backward-compat if you still saved as X_va / y_va.
if "X_va_std" in snap.files:
    X_va_std = snap["X_va_std"]  # (Rva,P,L) standardized (from noisy)
else:
    X_va_std = snap["X_va"]      # legacy key

if "y_va_std" in snap.files:
    y_va_std = snap["y_va_std"]  # (Rva,D) standardized
else:
    y_va_std = snap["y_va"]      # legacy key

# --- unstandardized snapshots ---
# these are requested: clean+noisy residuals of selected validation sample
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

Rva, P, L = X_va_std.shape
THETA_DIM = y_va_std.shape[1]
mdl.THETA_DIM = THETA_DIM

print("Loaded val snapshot:")
print("  X_va_std:", X_va_std.shape, "y_va_std:", y_va_std.shape)
print("  X_va_clean:", X_va_clean.shape, "X_va_noisy:", X_va_noisy.shape)
print("  y_va_full:", y_va_full.shape)
print("  P,L:", P, L, "| THETA_DIM:", THETA_DIM)
print("  Stats:",
      "X_mean/std:", tuple(X_mean.shape), tuple(X_std.shape),
      "| y_mean/std:", tuple(y_mean.shape), tuple(y_std.shape))

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
    print("PhaseProvider ready:", PHASE_CKPT)
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
item = val_ds[VAL_INDEX]
xb_std, yb_std = item
phib_std = None

xb_std = xb_std.unsqueeze(0).to(device)  # (1,P,L) standardized
yb_std = yb_std.unsqueeze(0).to(device)  # (1,D)   standardized

real_id = int(val_r[VAL_INDEX])
print(f"VAL_INDEX={VAL_INDEX} -> realization_id={real_id}")

# --- truth for targets in physical units (from standardized yb_std) ---
y_true = (yb_std * y_std.to(device) + y_mean.to(device)).squeeze(0)  # (D,)
y_true_np = y_true.detach().cpu().numpy()

# --- clean/noisy residuals for this chosen validation sample (physical / unstandardized) ---
x_clean_np = X_va_clean[VAL_INDEX]  # (P,L)
x_noisy_np = X_va_noisy[VAL_INDEX]  # (P,L)

# --- full parameter vector for this realization in physical units ---
y_full_np = y_va_full[VAL_INDEX]    # (nparam,)

# ==========================================================
# SAMPLE POSTERIOR (REPRODUCIBLE)
# ==========================================================
torch.manual_seed(POST_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(POST_SEED)
np.random.seed(POST_SEED)

@torch.no_grad()
def sample_posterior(model, xb, phib, nsamples):
    # Try common API: model.sample(nsamples, xb, phib)
    if hasattr(model, "sample"):
        try:
            return model.sample(nsamples, xb, phib)
        except TypeError:
            return model.sample(xb=xb, phib=phib, nsamples=nsamples)

    # Fallback: z -> inverse flow given context h = cond(x)
    h = model.cond(xb, phib if getattr(model, "use_phase", False) else None)  # (1,H)
    z = torch.randn(nsamples, THETA_DIM, device=xb.device)
    y_s, _ = model.flow.inv_from_z(z, h.expand(nsamples, -1))
    return y_s

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
# SAVE POSTERIOR (NPZ) — include clean/noisy residuals + y_va_full
# ==========================================================
npz_path = os.path.join(SAVE_POST_DIR, f"posterior_validx{VAL_INDEX:04d}.npz")
np.savez_compressed(
    npz_path,
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

    # bookkeeping
    val_index=VAL_INDEX,
    realization_id=real_id,
    nsamples=NSAMPLES,
    seed=POST_SEED,
)
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
fig.suptitle(
    f"Posterior (val_idx={VAL_INDEX}, real_id={real_id}, seed={POST_SEED})",
    y=1.02,
)
fig.savefig(corner_path, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved corner:", corner_path)
print("Done. Outputs in:", SAVE_POST_DIR)
