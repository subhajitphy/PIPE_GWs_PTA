#!/usr/bin/env python
# coding: utf-8
"""
phase_pred.py

Lightweight inference utilities for the *realisation-mode* phase model.

Expected checkpoint format (saved by run_ph_pred_realisation.py):
  ckpt = {
    "model_state_dict": ...,
    "mu_x": Tensor shape (1, P*L),
    "std_x": Tensor shape (1, P*L),
    "P": int,
    "L": int,
    "USE_FAST": bool,
    "DS_STRIDE": int,
    "D_MODEL": int,
    "DEPTH": int,
    "HEADS": int,
    "D_FF": int,
    ...
  }

Model contract:
  Input  x: (B,P,L)  (float32)
  Output y: (B,L,2)  (cos φ, sin φ)

Public:
  - load_phase_model(ckpt_path, device=None) -> PhaseModelBundle
  - predict_phase(x, bundle, assume_standardized=False, return_unit=False) -> phi
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# --------------------------- Positional Encoding ---------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,T,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def _normalize_unit(y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=eps)


# --------------------------- Models ---------------------------

class PhaseTransformerRealisation(nn.Module):
    """Full-rate transformer: (B,P,L) -> (B,L,2)."""
    def __init__(
        self,
        P: int,
        L: int,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        d_ff: int = 512,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.P = P
        self.L = L

        self.in_proj = nn.Linear(P, d_model)
        self.pe = SinusoidalPE(L, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()  # (B,L,P)
        z = self.in_proj(x)                  # (B,L,D)
        z = self.pe(z)
        z = self.encoder(z)
        return self.head(z)                  # (B,L,2)


class PhaseTransformerRealisationFast(nn.Module):
    """Conv-downsampled transformer: (B,P,L) -> (B,L,2)."""
    def __init__(
        self,
        P: int,
        L: int,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        d_ff: int = 512,
        p_drop: float = 0.1,
        ds_stride: int = 4,
    ):
        super().__init__()
        assert L % ds_stride == 0, "L must be divisible by ds_stride"
        self.P = P
        self.L = L
        self.S = ds_stride
        self.Ls = L // ds_stride

        self.in_proj = nn.Linear(P, d_model)

        self.ds_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=ds_stride, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.post_ds_norm = nn.LayerNorm(d_model)
        self.pe = SinusoidalPE(self.Ls, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head_lowrate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.upsampler = nn.Upsample(size=L, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()  # (B,L,P)
        z = self.in_proj(x)                  # (B,L,D)

        z = z.transpose(1, 2)                # (B,D,L)
        z = self.ds_conv(z)                  # (B,D,Ls)
        z = z.transpose(1, 2)                # (B,Ls,D)
        z = self.post_ds_norm(z)
        z = self.pe(z)

        z = self.encoder(z)                  # (B,Ls,D)
        y_low = self.head_lowrate(z)         # (B,Ls,2)
        y = self.upsampler(y_low.transpose(1, 2)).transpose(1, 2)  # (B,L,2)
        return y


# --------------------------- Bundle ---------------------------

@dataclass
class PhaseModelBundle:
    model: nn.Module
    mu_x: torch.Tensor   # (1, P*L)
    std_x: torch.Tensor  # (1, P*L)
    P: int
    L: int
    use_fast: bool
    ds_stride: int

    def to(self, device: torch.device) -> "PhaseModelBundle":
        self.model.to(device)
        self.mu_x = self.mu_x.to(device)
        self.std_x = self.std_x.to(device)
        return self


def _build_model_from_ckpt(ckpt: Dict) -> Tuple[nn.Module, int, int, bool, int]:
    P = int(ckpt["P"])
    L = int(ckpt["L"])
    use_fast = bool(ckpt.get("USE_FAST", True))
    ds_stride = int(ckpt.get("DS_STRIDE", 4))

    d_model = int(ckpt.get("D_MODEL", 128))
    depth   = int(ckpt.get("DEPTH", 4))
    heads   = int(ckpt.get("HEADS", 4))
    d_ff    = int(ckpt.get("D_FF", 512))
    p_drop  = float(ckpt.get("P_DROP", 0.1))

    if use_fast:
        model = PhaseTransformerRealisationFast(
            P=P, L=L, d_model=d_model, depth=depth, heads=heads,
            d_ff=d_ff, p_drop=p_drop, ds_stride=ds_stride
        )
    else:
        model = PhaseTransformerRealisation(
            P=P, L=L, d_model=d_model, depth=depth, heads=heads,
            d_ff=d_ff, p_drop=p_drop
        )
    return model, P, L, use_fast, ds_stride


# --------------------------- Public API ---------------------------

def load_phase_model(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> PhaseModelBundle:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    if map_location is None:
        map_location = device

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=map_location)

    model, P, L, use_fast, ds_stride = _build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    mu_x = ckpt["mu_x"].detach().float()
    std_x = ckpt["std_x"].detach().float()

    return PhaseModelBundle(
        model=model,
        mu_x=mu_x,
        std_x=std_x,
        P=P,
        L=L,
        use_fast=use_fast,
        ds_stride=ds_stride,
    ).to(device)


@torch.no_grad()
def predict_phase(
    x: torch.Tensor,
    bundle: PhaseModelBundle,
    assume_standardized: bool = False,
    return_unit: bool = False,
):
    """
    Predict φ(t) for realisation-mode input.

    Args:
      x: (P,L) or (B,P,L)
      assume_standardized: if False, standardize with bundle.(mu_x,std_x)
      return_unit: also return unit (cos,sin) if True

    Returns:
      phi: (B,L) in radians
      (optional) y_unit: (B,L,2)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"x must be (P,L) or (B,P,L); got {tuple(x.shape)}")

    B, P, L = x.shape
    if (P != bundle.P) or (L != bundle.L):
        raise ValueError(f"Input (P,L)=({P},{L}) != ckpt ({bundle.P},{bundle.L})")

    x = x.to(bundle.mu_x.device, dtype=torch.float32)

    if not assume_standardized:
        x_flat = x.reshape(B, -1)
        x_flat = (x_flat - bundle.mu_x) / bundle.std_x
        x = x_flat.reshape(B, P, L)

    y = bundle.model(x)          # (B,L,2)
    y_unit = _normalize_unit(y)  # enforce unit
    phi = torch.atan2(y_unit[..., 1], y_unit[..., 0])  # (B,L)

    return (phi, y_unit) if return_unit else phi



# ==========================================================
# PhaseProvider (included inside phase_pred.py)
# ==========================================================

# phase_provider_simple.py
import torch
import torch.nn as nn
from phase_pred import load_phase_model, predict_phase


class PhaseProvider(nn.Module):
    """
    Inputs:
      - (L,)
      - (B,L)
      - (P,L)
      - (B,P,L)

    Outputs:
      - (L,)   if input was (L,) or (P,L)
      - (B,L)  if input was (B,L) or (B,P,L)

    Notes:
      - If input has pulsar dimension (P), the phase model is assumed to return
        ONE shared phase series for the realisation:
          predict_phase((P,L), bundle) -> (L,)
      - Unstandardized inputs are standardized internally using externally
        supplied mu_x/std_x (shape (1, P*L)), exactly as in your notebook. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        phase_ckpt_path: str = "./best_fast.pt",
        *,
        mu_x: torch.Tensor | None = None,   # (1, P*L)
        std_x: torch.Tensor | None = None,  # (1, P*L)
        P: int | None = None,
        L: int | None = None,
        input_is_standardized: bool = True,
        pulsar_index: int = 0,   # used only for unstandardized (L,) or (B,L)
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.input_is_standardized = bool(input_is_standardized)
        self.pulsar_index = int(pulsar_index)

        # ---- load phase model ----
        self.bundle = load_phase_model(phase_ckpt_path, device=str(self.device))

        # ---- store stats (if provided) ----
        self.P = P
        self.L = L

        if not self.input_is_standardized:
            if mu_x is None or std_x is None or P is None or L is None:
                raise ValueError(
                    "For unstandardized inputs, mu_x, std_x, P, and L must be provided."
                )

            self.register_buffer("mu_x", mu_x.detach().float(), persistent=True)
            self.register_buffer("std_x", std_x.detach().float(), persistent=True)

    # ------------------------------------------------------------------
    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize using flattened (P*L) mu/std."""
        if self.input_is_standardized:
            return x

        P, L = self.P, self.L
        mu = self.mu_x.to(x.device)
        sd = self.std_x.to(x.device)

        # -------- (L,) --------
        if x.dim() == 1:
            p = max(0, min(self.pulsar_index, P - 1))
            mu_sl = mu[:, p * L : (p + 1) * L].view(L)
            sd_sl = sd[:, p * L : (p + 1) * L].view(L)
            return (x - mu_sl) / sd_sl

        # -------- (P,L) --------
        if x.dim() == 2 and x.shape == (P, L):
            xf = x.reshape(1, P * L)
            xf = (xf - mu) / sd
            return xf.reshape(P, L)

        # -------- (B,L) --------
        if x.dim() == 2 and x.shape[1] == L:
            p = max(0, min(self.pulsar_index, P - 1))
            mu_sl = mu[:, p * L : (p + 1) * L]
            sd_sl = sd[:, p * L : (p + 1) * L]
            return (x - mu_sl) / sd_sl

        # -------- (B,P,L) --------
        if x.dim() == 3 and x.shape[1:] == (P, L):
            B = x.shape[0]
            xf = x.reshape(B, P * L)
            xf = (xf - mu) / sd
            return xf.reshape(B, P, L)

        raise ValueError(f"Unsupported shape {tuple(x.shape)} for standardization")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).float().to(self.device)
        x = self._standardize(x)

        # -------- (L,) -> (L,) --------
        if x.dim() == 1:
            return predict_phase(x.unsqueeze(0), self.bundle)[0]

        # -------- 2D --------
        if x.dim() == 2:
            # (P,L) -> (L,)
            if self.P is not None and x.shape == (self.P, self.L):
                return predict_phase(x, self.bundle)[0]

            # (B,L) -> (B,L)
            B, L = x.shape
            out = torch.empty((B, L), device=self.device)
            for b in range(B):
                out[b] = predict_phase(x[b].unsqueeze(0), self.bundle)[0]
            return out

        # -------- (B,P,L) -> (B,L) --------
        if x.dim() == 3:
            B, P, L = x.shape
            out = torch.empty((B, L), device=self.device)
            for b in range(B):
                out[b] = predict_phase(x[b], self.bundle)[0]
            return out

        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")




class PhaseProvider(nn.Module):


    def __init__(
        self,
        phase_ckpt_path: str,
        *,
        device: str | torch.device = "cpu",
        base_len: int | None = None,   # per-pulsar length (L0). If None, uses ckpt bundle.L.
        snr_enabled: bool = False,     # set True only if you later add SNR prediction
    ):
        super().__init__()
        self.device = torch.device(device)
        self.bundle = load_phase_model(phase_ckpt_path, device=self.device)

        # base_len = per-pulsar series length
        self.base_len = int(base_len) if base_len is not None else int(self.bundle.L)
        self.snr_enabled = bool(snr_enabled)

        # These will be set by EAUnifiedPE before calling forward():
        #   phase_provider.x_mean = EA.x_mean_buf
        #   phase_provider.x_std  = EA.x_std_buf
        self.x_mean = None
        self.x_std = None

        self.last_snr_pred = None

    # -------------------------- helpers --------------------------

    def _as_2d_stats(self, t: torch.Tensor, D: int) -> torch.Tensor:
        """
        Ensure stats are shaped (1, D) on correct device/dtype.
        Accepts (D,), (1,D), or anything broadcastable to (1,D).
        """
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        if t.ndim == 1:
            t = t.view(1, -1)
        if t.shape[1] != D:
            raise ValueError(f"PhaseProvider: stats length mismatch: got {tuple(t.shape)} but need (1,{D})")
        return t

    def _unstandardize_ea(self, x_std_in: torch.Tensor, D_flat: int) -> torch.Tensor:
        """
        x_std_in: (B, D_flat) standardized by EA stats.
        returns:  (B, D_flat) unstandardized
        """
        if self.x_mean is None or self.x_std is None:
            # If you forgot to pass x_mean/x_std into EAUnifiedPE, it won't be able to set them here.
            raise ValueError("PhaseProvider: x_mean/x_std not set. Pass x_mean/x_std to EAUnifiedPE so it can assign them.")

        mu = self._as_2d_stats(self.x_mean, D_flat)
        sd = self._as_2d_stats(self.x_std,  D_flat)
        return x_std_in * sd + mu

    def _infer_P_from_flat(self, D_flat: int) -> int:
        L0 = self.base_len
        if D_flat % L0 != 0:
            raise ValueError(f"PhaseProvider: cannot infer P: D_flat={D_flat} not divisible by base_len={L0}")
        return D_flat // L0

    # -------------------------- forward --------------------------

    @torch.no_grad()
    def forward(self, x_std_in: torch.Tensor) -> torch.Tensor:
        """
        Input:  standardized x, shape (B,L) or (B,P,L) or (B,P*L)
        Output: phi, same shape as input
        """
        x = torch.as_tensor(x_std_in, device=self.device, dtype=torch.float32)

        # -------- normalize shapes to a canonical flat view for unstandardization --------
        if x.ndim == 2:
            B, D = x.shape

            # Case A: (B, P*L0) flattened
            if D != self.base_len:
                P = self._infer_P_from_flat(D)
                x_flat = x
                x_unstd_flat = self._unstandardize_ea(x_flat, D)
                x_unstd = x_unstd_flat.view(B, P, self.base_len)  # (B,P,L0)

                phi_BL = predict_phase(x_unstd, self.bundle, assume_standardized=False)  # (B,L0)

                # return with SAME shape as input: (B, P*L0)
                phi_rep = phi_BL.unsqueeze(1).expand(B, P, self.base_len).reshape(B, P * self.base_len)

                # last_snr_pred: (B,P)
                self.last_snr_pred = torch.zeros((B, P), device=self.device, dtype=torch.float32)

                return phi_rep

            # Case B: (B, L0) (single per-pulsar series)
            x_flat = x
            x_unstd_flat = self._unstandardize_ea(x_flat, self.base_len)  # (B,L0)

            # Phase model expects (B,P,L0). We replicate the single series across P.
            P = int(self.bundle.P)
            x_unstd = x_unstd_flat.unsqueeze(1).expand(B, P, self.base_len)  # (B,P,L0)

            phi_BL = predict_phase(x_unstd, self.bundle, assume_standardized=False)  # (B,L0)

            # last_snr_pred: (B,)
            self.last_snr_pred = torch.zeros((B,), device=self.device, dtype=torch.float32)

            return phi_BL

        elif x.ndim == 3:
            # (B,P,L0)
            B, P, L0 = x.shape
            if L0 != self.base_len:
                raise ValueError(f"PhaseProvider: expected L={self.base_len} but got {L0}")
            # Unstandardize using EA stats which are assumed to be for flattened (P*L0)
            x_flat = x.reshape(B, P * L0)
            x_unstd_flat = self._unstandardize_ea(x_flat, P * L0)
            x_unstd = x_unstd_flat.view(B, P, L0)

            phi_BL = predict_phase(x_unstd, self.bundle, assume_standardized=False)  # (B,L0)

            # return same shape as input: (B,P,L0)
            phi_BPL = phi_BL.unsqueeze(1).expand(B, P, L0)

            # last_snr_pred: (B,P)
            self.last_snr_pred = torch.zeros((B, P), device=self.device, dtype=torch.float32)

            return phi_BPL

        else:
            raise ValueError(f"PhaseProvider: unsupported input shape {tuple(x.shape)}")