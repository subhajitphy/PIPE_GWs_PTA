#!/usr/bin/env python
# coding: utf-8
"""
phase_pred.py

Loader + inference utilities for the NEW phase/SNR prediction model:
    - Model takes ONLY X-noisy
    - Model RETURNS (phase_vec, snr_log10_pred)
    - SNR is NOT an input anymore
"""

import numpy as np
import torch
import torch.nn as nn
import math


# ================================================================
# Sinusoidal Positional Encoding
# ================================================================
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


# ================================================================
# Slow Transformer (NOT used normally, but included for completeness)
# ================================================================
class PhaseTransformer(nn.Module):
    def __init__(self, seq_len, d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pe = SinusoidalPE(seq_len, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=d_ff, dropout=p_drop,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)

        # Phase head
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

        # NEW: SNR head
        self.snr_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        z = self.in_proj(x)
        z = self.pe(z)
        z = self.encoder(z)

        y_vec = self.phase_head(z)
        snr_log10 = self.snr_head(z.mean(dim=1)).squeeze(-1)

        return y_vec, snr_log10


# ================================================================
# Fast Transformer (downsample + upsample)
# ================================================================
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


# ================================================================
# Helper: convert 2D vecs → phase
# ================================================================
def _angle_from_vec(y):
    y = y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=1e-7)
    return torch.atan2(y[...,1], y[...,0])


# ================================================================
# Model loader
# ================================================================
def load_phase_model(ckpt_path, device="cpu", use_fast=True):
    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    T_NEW = int(ckpt["T_NEW"])
    DS_STRIDE = int(ckpt["DS_STRIDE"])

    if use_fast:
        model = PhaseTransformerFast(
            seq_len=T_NEW, ds_stride=DS_STRIDE,
            d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1
        )
    else:
        model = PhaseTransformer(seq_len=T_NEW)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    mu_x = ckpt["mu_x"].to(dev)
    std_x = ckpt["std_x"].to(dev)

    return model, mu_x, std_x, T_NEW, DS_STRIDE


# ================================================================
# Main prediction function
# ================================================================
@torch.no_grad()
def predict_phase(model, mu_x, std_x, x_raw,
                  device="cpu", unwrap=True, batch_dim=False):
    dev = torch.device(device)
    model = model.to(dev).eval()

    # Convert input
    if isinstance(x_raw, torch.Tensor):
        x = x_raw.to(dev)
    else:
        x = torch.tensor(x_raw, dtype=torch.float32, device=dev)

    if not batch_dim:
        x = x.unsqueeze(0)

    x_std = (x - mu_x) / std_x
    x_std = x_std.unsqueeze(-1)      # [B,T,1]

    y_vec, snr_log10_pred = model(x_std)     # <-- NEW: model output

    phi = _angle_from_vec(y_vec)
    phi = phi.cpu().numpy()
    snr_pred = (10.0**snr_log10_pred.cpu().numpy())   # convert back to linear SNR

    if unwrap:
        phi = np.unwrap(phi, axis=1)

    if not batch_dim:
        return phi[0], snr_pred[0]

    return phi, snr_pred

