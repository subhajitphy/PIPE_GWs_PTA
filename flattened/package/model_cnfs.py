#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

# ===================== Defaults / Config =====================

D_MODEL     = 128
NHEAD       = 4
DEPTH       = 4
CTX_DIM     = 256
FLOW_HIDDEN = 192

WEIGHTING_MODE   = "learned"   # "learned" or "manual"
ALPHA_POS_INIT   = 1.0
ALPHA_PHASE_INIT = 1.0


# ===================== Positional / Phase Encodings =====================

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        pe   = torch.zeros(max_len, d_model)
        pos  = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div  = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens just used for sequence length
        return self.pe[:, : x_tokens.size(1)]


class PhaseSinusoidalPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div, persistent=False)
        self.d_model = d_model

    def forward(self, phase_seq: torch.Tensor) -> torch.Tensor:
        """
        phase_seq: (B, S) in radians (wrapped or unwrapped)
        """
        B, S = phase_seq.shape
        angles = phase_seq.unsqueeze(-1) * self.div_term.view(1, 1, -1)
        pe = torch.zeros(B, S, self.d_model,
                         device=phase_seq.device, dtype=phase_seq.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe


# ===================== Conv Stem + Transformer Conditioner =====================

class ConvPoolStem(nn.Module):
    def __init__(self, in_ch=1, d_model=D_MODEL, k1=7, k2=5, pool=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model // 2, k1, padding=k1 // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, k2, padding=k2 // 2),
            nn.GELU(),
            nn.AvgPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        x = x.unsqueeze(1)           # (B,1,L)
        z = self.net(x)              # (B, C, L')
        return z.transpose(1, 2)     # (B, L', C)


class SeqTransformer(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        nhead=NHEAD,
        depth=DEPTH,
        use_posenc=True,
        use_phasepe=True,
        weighting_mode: str = WEIGHTING_MODE,
        alpha_pos: float = ALPHA_POS_INIT,
        alpha_phase: float = ALPHA_PHASE_INIT,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(2 * d_model),
            batch_first=True,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.use_posenc  = use_posenc
        self.use_phasepe = use_phasepe
        self.weighting_mode = weighting_mode

        self.posenc   = SinusoidalPE(d_model)      if use_posenc  else None
        self.phase_pe = PhaseSinusoidalPE(d_model) if use_phasepe else None

        # weights for positional and phase encoding
        if self.use_posenc:
            if weighting_mode == "learned":
                self.w_pos = nn.Parameter(torch.tensor(alpha_pos, dtype=torch.float32))
            else:
                self.register_buffer("w_pos", torch.tensor(alpha_pos, dtype=torch.float32))
        else:
            self.w_pos = None

        if self.use_phasepe:
            if weighting_mode == "learned":
                self.w_phase = nn.Parameter(torch.tensor(alpha_phase, dtype=torch.float32))
            else:
                self.register_buffer("w_phase", torch.tensor(alpha_phase, dtype=torch.float32))
        else:
            self.w_phase = None

    def forward(self, tokens: torch.Tensor, phase_seq: torch.Tensor | None = None):
        """
        tokens: (B, L', d_model) from ConvPoolStem
        phase_seq: (B, L') or None (already downsampled / pooled)
        """
        B = tokens.size(0)
        h = tokens

        # positional encoding
        if self.use_posenc and (self.posenc is not None):
            pos = self.posenc(h)  # (1, L', d_model)
            if self.w_pos is not None:
                h = h + self.w_pos * pos
            else:
                h = h + pos

        # phase encoding
        if self.use_phasepe and (self.phase_pe is not None) and (phase_seq is not None):
            ph_enc = self.phase_pe(phase_seq)  # (B, L', d_model)
            if self.w_phase is not None:
                h = h + self.w_phase * ph_enc
            else:
                h = h + ph_enc

        cls = self.cls.expand(B, 1, -1)
        h   = torch.cat([cls, h], dim=1)      # (B, 1+L', d_model)
        z   = self.enc(h)
        cls_tok  = z[:, 0]
        mean_tok = z[:, 1:].mean(dim=1)
        return torch.cat([cls_tok, mean_tok], dim=1)   # (B, 2*d_model)


class Conditioner(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        nhead=NHEAD,
        depth=DEPTH,
        out_dim=CTX_DIM,
        use_posenc=True,
        use_phasepe=True,
        pool=8,
        weighting_mode: str = WEIGHTING_MODE,
        alpha_pos: float = ALPHA_POS_INIT,
        alpha_phase: float = ALPHA_PHASE_INIT,
    ):
        super().__init__()
        self.stem = ConvPoolStem(in_ch=1, d_model=d_model, pool=pool)
        self.backbone = SeqTransformer(
            d_model=d_model,
            nhead=nhead,
            depth=depth,
            use_posenc=use_posenc,
            use_phasepe=use_phasepe,
            weighting_mode=weighting_mode,
            alpha_pos=alpha_pos,
            alpha_phase=alpha_phase,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, L) – standardized timeseries
        phase: (B, L) or None – phase sequence at same length as x
        """
        tokens = self.stem(x)            # (B, L', d_model)
        if phase is not None:
            Lp = tokens.size(1)
            phase_pooled = F.adaptive_avg_pool1d(
                phase.unsqueeze(1), output_size=Lp
            ).squeeze(1)
        else:
            phase_pooled = None

        h = self.backbone(tokens, phase_pooled)
        return self.proj(h)              # (B, out_dim)


# ===================== Continuous Normalizing Flow =====================

class CNF_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=FLOW_HIDDEN, depth=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    """
    CNF vector field with Hutchinson trace estimator for dlogp/dt.
    """
    def __init__(self, theta_dim: int, ctx_dim: int, hidden: int = FLOW_HIDDEN):
        super().__init__()
        self.theta_dim = theta_dim
        self.ctx_dim   = ctx_dim
        self.net = CNF_MLP(theta_dim + ctx_dim + 1, theta_dim,
                           hidden=hidden, depth=2)

    def forward(self, t: torch.Tensor, states):
        """
        states: (y, logp, h)
        y: (B,D), logp: (B,), h: (B,ctx)
        """
        y, logp, h = states

        # enable gradients even under outer no_grad
        with torch.enable_grad():
            y = y.requires_grad_(True)

            B = y.size(0)
            t_feat = torch.ones(B, 1, device=y.device, dtype=y.dtype) * t
            inp = torch.cat([y, h, t_feat], dim=1)   # (B, D+ctx+1)
            f   = self.net(inp)                      # dy/dt

            # Hutchinson trace: eps ~ N(0,I)
            eps    = torch.randn_like(y)
            scalar = (f * eps).sum()
            g = torch.autograd.grad(
                scalar, y, create_graph=True, retain_graph=True
            )[0]
            trace = (g * eps).sum(dim=1)             # (B,)

        dlogp = -trace
        dh    = torch.zeros_like(h)
        return f, dlogp, dh


class ConditionalCNF(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        ctx_dim: int = CTX_DIM,
        hidden: int = FLOW_HIDDEN,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        method: str = "rk4",
        step_size: float = 0.1,
    ):
        """
        Faster CNF:
          - rk4 fixed-step (if method='rk4')
          - looser tolerances
          - Hutchinson trace estimator
        """
        super().__init__()
        self.theta_dim = theta_dim
        self.ctx_dim   = ctx_dim
        self.func      = ODEFunc(theta_dim, ctx_dim, hidden)
        self.atol      = atol
        self.rtol      = rtol
        self.method    = method
        self.step_size = step_size

    def _base_logprob(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.theta_dim * math.log(2 * math.pi)

    def _odeint(self, y0, t_span: torch.Tensor):
        options = {"step_size": self.step_size} if self.method == "rk4" else None
        return odeint(
            self.func,
            y0,
            t_span,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=options,
        )

    def to_base(self, theta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Map theta -> z0 in base space (for diagnostics & per-target MSE).
        """
        B = theta.size(0)
        t_span = torch.tensor([1.0, 0.0], device=theta.device)
        logp0  = torch.zeros(B, device=theta.device, dtype=theta.dtype)
        y0     = (theta, logp0, h)
        yt, logpt, ht = self._odeint(y0, t_span)
        z0 = yt[-1]
        return z0

    def log_prob(self, theta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, D)
        h:     (B, ctx)
        returns: (B,)
        """
        B = theta.size(0)
        t_span = torch.tensor([1.0, 0.0], device=theta.device)
        logp0  = torch.zeros(B, device=theta.device, dtype=theta.dtype)
        y0     = (theta, logp0, h)
        yt, logpt, ht = self._odeint(y0, t_span)
        z0         = yt[-1]
        delta_logp = logpt[-1]
        return self._base_logprob(z0) - delta_logp

    @torch.no_grad()
    def sample(self, n: int, h: torch.Tensor) -> torch.Tensor:
        """
        Sample theta ~ p(theta | h) using reverse ODE from base N(0,I).
        """
        device = h.device
        if h.size(0) == 1:
            H = h.repeat(n, 1)
        elif h.size(0) != n:
            reps = (n + h.size(0) - 1) // h.size(0)
            H = h.repeat(reps, 1)[:n]
        else:
            H = h

        z0    = torch.randn(n, self.theta_dim, device=device)
        logp0 = self._base_logprob(z0)
        t_span = torch.tensor([0.0, 1.0], device=device)
        yT, logpT, hT = self._odeint((z0, logp0, H), t_span)
        theta = yT[-1]
        return theta


# ===================== Posterior Network =====================

class PosteriorNet(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        ctx_dim: int = CTX_DIM,
        d_model: int = D_MODEL,
        nhead: int   = NHEAD,
        depth: int   = DEPTH,
        flow_hidden: int = FLOW_HIDDEN,
        use_phase: bool = True,
        phase_provider=None,
        weighting_mode: str = WEIGHTING_MODE,
        alpha_pos: float = ALPHA_POS_INIT,
        alpha_phase: float = ALPHA_PHASE_INIT,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        method: str = "rk4",
        step_size: float = 0.1,
    ):
        super().__init__()
        self.use_phase      = use_phase
        self.phase_provider = phase_provider

        self.cond = Conditioner(
            d_model=d_model,
            nhead=nhead,
            depth=depth,
            out_dim=ctx_dim,
            use_posenc=True,
            use_phasepe=True,
            pool=8,
            weighting_mode=weighting_mode,
            alpha_pos=alpha_pos,
            alpha_phase=alpha_phase,
        )

        self.flow = ConditionalCNF(
            theta_dim=theta_dim,
            ctx_dim=ctx_dim,
            hidden=flow_hidden,
            atol=atol,
            rtol=rtol,
            method=method,
            step_size=step_size,
        )

    def _maybe_get_phase(self, x: torch.Tensor, phase: torch.Tensor | None):
        """
        Logic:
        - If use_phase=False -> no phase
        - If phase provided (from loader / prepare_batch) -> use it
        - Else if phase_provider exists -> predict from x
        - Else -> no phase
        """
        if not self.use_phase:
            return None
        if phase is not None:
            return phase
        if self.phase_provider is not None:
            with torch.no_grad():
                ph_pred = self.phase_provider(x)   # (B, L)
            return ph_pred
        return None

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, phase: torch.Tensor | None = None):
        phase_eff = self._maybe_get_phase(x, phase)
        h = self.cond(x, phase_eff)
        return self.flow.log_prob(theta, h)

    @torch.no_grad()
    def sample(self, n: int, x: torch.Tensor, phase: torch.Tensor | None = None):
        phase_eff = self._maybe_get_phase(x, phase)
        h = self.cond(x, phase_eff)
        return self.flow.sample(n, h)


# ======================= Batch utilities ===============================

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


def prepare_batch(batch, device: torch.device):
    """
    - Moves batch to device
    - Computes per-sample z-normalization for x
    Returns: xb_norm, phib_or_None, yb
    """
    xb, phib, yb = _unpack_batch(batch)

    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    if phib is not None:
        phib = phib.to(device, non_blocking=True)

    mu = xb.mean(dim=1, keepdim=True)
    sd = xb.std(dim=1, keepdim=True).clamp_min(1e-6)
    xb_norm = (xb - mu) / sd

    return xb_norm, phib, yb


__all__ = [
    "D_MODEL", "NHEAD", "DEPTH", "CTX_DIM", "FLOW_HIDDEN",
    "WEIGHTING_MODE", "ALPHA_POS_INIT", "ALPHA_PHASE_INIT",
    "Conditioner", "ConditionalCNF", "PosteriorNet", "prepare_batch",
]
