#!/usr/bin/env python
# coding: utf-8
"""
model_dnfs.py  (CNF VERSION)

You asked: "run_dnfs.py used model_dnfs.py; modify it to Continuous Normalizing Flow
(following model_cnfs.py)".

This file keeps the SAME high-level API as your DNF version:
- THETA_DIM (overwritten by run script)
- CTX_DIM
- PosteriorNet.log_prob(theta, x, phase=None)
- PosteriorNet.sample(n, x, phase=None)

But the flow is now a Conditional CNF (ODE-based).
Conditioner stays EAUnifiedPE-based (as in your model_dnfs.py).

Notes:
- Requires: torchdiffeq
- CNF uses Hutchinson trace estimator for dlogp/dt.
- AMP/fp16 is usually NOT stable for CNFs -> keep training in fp32.
"""

import math
import torch
import torch.nn as nn
from torchdiffeq import odeint

# ====== GLOBALS (kept compatible with your existing run_dnfs.py) ======
THETA_DIM   = 4      # overwritten by run script: mdl.THETA_DIM = len(target_names)
CTX_DIM     = 64
FLOW_HIDDEN = 192

# ---- EA conditioner defaults ----
EA_PATCH   = 20
EA_HEADS   = 8
EA_DMODEL  = 128
EA_DEPTH   = 4
EA_DIM_FF  = 256
EA_MEM     = 40

# CNF solver defaults (fast-ish)
CNF_ATOL      = 1e-3
CNF_RTOL      = 1e-3
CNF_METHOD    = "rk4"      # fixed-step RK4 (fast)
CNF_STEP_SIZE = 0.1

# Import EA model / PhaseProvider (your project)
from ea_model_hy_new import EAUnifiedPE


# ===================== EAUnifiedPE Conditioner =====================
class EAConditioner(nn.Module):
    """
    Conditioner: (x, phase) -> h (CTX_DIM)

    Supports x:
      - (B, L)
      - (B, P, L)

    Phase:
      - None (EAUnifiedPE may use internal phase_provider if provided)
      - (B, L) or (B, P, L) depending on your EAUnifiedPE implementation
    """
    def __init__(
        self,
        seq_len: int,
        ctx_dim: int = CTX_DIM,
        *,
        patch: int = EA_PATCH,
        heads: int = EA_HEADS,
        d_model: int = EA_DMODEL,
        depth: int = EA_DEPTH,
        dim_ff: int = EA_DIM_FF,
        mem_size: int = EA_MEM,
        weighting: str = "learned",
        alpha_pos: float = 1.0,
        alpha_phase: float = 1.0,
        alpha_snr: float = 1.0,
        use_posenc: bool = True,
        use_phasepe: bool = True,
        use_snrenc: bool = False,
        cnn_stem: bool = False,
        phase_provider=None,
        x_mean=None,
        x_std=None,
    ):
        super().__init__()
        self.net = EAUnifiedPE(
            seq_len=seq_len,
            out_dim=ctx_dim,
            patch=patch,
            heads=heads,
            d_model=d_model,
            depth=depth,
            dim_ff=dim_ff,
            mem_size=mem_size,
            weighting=weighting,
            alpha_pos=alpha_pos,
            alpha_phase=alpha_phase,
            alpha_snr=alpha_snr,
            use_posenc=use_posenc,
            use_phasepe=use_phasepe,
            use_snrenc=use_snrenc,
            cnn_stem=cnn_stem,
            phase_provider=phase_provider,
            x_mean=x_mean,
            x_std=x_std,
        )

    def forward(self, x, phase=None):
        return self.net(x, phase)  # (B, CTX_DIM)


# ===================== Continuous Normalizing Flow (CNF) =====================

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
    State: (y, logp, h) where:
      y:    (B, D)
      logp: (B,)
      h:    (B, ctx) (constant along t)
    """
    def __init__(self, theta_dim: int, ctx_dim: int, hidden: int = FLOW_HIDDEN):
        super().__init__()
        self.theta_dim = theta_dim
        self.ctx_dim   = ctx_dim
        self.net = CNF_MLP(theta_dim + ctx_dim + 1, theta_dim, hidden=hidden, depth=2)

    def forward(self, t: torch.Tensor, states):
        y, logp, h = states

        # Need gradients for Hutchinson trace even if outer is no_grad
        with torch.enable_grad():
            y = y.requires_grad_(True)

            B = y.size(0)
            t_feat = torch.ones(B, 1, device=y.device, dtype=y.dtype) * t
            inp = torch.cat([y, h, t_feat], dim=1)        # (B, D+ctx+1)
            f   = self.net(inp)                           # dy/dt

            # Hutchinson trace estimator: tr(df/dy) ≈ eps^T (df/dy) eps
            eps    = torch.randn_like(y)
            scalar = (f * eps).sum()
            g = torch.autograd.grad(scalar, y, create_graph=True, retain_graph=True)[0]
            trace = (g * eps).sum(dim=1)                  # (B,)

        dlogp = -trace
        dh    = torch.zeros_like(h)
        return f, dlogp, dh


class ConditionalCNF(nn.Module):
    """
    Conditional CNF:
      p(theta|h) via ODE flow between base N(0,I) and data space.
    """
    def __init__(
        self,
        theta_dim: int,
        ctx_dim: int = CTX_DIM,
        hidden: int = FLOW_HIDDEN,
        atol: float = CNF_ATOL,
        rtol: float = CNF_RTOL,
        method: str = CNF_METHOD,
        step_size: float = CNF_STEP_SIZE,
    ):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.ctx_dim   = int(ctx_dim)
        self.func      = ODEFunc(self.theta_dim, self.ctx_dim, hidden=hidden)
        self.atol      = float(atol)
        self.rtol      = float(rtol)
        self.method    = str(method)
        self.step_size = float(step_size)

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
        Diagnostic mapping theta -> z0 in base space.
        """
        B = theta.size(0)
        t_span = torch.tensor([1.0, 0.0], device=theta.device, dtype=theta.dtype)
        logp0  = torch.zeros(B, device=theta.device, dtype=theta.dtype)
        y0     = (theta, logp0, h)
        yt, logpt, ht = self._odeint(y0, t_span)
        z0 = yt[-1]
        return z0

    def log_prob(self, theta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, D), h: (B, ctx) -> log p(theta|h): (B,)
        """
        B = theta.size(0)
        t_span = torch.tensor([1.0, 0.0], device=theta.device, dtype=theta.dtype)
        logp0  = torch.zeros(B, device=theta.device, dtype=theta.dtype)
        y0     = (theta, logp0, h)

        yt, logpt, ht = self._odeint(y0, t_span)
        z0         = yt[-1]
        delta_logp = logpt[-1]
        return self._base_logprob(z0) - delta_logp

    @torch.no_grad()
    def sample(self, n: int, h: torch.Tensor) -> torch.Tensor:
        """
        Sample theta ~ p(theta|h) by integrating base -> data (t: 0->1).
        """
        device = h.device
        dtype  = h.dtype

        if h.size(0) == 1:
            H = h.repeat(n, 1)
        elif h.size(0) != n:
            reps = (n + h.size(0) - 1) // h.size(0)
            H = h.repeat(reps, 1)[:n]
        else:
            H = h

        z0    = torch.randn(n, self.theta_dim, device=device, dtype=dtype)
        logp0 = self._base_logprob(z0)
        t_span = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        yT, logpT, hT = self._odeint((z0, logp0, H), t_span)
        theta = yT[-1]
        return theta


# ===================== PosteriorNet =====================

class PosteriorNet(nn.Module):
    """
    Full posterior net:
      h = EAConditioner(x, phase)          # (B, CTX_DIM)
      log p(theta|x) = CNF.log_prob(theta, h)
    """
    def __init__(
        self,
        seq_len: int,
        *,
        use_phase: bool = True,
        phase_provider=None,
        x_mean=None,
        x_std=None,
        ctx_dim: int = CTX_DIM,
        cnf_hidden: int = FLOW_HIDDEN,
        atol: float = CNF_ATOL,
        rtol: float = CNF_RTOL,
        method: str = CNF_METHOD,
        step_size: float = CNF_STEP_SIZE,
    ):
        super().__init__()
        self.use_phase = bool(use_phase)
        self.phase_provider = phase_provider

        # Keep EA as conditioner (your DNF version already uses EAUnifiedPE)
        self.cond = EAConditioner(
            seq_len=seq_len,
            ctx_dim=ctx_dim,
            use_posenc=True,
            use_phasepe=self.use_phase,   # if False, disable phase path
            use_snrenc=False,
            cnn_stem=False,
            weighting="learned",
            alpha_pos=1.0,
            alpha_phase=1.0,
            alpha_snr=1.0,
            phase_provider=phase_provider,  # EA may predict internally when phase=None
            x_mean=x_mean,
            x_std=x_std,
        )

        self.flow = ConditionalCNF(
            theta_dim=THETA_DIM,
            ctx_dim=ctx_dim,
            hidden=cnf_hidden,
            atol=atol,
            rtol=rtol,
            method=method,
            step_size=step_size,
        )

    def _maybe_get_phase(self, x: torch.Tensor, phase: torch.Tensor | None):
        """
        Logic:
        - If use_phase=False -> None
        - If phase provided by loader -> use it
        - Else if phase_provider exists -> predict from x (optional; EA may also do it internally)
        - Else -> None
        """
        if not self.use_phase:
            return None
        if phase is not None:
            return phase
        if self.phase_provider is not None:
            # If your EAUnifiedPE already calls phase_provider internally, you can skip this.
            # Keeping it here makes PosteriorNet robust even if EA does not.
            with torch.no_grad():
                return self.phase_provider(x)
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
