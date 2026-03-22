#!/usr/bin/env python
# coding: utf-8

"""
model_dnfs.py  (UPDATED)

Change:
- Replace old ConvPoolStem+SeqTransformer Conditioner with EAUnifiedPE conditioner.
- EAUnifiedPE output is used as context "h" for ConditionalRealNVP.

Works for:
- x shaped (B,L) OR (B,P,L) (EAUnifiedPE flattens internally)  :contentReference[oaicite:2]{index=2}
- phase can be provided (true phase) OR predicted internally via PhaseProvider.
"""

import math
import torch
import torch.nn as nn

# ====== GLOBALS (kept compatible with your existing run_dnfs.py) ======
THETA_DIM   = 4        # overwritten by run script: mdl.THETA_DIM = len(target_names)
CTX_DIM     = 64
FLOW_LAYERS = 10
FLOW_HIDDEN = 128

# ---- EA conditioner defaults ----
EA_PATCH   = 20
EA_HEADS   = 8
EA_DMODEL  = 128
EA_DEPTH   = 4
EA_DIM_FF  = 256
EA_MEM     = 40

# Import EA model / PhaseProvider
# (Use your ea_model_hy_new.py instead of the old conditioner)
from ea_model_hy_new import EAUnifiedPE, PhaseProvider


# ===================== Conditional RealNVP (same as before) =====================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=FLOW_HIDDEN, depth=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, D, keep_idx, ctx_dim=CTX_DIM, hidden=FLOW_HIDDEN):
        super().__init__()
        self.D = D
        self.keep_idx = keep_idx
        self.trans_idx = [i for i in range(D) if i not in keep_idx]
        in_dim = len(self.keep_idx) + ctx_dim
        self.s_net = MLP(in_dim, len(self.trans_idx), hidden=hidden, depth=2)
        self.t_net = MLP(in_dim, len(self.trans_idx), hidden=hidden, depth=2)

    def forward(self, theta, h, reverse=False):
        x_pass = torch.cat([theta[:, self.keep_idx], h], dim=1)
        s = self.s_net(x_pass)
        t = self.t_net(x_pass)
        out = theta.clone()
        if not reverse:
            out[:, self.trans_idx] = theta[:, self.trans_idx] * torch.exp(s) + t
            logdet = s.sum(dim=1)
        else:
            out[:, self.trans_idx] = (theta[:, self.trans_idx] - t) * torch.exp(-s)
            logdet = (-s).sum(dim=1)
        return out, logdet


def default_masks_D(D, n_layers):
    pairs = []
    base = list(range(D))
    for i in range(D * 3):
        a = base[i % D]
        b = base[(i + 1) % D]
        if a != b:
            pairs.append(sorted(list({a, b})))
    uniq, seen = [], set()
    for p in pairs:
        t = tuple(p)
        if t not in seen:
            uniq.append(p)
            seen.add(t)
        if len(uniq) >= n_layers:
            break
    return uniq[:n_layers]


class ConditionalRealNVP(nn.Module):
    def __init__(self, D=THETA_DIM, ctx_dim=CTX_DIM, n_layers=FLOW_LAYERS, hidden=FLOW_HIDDEN):
        super().__init__()
        masks = default_masks_D(D, n_layers)
        self.D = D
        self.layers = nn.ModuleList(
            [AffineCoupling(D, keep_idx=m, ctx_dim=ctx_dim, hidden=hidden) for m in masks]
        )

    def fwd_to_z(self, theta, h):
        logdet = torch.zeros(theta.size(0), device=theta.device)
        x = theta
        for g in self.layers:
            x, ld = g(x, h, reverse=False)
            logdet = logdet + ld
        return x, logdet

    def inv_from_z(self, z, h):
        logdet = torch.zeros(z.size(0), device=z.device)
        x = z
        for g in reversed(self.layers):
            x, ld = g(x, h, reverse=True)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, theta, h):
        z, logdet = self.fwd_to_z(theta, h)
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.D * math.log(2 * math.pi)
        return log_pz + logdet

    @torch.no_grad()
    def sample(self, n, h):
        if h.size(0) == 1:
            H = h.repeat(n, 1)
        elif h.size(0) != n:
            reps = (n + h.size(0) - 1) // h.size(0)
            H = h.repeat(reps, 1)[:n]
        else:
            H = h
        z = torch.randn(n, self.D, device=H.device)
        theta, _ = self.inv_from_z(z, H)
        return theta


# ===================== NEW: EAUnifiedPE Conditioner =====================
class EAConditioner(nn.Module):
    """
    Conditioner: (x, phase) -> h (CTX_DIM)

    - If phase_provider is provided, EAUnifiedPE can predict phase internally when phase=None.
    - Supports x: (B,L) or (B,P,L).  :contentReference[oaicite:3]{index=3}
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
        # IMPORTANT: out_dim = CTX_DIM (context)
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


# ===================== PosteriorNet =====================
class PosteriorNet(nn.Module):
    """
    Full posterior net:
      h = EAConditioner(x, phase)  # (B, CTX_DIM)
      log p(theta|x) = flow.log_prob(theta, h)
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
    ):
        super().__init__()
        self.use_phase = bool(use_phase)

        self.cond = EAConditioner(
            seq_len=seq_len,
            ctx_dim=ctx_dim,
            use_posenc=True,
            use_phasepe=use_phase,   # if False, EA won't use phase path
            use_snrenc=False,
            cnn_stem=False,
            weighting="learned",
            alpha_pos=1.0,
            alpha_phase=1.0,
            alpha_snr=1.0,
            phase_provider=phase_provider,
            x_mean=x_mean,
            x_std=x_std,
        )

        self.flow = ConditionalRealNVP(
            D=THETA_DIM,
            ctx_dim=ctx_dim,
            n_layers=FLOW_LAYERS,
            hidden=FLOW_HIDDEN,
        )

    def log_prob(self, theta, x, phase=None):
        phase_eff = phase if self.use_phase else None
        h = self.cond(x, phase_eff)
        return self.flow.log_prob(theta, h)

    @torch.no_grad()
    def sample(self, n, x, phase=None):
        phase_eff = phase if self.use_phase else None
        h = self.cond(x, phase_eff)
        return self.flow.sample(n, h)
