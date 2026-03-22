#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Hyperparams used inside the model ----------------
# NOTE: THETA_DIM will be overwritten from the main script based on y.shape[1]
THETA_DIM   = 8          # placeholder, main script sets this to y.shape[1]
CTX_DIM     = 256
D_MODEL     = 128
NHEAD       = 4
DEPTH       = 4
FLOW_LAYERS = 8
FLOW_HIDDEN = 256

# ---- New: weighting control for PE and phase-PE ----
WEIGHTING_MODE   = "learned"   # "learned" or "manual"
ALPHA_POS_INIT   = 1.0         # initial / fixed weight for positional encoding
ALPHA_PHASE_INIT = 1.0         # initial / fixed weight for phase encoding


# ---- Sinusoidal PE and PhaseSinusoidalPE (EA-style) ----
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, d)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens only used for length
        return self.pe[:, : x_tokens.size(1)]


class PhaseSinusoidalPE(nn.Module):
    """EA-style phase encoding: uses φ as the 'position' to build sin/cos features."""
    def __init__(self, d_model: int):
        super().__init__()
        div = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        self.register_buffer("div_term", torch.exp(div), persistent=False)  # (d/2,)
        self.d_model = d_model

    def forward(self, phase_seq: torch.Tensor) -> torch.Tensor:
        # phase_seq: (B, L') wrapped (-pi, pi]
        B, S = phase_seq.shape
        angles = phase_seq.unsqueeze(-1) * self.div_term.view(1, 1, -1)  # (B, S, d/2)
        pe = torch.zeros(B, S, self.d_model, device=phase_seq.device, dtype=phase_seq.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe


# ---- Conv stem (no patching), avg pool to reduce length ----
class ConvPoolStem(nn.Module):
    def __init__(self, in_ch=1, d_model=128, k1=7, k2=5, pool=8):
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model // 2, k1, padding=k1 // 2),
            nn.GELU(),
            # second conv
            nn.Conv1d(d_model // 2, d_model, k2, padding=k2 // 2),
            nn.GELU(),
            nn.AvgPool1d(pool)  # e.g., 400 -> 50 if pool=8
        )

    def forward(self, x):        # x: (B, L)
        x = x.unsqueeze(1)       # (B, 1, L)
        z = self.net(x)          # (B, d_model, L')
        return z.transpose(1, 2) # (B, L', d_model)


# ---- Transformer encoder with time PE + additive phase PE, with weights ----
class SeqTransformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        depth=4,
        mlp_ratio=2.0,
        dropout=0.1,
        use_posenc=True,
        use_phasepe=True,
        weighting_mode: str = "learned",   # "learned" or "manual"
        alpha_pos: float = 1.0,
        alpha_phase: float = 1.0,
    ):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.use_posenc  = use_posenc
        self.use_phasepe = use_phasepe
        self.weighting_mode = weighting_mode

        self.posenc   = SinusoidalPE(d_model)      if use_posenc  else None
        self.phase_pe = PhaseSinusoidalPE(d_model) if use_phasepe else None

        # ---- weights for positional & phase encodings ----
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
        tokens:    (B, L', d_model) from ConvPoolStem
        phase_seq: (B, L') pooled/wrapped phase aligned to L' or None
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
        h = torch.cat([cls, h], dim=1)
        z = self.enc(h)
        cls_tok  = z[:, 0]
        mean_tok = z[:, 1:].mean(dim=1)
        return torch.cat([cls_tok, mean_tok], dim=1)  # (B, 2*d_model)


# ---- Conditioner that consumes (x, phase) and outputs context ----
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
        self.pool = pool
        self.stem = ConvPoolStem(in_ch=1, d_model=d_model, pool=pool)
        self.backbone = SeqTransformer(
            d_model=d_model, nhead=nhead, depth=depth, mlp_ratio=2.0, dropout=0.1,
            use_posenc=use_posenc, use_phasepe=use_phasepe,
            weighting_mode=weighting_mode,
            alpha_pos=alpha_pos,
            alpha_phase=alpha_phase,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, phase: torch.Tensor | None = None):  # x: (B, L)
        tokens = self.stem(x)                       # (B, L', d_model)
        phase_pooled = None
        if phase is not None:
            # Always match token length via adaptive pooling
            Lp = tokens.size(1)
            phase_pooled = F.adaptive_avg_pool1d(phase.unsqueeze(1), output_size=Lp).squeeze(1)  # (B, L')
        h = self.backbone(tokens, phase_pooled)     # (B, 2*d_model)
        return self.proj(h)                         # (B, CTX_DIM)


# ---- Conditional RealNVP generalized to D = THETA_DIM ----
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


# ---- Full posterior net (condition on x & phase, with optional PhaseProvider) ----
class PosteriorNet(nn.Module):
    def __init__(
        self,
        use_phase: bool = True,
        phase_provider=None,
    ):
        super().__init__()
        self.use_phase = use_phase
        self.phase_provider = phase_provider
        self.cond = Conditioner(
            d_model=D_MODEL, nhead=NHEAD, depth=DEPTH, out_dim=CTX_DIM,
            use_posenc=True, use_phasepe=True, pool=8,
            weighting_mode=WEIGHTING_MODE,
            alpha_pos=ALPHA_POS_INIT,
            alpha_phase=ALPHA_PHASE_INIT,
        )
        self.flow = ConditionalRealNVP(D=THETA_DIM, ctx_dim=CTX_DIM,
                                       n_layers=FLOW_LAYERS, hidden=FLOW_HIDDEN)

    def _maybe_get_phase(self, x: torch.Tensor, phase: torch.Tensor | None):
        """
        Logic:
        - If self.use_phase is False -> return None (ignore phase entirely)
        - If phase is already provided -> use it (true phase from loader)
        - Else if phase_provider exists -> predict phase from x
        - Else -> None
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

    def log_prob(self, theta, x, phase=None):
        phase_eff = self._maybe_get_phase(x, phase)
        h = self.cond(x, phase_eff)
        return self.flow.log_prob(theta, h)

    @torch.no_grad()
    def sample(self, n, x, phase=None):
        phase_eff = self._maybe_get_phase(x, phase)
        h = self.cond(x, phase_eff)
        return self.flow.sample(n, h)

