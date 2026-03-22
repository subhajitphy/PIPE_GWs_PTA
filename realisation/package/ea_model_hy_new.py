# ea_model_hybrid.py  (UPDATED)
# - EAUnifiedPE now accepts x shaped (B,L) or (B,P,L) and flattens to (B,P*L).
# - PhaseProvider now accepts x shaped (B,L) or (B,P,L) or (B,P*L) (auto-detect P).
# - If PhaseProvider returns snr per-pulsar (B,P), EAUnifiedPE will average to (B,).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from phase_pred import predict_phase, load_phase_model, PhaseProvider

# --------------------------- Sinusoidal Positional Encodings -----------------
class SinusoidalPE(nn.Module):
    """Classic index-based sinusoidal PE table; you add it to tokens."""
    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (S,1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (S,d)


class PhaseSinusoidalPE(nn.Module):
    """
    Phase-based sinusoidal positional encoding.

    IMPORTANT:
    - If input phase is unwrapped, wrap it into [-π, π] so sin/cos stay stable.

    Input:
        phase_tok: (B, S)
    Output:
        (B, S, d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        div = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        self.register_buffer("div_term", torch.exp(div))
        self.d_model = d_model

    def forward(self, phase_tok: torch.Tensor) -> torch.Tensor:
        B, S = phase_tok.shape
        phase_wrapped = (phase_tok + math.pi) % (2 * math.pi) - math.pi
        angles = phase_wrapped.unsqueeze(-1) * self.div_term.view(1, 1, -1)
        pe = torch.zeros(B, S, self.d_model, device=phase_tok.device, dtype=phase_tok.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe


# # --------------------------- Phase Provider ----------------------------------
# class PhaseProvider(nn.Module):
#     """
#     Wrap a pretrained phase+SNR model so EAUnifiedPE can fetch φ and SNR internally.

#     Accepts standardized inputs shaped:
#       - (B, L)
#       - (B, P, L)
#       - (B, P*L)    (auto-detect P from base_len)

#     Returns:
#       - phi: same shape as input (B,L) or (B,P,L) or (B,P*L)
#     Stores:
#       - self.last_snr_pred:
#           (B,) for (B,L)
#           (B,P) for (B,P,L)
#           (B,P) for (B,P*L) when P>1
#     """
#     def __init__(
#         self,
#         phase_model,
#         mu_x: torch.Tensor,
#         std_x: torch.Tensor,
#         predict_fn,
#         *,
#         device: torch.device | str = "cpu",
#         unwrap: bool = True,
#         x_mean: torch.Tensor | None = None,
#         x_std:  torch.Tensor | None = None,
#     ):
#         super().__init__()
#         self.model = phase_model
#         self.mu_x = mu_x
#         self.std_x = std_x
#         self.predict_fn = predict_fn
#         self.device = torch.device(device)
#         self.unwrap = bool(unwrap)

#         # dataset-level normalisation (used by main pipeline)
#         self.register_buffer("x_mean", torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach())
#         self.register_buffer("x_std",  torch.as_tensor(1.0) if x_std  is None else x_std.clone().detach())

#         self.last_snr_pred: torch.Tensor | None = None

#         # infer the "base" time length the phase model expects
#         self.base_len = self._infer_base_len()

#     def _infer_base_len(self) -> int:
#         # Try to infer expected per-pulsar length robustly.
#         for t in (self.mu_x, self.std_x):
#             if isinstance(t, torch.Tensor):
#                 if t.ndim >= 1:
#                     return int(t.shape[-1])
#         return 0

#     def _split_flat_to_bpL(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, int]:
#         """
#         x_flat: (B, Ltot)
#         Returns x_bpL: (B, P, Lbase), P
#         """
#         B, Ltot = x_flat.shape
#         Lbase = int(self.base_len) if self.base_len and self.base_len > 0 else None
#         if Lbase is None or Lbase <= 0:
#             raise ValueError("PhaseProvider: could not infer base_len from mu_x/std_x.")
#         if Ltot % Lbase != 0:
#             raise ValueError(f"PhaseProvider: Ltot={Ltot} not divisible by base_len={Lbase}.")
#         P = Ltot // Lbase
#         x_bpL = x_flat.view(B, P, Lbase)
#         return x_bpL, P

#     @torch.no_grad()
#     def forward(self, x_std_in: torch.Tensor) -> torch.Tensor:
#         # Normalize shapes:
#         #  - if (B,P,L) -> predict per pulsar
#         #  - if (B,P*L) -> auto split into (B,P,Lbase) and predict per pulsar
#         #  - if (B,L) -> predict directly

#         orig_shape = x_std_in.shape

#         if x_std_in.ndim == 3:
#             B, P, L = x_std_in.shape
#             x_bpL = x_std_in
#             mode = "bpL"
#         elif x_std_in.ndim == 2:
#             B, L = x_std_in.shape
#             # If looks like concatenated pulsars:
#             if self.base_len and self.base_len > 0 and L != self.base_len and (L % self.base_len == 0):
#                 x_bpL, P = self._split_flat_to_bpL(x_std_in)
#                 mode = "flat_as_bpL"
#             else:
#                 x_bpL = None
#                 mode = "bL"
#         else:
#             raise ValueError(f"PhaseProvider: expected 2D or 3D input, got {orig_shape}.")

#         self.model.eval()

#         if mode == "bL":
#             # un-standardize
#             x_raw = x_std_in * self.x_std + self.x_mean  # (B,L)
#             phi, snr_pred = self.predict_fn(
#                 self.model, self.mu_x, self.std_x, x_raw,
#                 device=self.device, unwrap=self.unwrap, batch_dim=True
#             )
#             if not isinstance(snr_pred, torch.Tensor):
#                 snr_pred = torch.as_tensor(snr_pred)
#             self.last_snr_pred = snr_pred.to(self.device, dtype=torch.float32)  # (B,)

#             if not isinstance(phi, torch.Tensor):
#                 phi = torch.as_tensor(phi)
#             return phi.to(x_std_in.device, dtype=x_std_in.dtype)  # (B,L)

#         # Per-pulsar prediction path:
#         # x_bpL: (B,P,Lbase) -> (B*P, Lbase)
#         B, P, Lbase = x_bpL.shape
#         x_2d = x_bpL.reshape(B * P, Lbase)

#         # IMPORTANT: x_mean/x_std might be (1,Lbase) OR (1,P*Lbase).
#         # For phase model, we need to un-standardize per pulsar; so we slice if needed.
#         x_mean = self.x_mean
#         x_std  = self.x_std
#         if x_mean.ndim == 2 and x_mean.shape[1] == P * Lbase:
#             x_mean = x_mean.view(1, P, Lbase)[:, 0, :]  # (1,Lbase)
#         if x_std.ndim == 2 and x_std.shape[1] == P * Lbase:
#             x_std = x_std.view(1, P, Lbase)[:, 0, :]

#         x_raw = x_2d * x_std + x_mean  # (B*P, Lbase)

#         phi, snr_pred = self.predict_fn(
#             self.model, self.mu_x, self.std_x, x_raw,
#             device=self.device, unwrap=self.unwrap, batch_dim=True
#         )

#         if not isinstance(snr_pred, torch.Tensor):
#             snr_pred = torch.as_tensor(snr_pred)
#         snr_pred = snr_pred.to(self.device, dtype=torch.float32).view(B, P)
#         self.last_snr_pred = snr_pred  # (B,P)

#         if not isinstance(phi, torch.Tensor):
#             phi = torch.as_tensor(phi)
#         phi = phi.to(x_std_in.device, dtype=x_std_in.dtype).view(B, P, Lbase)

#         if mode == "bpL":
#             return phi  # (B,P,Lbase)
#         else:
#             # return flattened to match original input (B, P*Lbase)
#             return phi.reshape(B, P * Lbase)



            
# ------------------------------ Patch Embedding ------------------------------
class PatchEmbed1D(nn.Module):
    """(B,L) -> tokens (B,S,d_model) with Linear(patch->d_model)."""
    def __init__(self, patch: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(-1)  # (B,L,1)->(B,L)
        B, L = x.shape
        P = self.patch
        if L % P != 0:
            x = F.pad(x, (0, P - (L % P)))
            L = x.shape[1]
        tok = x.view(B, L // P, P)   # (B,S,P)
        tok = self.proj(tok)         # (B,S,d)
        return self.drop(F.gelu(tok)), (B, L, P)


# ------------------------------ External Attention ---------------------------
class EAEncoderLayer(nn.Module):
    """External-Attention encoder (per-head memory) + FFN. I/O: (B,S,d)"""
    def __init__(self, d_model: int, dim_ff: int, heads: int = 4,
                 mem_size: int | None = 64, p_drop: float = 0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.h = heads
        self.dh = d_model // heads
        self.M = mem_size or d_model

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.Mk = nn.Parameter(torch.randn(self.h, self.dh, self.M) / math.sqrt(self.dh))
        self.Mv = nn.Parameter(torch.randn(self.h, self.M, self.dh) / math.sqrt(self.dh))

        self.drop_attn = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(dim_ff, d_model)
        )
        self.drop_ff = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h, dh = self.h, self.dh

        q = self.q(x); k = self.k(x); v = self.v(x)

        def split(t): return t.view(B, S, h, dh).transpose(1, 2).contiguous()
        qh, kh, vh = split(q), split(k), split(v)

        q_mem = torch.einsum("bhsd,hdm->bhsm", qh, self.Mk)
        k_mem = torch.einsum("bhsd,hdm->bhsm", kh, self.Mk)

        attn = torch.matmul(q_mem, k_mem.transpose(-1, -2)) / math.sqrt(dh)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        ctx = torch.matmul(attn, vh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)

        x = self.norm1(x + self.o(ctx))
        x = self.norm2(x + self.drop_ff(self.ff(x)))
        return x


# ------------------------------ CNN Stem -------------------------------------
class CNNStem1D(nn.Module):
    """
    Two-layer 1D CNN stem producing tokens (B,S,d_model) from (B,L).
    """
    def __init__(self, d_model: int, *, k1: int = 7, k2: int = 5,
                 pool: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=k1, stride=2, padding=k1//2)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(pool)
        self.drop1 = nn.Dropout(p_drop)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=k2, stride=2, padding=k2//2)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(pool)
        self.drop2 = nn.Dropout(p_drop)

        self.proj  = nn.Conv1d(128, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        x = x.unsqueeze(1)  # (B,1,L)
        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.proj(x)      # (B,d_model,S)
        x = x.transpose(1, 2) # (B,S,d_model)
        return x


# --------------------------- EA with Unified PEs -----------------------------
class EAUnifiedPE(nn.Module):
    """
    EA encoder with:
      - optional sinusoidal index PE
      - optional phase PE (either provided or via PhaseProvider)
      - optional SNR encoding (provided or predicted by PhaseProvider)

    UPDATED: x can be (B,L) OR (B,P,L). If (B,P,L), it flattens to (B,P*L).
    Phase can be (B,L) or (B,P,L); it is flattened to match x if needed.
    """
    def __init__(self, seq_len: int, out_dim: int, *,
                 patch: int = 16, d_model: int = 128, depth: int = 4, dim_ff: int = 256,
                 heads: int = 4, mem_size: int | None = 64, p_drop: float = 0.1,
                 use_posenc: bool = True, use_phasepe: bool = True,
                 weighting: str = "learned",
                 alpha_pos: float = 1.0, alpha_phase: float = 1.0,
                 phase_pool: str = "mean",
                 cnn_stem: bool = True, stem_k1: int = 7, stem_k2: int = 5, stem_pool: int = 2,
                 phase_provider: PhaseProvider | None = None,
                 x_mean: torch.Tensor | None = None,
                 x_std:  torch.Tensor | None = None,
                 use_snrenc: bool = False,
                 alpha_snr: float = 1.0):
        super().__init__()
        assert weighting in {"manual", "learned", "hybrid"}
        self.patch = patch
        self.phase_pool = phase_pool
        self.use_pos, self.use_phase = use_posenc, use_phasepe
        self.weighting = weighting
        self.alpha_pos, self.alpha_phase = float(alpha_pos), float(alpha_phase)
        self.cnn_stem = cnn_stem
        self.use_snr = bool(use_snrenc)

        self.phase_provider = phase_provider
        self.register_buffer("x_mean_buf", torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach())
        self.register_buffer("x_std_buf",  torch.as_tensor(1.0) if x_std  is None else x_std.clone().detach())

        # token embedding
        if cnn_stem:
            self.stem = CNNStem1D(d_model, k1=stem_k1, k2=stem_k2, pool=stem_pool, p_drop=p_drop)
            self.embed = None
        else:
            self.embed = PatchEmbed1D(patch, d_model, dropout=p_drop)
            self.stem  = None

        # PEs
        MAX_TOKENS = 20000
        
        if self.use_pos:
            self.pos_pe = SinusoidalPE(d_model, max_len=MAX_TOKENS)
        if self.use_phase:
            self.phase_pe = PhaseSinusoidalPE(d_model)

        # weights for pos & phase
        if self.use_pos:
            self.w_pos = nn.Parameter(torch.tensor(self.alpha_pos)) if weighting in {"learned", "hybrid"} else torch.tensor(self.alpha_pos)
        else:
            self.w_pos = None
        if self.use_phase:
            self.w_phase = nn.Parameter(torch.tensor(self.alpha_phase)) if weighting in {"learned", "hybrid"} else torch.tensor(self.alpha_phase)
        else:
            self.w_phase = None

        # SNR encoding
        if self.use_snr:
            self.snr_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )
            self.w_snr = nn.Parameter(torch.tensor(float(alpha_snr))) if weighting in {"learned", "hybrid"} else torch.tensor(float(alpha_snr))
        else:
            self.snr_mlp = None
            self.w_snr   = None

        # EA encoder + head
        self.blocks = nn.ModuleList([
            EAEncoderLayer(d_model, dim_ff, heads=heads, mem_size=mem_size, p_drop=p_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d_model, out_dim)
        )

    def _pool_phase_to_tokens_by_adaptive(self, phase: torch.Tensor, S: int) -> torch.Tensor:
        if phase.ndim == 3:
            phase = phase.squeeze(-1)
        return F.adaptive_avg_pool1d(phase.unsqueeze(1), output_size=S).squeeze(1)

    def _pool_phase_by_patch(self, phase: torch.Tensor, P: int) -> torch.Tensor:
        if phase.ndim == 3:
            phase = phase.squeeze(-1)
        if phase.size(1) % P != 0:
            phase = F.pad(phase, (0, P - (phase.size(1) % P)))
        chunks = phase.view(phase.size(0), -1, P)  # (B,S,P)
        return chunks.mean(-1) if self.phase_pool == "mean" else chunks[:, :, P//2]

    def forward(self, x: torch.Tensor,
                phase: torch.Tensor | None = None,
                snr: torch.Tensor | None = None) -> torch.Tensor:

        # Accept x: (B,L) or (B,P,L)

        x_orig = x  # keep original shape
        
        # ---- compute phase/snr BEFORE flattening ----
        need_phase = self.use_phase and (phase is None)
        need_snr   = self.use_snr   and (snr   is None)
        
        if (need_phase or need_snr) and (self.phase_provider is not None):
            # make sure provider has correct dataset stats
            self.phase_provider.x_mean = self.x_mean_buf
            self.phase_provider.x_std  = self.x_std_buf
            
            # IMPORTANT: call on original x (B,P,L) if available
            phase_from_provider = self.phase_provider(x_orig)
            if need_phase:
                phase = phase_from_provider
            if need_snr:
                snr = self.phase_provider.last_snr_pred
                if isinstance(snr, torch.Tensor) and snr.ndim == 2:
                    snr = snr.mean(dim=1)
                
        if x.ndim == 3:
            B, Pp, Lp = x.shape
        
            # FIX: if phase is (B,Lp), expand to (B,Pp,Lp) so it matches x after flatten
            if phase is not None and phase.ndim == 2 and phase.size(1) == Lp:
                phase = phase[:, None, :].repeat(1, Pp, 1)  # (B,P,L)
        
            x = x.reshape(B, Pp * Lp)
        
            # keep phase aligned with flattened x
            if phase is not None and phase.ndim == 3:
                phase = phase.reshape(B, Pp * Lp)
        else:
            B = x.size(0)


        # maybe compute φ / SNR via PhaseProvider
        need_phase = self.use_phase and (phase is None)
        need_snr   = self.use_snr   and (snr   is None)

        if (need_phase or need_snr) and (self.phase_provider is not None):
            # sync stats
            if hasattr(self.phase_provider, "x_mean"):
                self.phase_provider.x_mean = self.x_mean_buf
            if hasattr(self.phase_provider, "x_std"):
                self.phase_provider.x_std  = self.x_std_buf

            phase_from_provider = self.phase_provider(x)  # (B,L) or (B,P,L) or (B,P*L)
            if need_phase:
                phase = phase_from_provider

            if need_snr:
                snr = self.phase_provider.last_snr_pred
                # if per-pulsar, average to per-sample
                if isinstance(snr, torch.Tensor) and snr.ndim == 2:
                    snr = snr.mean(dim=1)

        # tokens
        if self.cnn_stem:
            z = self.stem(x)        # (B,S,d)
            S = z.size(1); Ppatch = None
        else:
            z, (_B, _L, Ppatch) = self.embed(x)
            S = z.size(1)

        # index pos encoding
        if self.use_pos:
            pos_term = self.pos_pe.pe[:S].unsqueeze(0).to(z.dtype)
            z = z + (self.w_pos * pos_term if isinstance(self.w_pos, nn.Parameter) else float(self.w_pos) * pos_term)

        # phase encoding
        if self.use_phase:
            if phase is None:
                raise ValueError("use_phasepe=True but no phase provided (and no PhaseProvider)!")
            if phase.ndim == 3:
                phase = phase.reshape(phase.size(0), -1)

            if self.cnn_stem:
                phase_tok = self._pool_phase_to_tokens_by_adaptive(phase, S)  # (B,S)
            else:
                phase_tok = self._pool_phase_by_patch(phase, Ppatch)          # (B,S)

            phi_term = self.phase_pe(phase_tok)  # (B,S,d)
            z = z + (self.w_phase * phi_term if isinstance(self.w_phase, nn.Parameter) else float(self.w_phase) * phi_term)

        # snr encoding
        if self.use_snr:
            if snr is None:
                raise ValueError("use_snrenc=True but no snr provided (and no PhaseProvider)!")
            snr_val = snr.to(z.device, z.dtype).view(B, 1)
            snr_log10 = torch.log10(torch.clamp(snr_val, min=1e-3))
            snr_emb = self.snr_mlp(snr_log10)                  # (B,d_model)
            snr_tok = snr_emb.unsqueeze(1).expand(-1, S, -1)    # (B,S,d_model)
            z = z + (self.w_snr * snr_tok if isinstance(self.w_snr, nn.Parameter) else float(self.w_snr) * snr_tok)

        # EA encoder + head
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).mean(1)
        return self.head(z)

