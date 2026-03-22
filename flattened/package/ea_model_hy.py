# ea_model_hybrid.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    IMPORTANT UPDATE:
    - If input phase is unwrapped (e.g., monotonic from 0→200 rad),
      we FIRST wrap it into [-π, π] so that sin/cos remain periodic and stable.

    Input:
        phase_tok: (B, S) raw or unwrapped phase
    Output:
        (B, S, d_model) sinusoidal embedding using wrapped phase
    """
    def __init__(self, d_model: int):
        super().__init__()
        div = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        self.register_buffer("div_term", torch.exp(div))
        self.d_model = d_model

    def forward(self, phase_tok: torch.Tensor) -> torch.Tensor:
        B, S = phase_tok.shape

        # ---- NEW: wrap unwrapped/continuous phase into [-π, π] ----
        phase_wrapped = (phase_tok + math.pi) % (2 * math.pi) - math.pi

        # ---- sinusoidal encoding ----
        angles = phase_wrapped.unsqueeze(-1) * self.div_term.view(1, 1, -1)
        pe = torch.zeros(B, S, self.d_model,
                         device=phase_tok.device,
                         dtype=phase_tok.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)

        return pe

class PhaseProvider(nn.Module):
    """
    Wrap a pretrained phase+SNR model for EAUnifiedPE.

    The wrapped predict_fn must return:
        phi, snr_pred

    Inputs:
        - x_std_in: standardized (B, L) input from CNF/EA loader
        - x_mean/x_std: dataset-level normalization used by the main model
        - mu_x/std_x:  training normalization used by the phase model

    Outputs:
        - phi_pred: (B, L) predicted phase (float32 tensor on correct device)
        - Also stores predicted SNR in self.last_snr_pred for downstream use.
    """
    def __init__(
        self,
        phase_model,
        mu_x: torch.Tensor,
        std_x: torch.Tensor,
        predict_fn,
        *,
        device: torch.device | str = "cpu",
        unwrap: bool = True,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ):
        super().__init__()

        self.model = phase_model
        self.mu_x = mu_x
        self.std_x = std_x
        self.predict_fn = predict_fn
        self.device = torch.device(device)
        self.unwrap = bool(unwrap)

        # dataset-level normalisation (used by CNF/EA pipeline)
        self.register_buffer(
            "x_mean",
            torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach()
        )
        self.register_buffer(
            "x_std",
            torch.as_tensor(1.0) if x_std is None else x_std.clone().detach()
        )

        # where predicted SNR will be stored
        self.last_snr_pred = None

    @torch.no_grad()
    def forward(self, x_std_in: torch.Tensor) -> torch.Tensor:
        """
        x_std_in : (B, L) standardized inputs from EA/CNF loader

        Steps:
            1. Un-standardize using dataset-level x_mean/x_std
            2. Call predict_fn(model, ...) which returns phi, snr_pred
            3. Store snr_pred in self.last_snr_pred
            4. Return only phi (B, L)
        """
        # 1) Un-standardize
        x_raw = x_std_in * self.x_std + self.x_mean      # (B, L)

        # 2) Phase + SNR prediction
        self.model.eval()
        phi, snr_pred = self.predict_fn(
            self.model,
            self.mu_x,
            self.std_x,
            x_raw,
            device=self.device,
            unwrap=self.unwrap,
            batch_dim=True
        )

        # 3) Store SNR prediction
        if not isinstance(snr_pred, torch.Tensor):
            snr_pred = torch.as_tensor(snr_pred)
        self.last_snr_pred = snr_pred.to(self.device, dtype=torch.float32)

        # 4) Return phase as tensor
        if not isinstance(phi, torch.Tensor):
            phi = torch.as_tensor(phi)

        return phi.to(x_std_in.device, dtype=x_std_in.dtype)  # (B,L)


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
            x = x.squeeze(-1)     # (B,L,1) -> (B,L)
        B, L = x.shape
        P = self.patch
        if L % P != 0:
            x = F.pad(x, (0, P - (L % P)))
            L = x.shape[1]
        tok = x.view(B, L // P, P)            # (B,S,P)
        tok = self.proj(tok)                  # (B,S,d)
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
        h, dh, M = self.h, self.dh, self.M

        q = self.q(x); k = self.k(x); v = self.v(x)
        def split(t): return t.view(B, S, h, dh).transpose(1, 2).contiguous()
        qh, kh, vh = split(q), split(k), split(v)

        q_mem = torch.einsum("bhsd,hdm->bhsm", qh, self.Mk)  # (B,h,S,M)
        k_mem = torch.einsum("bhsd,hdm->bhsm", kh, self.Mk)  # (B,h,S,M)

        attn = torch.matmul(q_mem, k_mem.transpose(-1, -2)) / math.sqrt(dh)  # (B,h,S,S)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        ctx = torch.matmul(attn, vh)                          # (B,h,S,dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)  # (B,S,D)
        x = self.norm1(x + self.o(ctx))
        x = self.norm2(x + self.drop_ff(self.ff(x)))
        return x

# ------------------------------ CNN Stem -------------------------------------
class CNNStem1D(nn.Module):
    """
    Two-layer 1D CNN stem:
      Conv(1->64) + BN + GELU + MaxPool
      Conv(64->128) + BN + GELU + MaxPool
      1x1 Conv to project -> d_model
    Outputs tokens as (B, S, d_model).
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
        x = x.unsqueeze(1)              # (B,1,L)
        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.proj(x)                # (B,d_model,S)
        x = x.transpose(1, 2)           # (B,S,d_model)
        return x

# --------------------------- Phase Provider ----------------------------------
class PhaseProvider(nn.Module):
    """
    Wrap a pretrained phase+SNR model so EAUnifiedPE can fetch φ and SNR internally.
    Expects standardized X as input; we un-standardize using x_mean/x_std
    before calling predict_fn.

    NEW predict_fn signature must be:
      phi_pred, snr_pred = predict_fn(
          phase_model, mu_x, std_x, x_raw,
          device=..., unwrap=True, batch_dim=True
      )

    - phi_pred: (B, L)
    - snr_pred: (B,) physical SNR
    """
    def __init__(
        self,
        phase_model,
        mu_x: torch.Tensor,
        std_x: torch.Tensor,
        predict_fn,
        *,
        device: torch.device | str = "cpu",
        unwrap: bool = True,
        x_mean: torch.Tensor | None = None,
        x_std:  torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = phase_model
        self.mu_x = mu_x
        self.std_x = std_x
        self.predict_fn = predict_fn
        self.device = torch.device(device)
        self.unwrap = bool(unwrap)

        self.register_buffer(
            "x_mean",
            torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach()
        )
        self.register_buffer(
            "x_std",
            torch.as_tensor(1.0) if x_std is None else x_std.clone().detach()
        )

        # latest predicted SNR (B,)
        self.last_snr_pred: torch.Tensor | None = None

    @torch.no_grad()
    def forward(self, x_std_in: torch.Tensor) -> torch.Tensor:
        # x_std_in: (B,L) standardized (your loader input)
        x_raw = x_std_in * self.x_std + self.x_mean

        self.model.eval()
        phi, snr_pred = self.predict_fn(
            self.model,
            self.mu_x,
            self.std_x,
            x_raw,
            device=self.device,
            unwrap=self.unwrap,
            batch_dim=True,
        )

        # store predicted SNR
        if not isinstance(snr_pred, torch.Tensor):
            snr_pred = torch.as_tensor(snr_pred)
        self.last_snr_pred = snr_pred.to(self.device, dtype=torch.float32)  # (B,)

        # ensure float32 tensor on same device for φ
        if not isinstance(phi, torch.Tensor):
            phi = torch.as_tensor(phi)
        return phi.to(x_std_in.device, dtype=x_std_in.dtype)  # (B,L)

# --------------------------- EA with Unified PEs -----------------------------
class EAUnifiedPE(nn.Module):
    """
    EA-based encoder with unified positional + phase + (optional) SNR encoding.

    Added args:
      cnn_stem: bool                 -> use 1D conv stem instead of Linear patching
      phase_provider: PhaseProvider  -> auto-compute φ and SNR if not supplied
      x_mean/x_std:                  -> de-standardize X before φ/SNR prediction
      use_snrenc: bool               -> enable SNR-based encoding
      alpha_snr: float               -> initial weight for SNR encoding
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

        # optional internal phase provider + X stats for de-standardization
        self.phase_provider = phase_provider
        self.register_buffer(
            "x_mean_buf",
            torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach()
        )
        self.register_buffer(
            "x_std_buf",
            torch.as_tensor(1.0) if x_std is None else x_std.clone().detach()
        )

        # token embedding
        if cnn_stem:
            self.stem = CNNStem1D(d_model, k1=stem_k1, k2=stem_k2, pool=stem_pool, p_drop=p_drop)
            self.embed = None
        else:
            self.embed = PatchEmbed1D(patch, d_model, dropout=p_drop)
            self.stem  = None

        # PEs
        max_tokens = (seq_len + patch - 1) // patch + 32
        if self.use_pos:
            self.pos_pe = SinusoidalPE(d_model, max_len=max_tokens)
        if self.use_phase:
            self.phase_pe = PhaseSinusoidalPE(d_model)

        # weights for pos & phase
        if self.use_pos:
            self.w_pos = nn.Parameter(torch.tensor(self.alpha_pos)) \
                         if weighting in {"learned", "hybrid"} else \
                         torch.tensor(self.alpha_pos)
        else:
            self.w_pos = None
        if self.use_phase:
            self.w_phase = nn.Parameter(torch.tensor(self.alpha_phase)) \
                           if weighting in {"learned", "hybrid"} else \
                           torch.tensor(self.alpha_phase)
        else:
            self.w_phase = None

        # SNR encoding (scalar per sample -> token-wise vector)
        if self.use_snr:
            self.snr_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )
            self.w_snr = nn.Parameter(torch.tensor(float(alpha_snr))) \
                         if weighting in {"learned", "hybrid"} else \
                         torch.tensor(float(alpha_snr))
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

    # ----- helpers to pool phase to token length --------------------------------
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

    # ----- forward ----------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                phase: torch.Tensor | None = None,
                snr: torch.Tensor | None = None) -> torch.Tensor:
        """
        x     : (B,L) or (B,L,1) standardized input
        phase : optional (B,L) ground-truth φ; if None and use_phaseenc, uses PhaseProvider
        snr   : optional (B,) SNR; if None and use_snrenc, uses PhaseProvider.predicted SNR
        """

        B = x.size(0)

        # ---------------------- maybe compute φ and/or SNR via PhaseProvider ------
        need_phase_from_provider = self.use_phase and (phase is None)
        need_snr_from_provider   = self.use_snr   and (snr is None)

        if (need_phase_from_provider or need_snr_from_provider) and (self.phase_provider is not None):
            # sync stats
            if hasattr(self.phase_provider, "x_mean"):
                self.phase_provider.x_mean = self.x_mean_buf
            if hasattr(self.phase_provider, "x_std"):
                self.phase_provider.x_std  = self.x_std_buf

            # Call once; fills phase and self.phase_provider.last_snr_pred
            phase_from_provider = self.phase_provider(x)  # (B,L)
            if need_phase_from_provider:
                phase = phase_from_provider
            if need_snr_from_provider:
                snr = self.phase_provider.last_snr_pred

        # ---------------------- tokens -------------------------------------------
        if self.cnn_stem:
            z = self.stem(x)                # (B,S,d)
            S = z.size(1); P = None
        else:
            z, (_B, _L, P) = self.embed(x)  # (B,S,d)
            S = z.size(1)

        # ---------------------- index (positional) encoding ----------------------
        if self.use_pos:
            pos_term = self.pos_pe.pe[:S].unsqueeze(0).to(z.dtype)  # (1,S,d)
            if isinstance(self.w_pos, nn.Parameter):
                z = z + self.w_pos * pos_term
            else:
                z = z + float(self.w_pos) * pos_term

        # ---------------------- phase-based encoding -----------------------------
        if self.use_phase:
            assert phase is not None, "use_phasepe=True but no phase provided (and no PhaseProvider)!"
            if self.cnn_stem:
                phase_tok = self._pool_phase_to_tokens_by_adaptive(phase, S)  # (B,S)
            else:
                phase_tok = self._pool_phase_by_patch(phase, P)               # (B,S)
            phi_term = self.phase_pe(phase_tok)                               # (B,S,d)
            if isinstance(self.w_phase, nn.Parameter):
                z = z + self.w_phase * phi_term
            else:
                z = z + float(self.w_phase) * phi_term

        # ---------------------- SNR-based encoding -------------------------------
        if self.use_snr:
            assert snr is not None, "use_snrenc=True but no SNR provided (and no PhaseProvider)!"
            # snr: (B,) physical SNR -> log10 -> [B,1] -> MLP -> [B,d_model] -> [B,S,d_model]
            snr_val = snr.to(z.device, z.dtype).view(B, 1)  # (B,1)
            snr_log10 = torch.log10(torch.clamp(snr_val, min=1e-3))
            snr_emb = self.snr_mlp(snr_log10)               # (B,d_model)
            snr_tok = snr_emb.unsqueeze(1).expand(-1, S, -1)  # (B,S,d_model)
            if isinstance(self.w_snr, nn.Parameter):
                z = z + self.w_snr * snr_tok
            else:
                z = z + float(self.w_snr) * snr_tok

        # ---------------------- EA encoder + head --------------------------------
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).mean(1)
        return self.head(z)
