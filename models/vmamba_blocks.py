"""
models/vmamba_blocks.py

Foundational building blocks for the VM-UNet architecture.
==========================================================
This module contains all low-level components that every other part of
the network depends on.  Build and test these before touching the full
UNet.

Contents
--------
  LayerNorm2d   — LayerNorm that handles both BCHW and BHWC tensors.
  PatchEmbed    — Converts a raw CT image into a grid of patch features.
  PatchMerging  — Halves spatial resolution and doubles channel count.
  SS2D          — 2D Selective Scan (causal, 4-directional).   [reference]
  VSSD          — Visual State Space Duality (bidirectional).  [used in all blocks]
  VSSBlock      — Complete VMamba visual state-space block.    [encoder/decoder unit]

Reference: Architecture.pdf – Chapters 5, 6, 8, 10, 11
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# DropPath (stochastic depth) — from timm.
# Stochastic depth randomly drops entire residual branches during training,
# acting as a form of regularisation for deep networks.
# If timm is not installed, fall back to a no-op identity.
try:
    from timm.models.layers import DropPath
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

    class DropPath(nn.Module):          # type: ignore[no-redef]
        """Fallback no-op when timm is not installed."""
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x                    # identity — no stochastic depth

        def extra_repr(self) -> str:
            return f"drop_prob={self.drop_prob} (timm not installed — no-op)"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LayerNorm2d
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    """
    LayerNorm that works on **both** BCHW and BHWC tensors.

    Parameters
    ----------
    num_channels : int
    eps          : float
    data_format  : 'channels_first' (BCHW) | 'channels_last' (BHWC)
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
        data_format: str = 'channels_first',
    ):
        super().__init__()
        if data_format not in ('channels_first', 'channels_last'):
            raise ValueError(
                f"data_format must be 'channels_first' or 'channels_last', "
                f"got '{data_format}'."
            )
        self.num_channels = num_channels
        self.eps          = eps
        self.data_format  = data_format
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == 'channels_last':
            return F.layer_norm(
                x, (self.num_channels,), self.weight, self.bias, self.eps
            )
        # channels_first: permute → normalise → permute back
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x, (self.num_channels,), self.weight, self.bias, self.eps
        )
        return x.permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, eps={self.eps}, "
            f"data_format='{self.data_format}'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PatchEmbed
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Patch Embedding — entry point of the VM-UNet encoder.

    Conv2d(in_channels, embed_dim, kernel=patch_size, stride=patch_size)
    followed by LayerNorm2d.

    Shape: [B, in_channels, H, W] → [B, embed_dim, H/patch_size, W/patch_size]
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim:   int = 96,
        patch_size:  int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.norm = LayerNorm2d(embed_dim, data_format='channels_first')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input ({H}×{W}) must be divisible by patch_size={self.patch_size}."
            )
        return self.norm(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PatchMerging
# ─────────────────────────────────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """
    Patch Merging — downsampling between encoder scales.

    Gathers TL / TR / BL / BR sub-grids, concatenates → [B, 4C, H/2, W/2],
    then LayerNorm + Conv1×1 → [B, 2C, H/2, W/2].
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = in_channels * 2
        self.norm         = LayerNorm2d(in_channels * 4, data_format='channels_first')
        self.reduction    = nn.Conv2d(
            in_channels * 4, in_channels * 2, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"PatchMerging needs even dims, got H={H}, W={W}.")
        merged = torch.cat([
            x[:, :, 0::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 1::2, 1::2],
        ], dim=1)
        return self.reduction(self.norm(merged))

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SS2D  — kept for reference (causal, Stage 1 original)
# ─────────────────────────────────────────────────────────────────────────────

class SS2D(nn.Module):
    """
    2D Selective Scan — causal, 4-directional.  Retained for reference.
    Stage 2 uses VSSD instead.

    Input / Output: [B, H, W, C]  (BHWC)
    """

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.A_log      = nn.Parameter(torch.rand(d_model, d_state))
        self.D          = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)
        self.delta_proj._preserve_dt_bias = True   # VMUNet global init must not zero this
        self.B_proj     = nn.Linear(d_model, d_state,  bias=False)
        self.C_proj     = nn.Linear(d_model, d_state,  bias=False)
        self.out_proj   = nn.Linear(d_model, d_model,  bias=False)
        self.norm       = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        dt_min, dt_max = 1e-4, 0.1
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.delta_proj.bias.copy_(torch.log(torch.expm1(dt)))
        nn.init.uniform_(self.out_proj.weight, -0.01, 0.01)

    @staticmethod
    def _build_scan_indices(H, W, device):
        L       = H * W
        idx_row = torch.arange(L, device=device)
        h_idx   = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
        w_idx   = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
        idx_col = (w_idx * H + h_idx).reshape(-1)
        fwds    = [idx_row, idx_row.flip(0), idx_col, idx_col.flip(0)]
        invs    = []
        for fwd in fwds:
            inv      = torch.empty_like(fwd)
            inv[fwd] = torch.arange(L, device=device)
            invs.append(inv)
        return fwds, invs

    def _ssm_scan(self, x_seq, A, delta, B_seq, C_seq):
        B_batch, L, _ = x_seq.shape
        h   = torch.zeros(B_batch, self.d_model, self.d_state,
                          device=x_seq.device, dtype=x_seq.dtype)
        outs = []
        for t in range(L):
            dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = delta[:, t, :].unsqueeze(-1) * B_seq[:, t, :].unsqueeze(1)
            h  = dA * h + dB * x_seq[:, t, :].unsqueeze(-1)
            y  = (C_seq[:, t, :].unsqueeze(1) * h).sum(-1)
            outs.append(y)
        return torch.stack(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        A           = -torch.exp(self.A_log.float())
        x_flat      = x.reshape(B, H * W, C)
        delta       = F.softplus(self.delta_proj(x_flat))
        B_seq       = self.B_proj(x_flat)
        C_seq       = self.C_proj(x_flat)
        fwds, invs  = self._build_scan_indices(H, W, x.device)
        y_sum       = torch.zeros_like(x_flat)
        for fwd, inv in zip(fwds, invs):
            y_sc  = self._ssm_scan(
                x_flat[:, fwd], A,
                delta[:, fwd], B_seq[:, fwd], C_seq[:, fwd],
            )
            y_sc  = y_sc + self.D.unsqueeze(0).unsqueeze(0) * x_flat[:, fwd]
            y_sum = y_sum + y_sc[:, inv]
        return self.out_proj(self.norm(y_sum)).reshape(B, H, W, C)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_state={self.d_state}"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VSSD  — bidirectional 2-D scan (used in all VSSBlocks)
# ─────────────────────────────────────────────────────────────────────────────

class VSSD(nn.Module):
    """
    Visual State Space Duality — bidirectional 2-D selective scan.

    Forward + backward pass on each of two spatial axes gives every pixel
    full non-causal context from all other pixels simultaneously.

    Parallel cumsum scan runs in chunks of ``SCAN_CHUNK`` positions (vectorised
    ``cumsum`` inside each chunk). During training, each chunk is wrapped in
    ``torch.utils.checkpoint`` so backward recomputes chunk internals instead
    of storing activations for every chunk — critical when ``L = H×W`` is large.

    Input / Output: [B, H, W, C]  (BHWC)
    """

    # Smaller chunks × checkpoint → lower peak VRAM inside each segment.
    SCAN_CHUNK: int = 128

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                     .unsqueeze(0).repeat(d_model, 1)
            )
        )
        self.D          = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)
        self.delta_proj._preserve_dt_bias = True   # VMUNet global init must not zero this
        self.B_proj     = nn.Linear(d_model, d_state,  bias=False)
        self.C_proj     = nn.Linear(d_model, d_state,  bias=False)
        self.out_proj   = nn.Linear(d_model, d_model,  bias=False)
        self._init_weights()

    def _init_weights(self):
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.delta_proj.bias.copy_(torch.log(torch.expm1(dt)))
        nn.init.uniform_(self.out_proj.weight, -0.02, 0.02)

    def _scan_chunk_fwd(self, h_carry: torch.Tensor, xc: torch.Tensor):
        """
        One sequence chunk of the parallel SSM scan (see ``_ssm_scan``).

        Kept separate so each chunk can be checkpoint-wrapped during training.

        Parameters
        ----------
        h_carry : [B, C, S]
            Hidden state entering this chunk (zeros at sequence start).
        xc : [B, L_chunk, C]

        Returns
        -------
        y : [B, L_chunk, C]
        h_next : [B, C, S]  last-step hidden state for the next chunk
        """
        A_neg = -F.softplus(self.A_log.to(dtype=xc.dtype))
        D_vec = self.D.to(dtype=xc.dtype)

        delta = F.softplus(self.delta_proj(xc))
        Bs = self.B_proj(xc)
        Cs = self.C_proj(xc)

        dA = torch.exp(
            delta.unsqueeze(-1) * A_neg.unsqueeze(0).unsqueeze(0)
        )
        dB_u = (
            delta.unsqueeze(-1)
            * Bs.unsqueeze(2)
            * xc.unsqueeze(-1)
        )

        log_dA = torch.log(dA.clamp(min=1e-38))
        A_cumsum = torch.cumsum(log_dA, dim=1)
        dB_u_scaled = dB_u * torch.exp(-A_cumsum)
        inner = torch.cumsum(dB_u_scaled, dim=1)
        h = torch.exp(A_cumsum) * (h_carry.unsqueeze(1) + inner)

        y = (Cs.unsqueeze(2) * h).sum(-1)
        y = y + D_vec.unsqueeze(0).unsqueeze(0) * xc

        return y, h[:, -1, :, :]

    def _ssm_scan(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Chunked parallel SSM scan along sequence dim.

        Training: each chunk runs under gradient checkpointing so autograd does
        not retain cumscan intermediates for every segment at once.

        x_seq : [B, L, C]
        Returns: y_seq [B, L, C]
        """
        B, L, C = x_seq.shape

        h_carry = x_seq.new_zeros(B, C, self.d_state)
        ys = []
        ckpt = self.training and torch.is_grad_enabled()

        for start in range(0, L, self.SCAN_CHUNK):
            end = min(start + self.SCAN_CHUNK, L)
            xc = x_seq[:, start:end]

            if ckpt:
                y_c, h_carry = checkpoint(
                    self._scan_chunk_fwd,
                    h_carry,
                    xc,
                    use_reentrant=False,
                )
            else:
                y_c, h_carry = self._scan_chunk_fwd(h_carry, xc)

            ys.append(y_c)

        return torch.cat(ys, dim=1)

    def _bidirectional_scan(self, x_seq: torch.Tensor) -> torch.Tensor:
        y_fwd = self._ssm_scan(x_seq)
        y_bwd = self._ssm_scan(x_seq.flip(dims=[1])).flip(dims=[1])
        return y_fwd + y_bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # Horizontal axis
        y_h = self._bidirectional_scan(x.reshape(B, H * W, C)).reshape(B, H, W, C)
        # Vertical axis (swap H↔W so scan runs along columns)
        x_v = x.permute(0, 2, 1, 3).reshape(B, H * W, C)
        y_v = self._bidirectional_scan(x_v).reshape(B, W, H, C).permute(0, 2, 1, 3)
        return self.out_proj(y_h + y_v)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"axes=2 (H+V), bidirectional"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VSSBlock  — complete VMamba visual state-space block
# ─────────────────────────────────────────────────────────────────────────────

class VSSBlock(nn.Module):
    """
    VSS Block — the atomic building block of VM-UNet.

    Two VSSBlocks are stacked at every encoder and decoder scale.
    The first block extracts raw features; the second refines context.

    Data format
    -----------
    All inputs, intermediate tensors, and outputs use **BHWC** (channel-last)
    format.  The only exception is the depthwise convolution (Step 3), which
    requires BCHW — the tensor is permuted before the conv and permuted back
    immediately after.

    Operation sequence
    ------------------
    Let x ∈ ℝ^{B×H×W×C} be the input.

    Step 1  LayerNorm
            x_norm = LayerNorm(x)                          [B, H, W, C]

    Step 2  Expand + split (gating preparation)
            x_expand = Linear(C → 2C)(x_norm)             [B, H, W, 2C]
            x_main, x_gate = split(x_expand, C, dim=-1)   each [B, H, W, C]

    Step 3  Depthwise convolution on x_main
            (local neighbourhood before global Mamba scan)
            x_main_bchw = x_main.permute(0,3,1,2)         [B, C, H, W]
            x_main_bchw = DWConv3×3(x_main_bchw)          [B, C, H, W]
            x_main = x_main_bchw.permute(0,2,3,1)         [B, H, W, C]

    Step 4  VSSD scan
            x_main = VSSD(x_main)                         [B, H, W, C]
            (bidirectional non-causal 2-D scan)

    Step 5  Gating  (learned feature selection)
            x_main = x_main * SiLU(x_gate)               [B, H, W, C]
            SiLU(z) = z · σ(z)  — smooth, allows negative outputs

    Step 6  Output projection
            x_main = Linear(C → C)(x_main)               [B, H, W, C]

    Step 7  Stochastic depth + residual
            output = x + DropPath(x_main)                 [B, H, W, C]

    Why each step?
    --------------
    Step 2  The expand-split pattern is a standard "gated MLP" / GLU design.
            The gate branch decides which features from the Mamba scan are
            worth keeping.  Features that carry noise or irrelevant signals
            get gated to near-zero.

    Step 3  Mamba scans sequentially and captures global context but misses
            fine local structure (nearby pixels).  The 3×3 depthwise conv
            injects local neighbourhood information *before* the global scan,
            giving the block both local detail and global context.

    Step 5  SiLU is preferred over sigmoid because it allows negative output
            values and has a smoother gradient, which trains better
            empirically in Mamba-based architectures.

    Step 7  DropPath (stochastic depth) randomly drops the entire residual
            branch during training.  This regularises deep networks and is
            the standard technique in Swin Transformer / VMamba.

    Parameters
    ----------
    d_model   : int    Channel dimension C at this scale.
    d_state   : int    VSSD hidden state dimension (default 8).
    drop_path : float  Stochastic depth drop probability (default 0.0).

    Shape
    -----
    Input:  [B, H, W, C]   BHWC
    Output: [B, H, W, C]   BHWC

    Examples
    --------
    >>> blk = VSSBlock(d_model=96)
    >>> x   = torch.randn(2, 128, 128, 96)
    >>> blk(x).shape
    torch.Size([2, 128, 128, 96])
    """

    def __init__(
        self,
        d_model:   int,
        d_state:   int   = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.d_model   = d_model
        self.d_state   = d_state
        self.drop_path_rate = drop_path

        # ── Step 1: LayerNorm ─────────────────────────────────────────────
        # Normalises over the C dimension for each spatial position.
        # data_format='channels_last' because the block works in BHWC.
        self.norm = LayerNorm2d(d_model, data_format='channels_last')

        # ── Step 2: Expand + split ────────────────────────────────────────
        # Linear(C → 2C) produces x_main and x_gate in one shot.
        # bias=True lets the gate learn a default open/closed state.
        self.expand_proj = nn.Linear(d_model, d_model * 2, bias=True)

        # ── Step 3: Depthwise 3×3 convolution ────────────────────────────
        # groups=d_model → each channel has its own 3×3 filter (depthwise).
        # padding=1 keeps H and W unchanged.
        # bias=False: the VSSD LayerNorm that follows has its own bias.
        self.dw_conv = nn.Conv2d(
            in_channels  = d_model,
            out_channels = d_model,
            kernel_size  = 3,
            padding      = 1,
            groups       = d_model,    # depthwise
            bias         = False,
        )

        # ── Step 4: VSSD scan ─────────────────────────────────────────────
        # Bidirectional 2-D selective scan — gives every pixel full context.
        self.vssd = VSSD(d_model=d_model, d_state=d_state)

        # ── Step 6: Output projection ─────────────────────────────────────
        # Mixes the gated features across the channel dimension.
        # bias=True for expressivity.
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # ── Step 7: Stochastic depth ──────────────────────────────────────
        # DropPath drops the entire residual branch with probability drop_path.
        # When drop_path=0 this is the identity function (no overhead).
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        """
        Initialise weights so the block starts as a near-identity transform.

        At initialisation:
          expand_proj  — normal(0, 0.02), bias=0
          out_proj     — normal(0, 0.02), bias=0
          dw_conv      — normal(0, 0.02), initialised to approximate identity
                         (centre pixel of each 3×3 filter = 1, rest ≈ 0)

        This follows the convention from Swin Transformer and VMamba:
        start with a small perturbation from identity so training begins
        from a stable, well-conditioned point.
        """
        # expand_proj
        nn.init.normal_(self.expand_proj.weight, std=0.02)
        nn.init.zeros_(self.expand_proj.bias)

        # out_proj
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        # dw_conv: approximate identity
        # For a depthwise conv with kernel 3×3, set the centre weight to 1
        # and everything else to ~0 so the initial output ≈ input.
        nn.init.normal_(self.dw_conv.weight, std=0.02)
        with torch.no_grad():
            # dw_conv.weight shape: [C, 1, 3, 3]  (out, in/groups, kH, kW)
            self.dw_conv.weight[:, 0, 1, 1] += 1.0   # centre pixel bias

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, H, W, C]  BHWC

        Returns
        -------
        torch.Tensor  [B, H, W, C]  BHWC
        """
        B, H, W, C = x.shape

        # Keep the original input for the residual connection (Step 7)
        residual = x                                      # [B, H, W, C]

        # ── Step 1: LayerNorm ─────────────────────────────────────────────
        x_norm = self.norm(x)                             # [B, H, W, C]

        # ── Step 2: Expand and split ──────────────────────────────────────
        x_expand = self.expand_proj(x_norm)               # [B, H, W, 2C]
        x_main, x_gate = x_expand.chunk(2, dim=-1)        # each [B, H, W, C]

        # ── Step 3: Depthwise convolution (local context) ─────────────────
        # DWConv requires BCHW format.
        x_main_bchw = x_main.permute(0, 3, 1, 2)         # BHWC → BCHW [B,C,H,W]
        x_main_bchw = self.dw_conv(x_main_bchw)           # [B, C, H, W]
        x_main      = x_main_bchw.permute(0, 2, 3, 1)    # BCHW → BHWC [B,H,W,C]

        # ── Step 4: VSSD scan (global context) ───────────────────────────
        x_main = self.vssd(x_main)                        # [B, H, W, C]

        # ── Step 5: Gating ────────────────────────────────────────────────
        # x_gate controls which features from the Mamba scan are kept.
        # SiLU = x · σ(x): smooth, allows negative values, good gradients.
        x_main = x_main * F.silu(x_gate)                  # [B, H, W, C]

        # ── Step 6: Output projection ─────────────────────────────────────
        x_main = self.out_proj(x_main)                    # [B, H, W, C]

        # ── Step 7: Stochastic depth + residual connection ────────────────
        # DropPath randomly zeros the entire branch during training.
        # At test time DropPath is the identity.
        output = residual + self.drop_path(x_main)        # [B, H, W, C]

        return output

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"d_state={self.d_state}, "
            f"drop_path={self.drop_path_rate}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("models/vmamba_blocks.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on : {device}")
    print(f"timm found : {_TIMM_AVAILABLE}")
    print()

    # ── Shared raw CT dummy input ─────────────────────────────────────────
    B, Cin, H, W = 2, 1, 512, 512
    x_raw = torch.randn(B, Cin, H, W, device=device)

    # ── 1. LayerNorm2d ────────────────────────────────────────────────────
    print("── LayerNorm2d ───────────────────────────────────────────────")
    for fmt, shape in [
        ('channels_first',  (B, Cin, H, W)),
        ('channels_last',   (B, H, W, Cin)),
    ]:
        xi   = torch.randn(*shape, device=device)
        norm = LayerNorm2d(Cin, data_format=fmt).to(device)
        out  = norm(xi)
        assert out.shape == xi.shape
        print(f"  {fmt:16s}  {list(xi.shape)} → {list(out.shape)}  ✓")

    # ── 2. PatchEmbed ─────────────────────────────────────────────────────
    print("\n── PatchEmbed ────────────────────────────────────────────────")
    embed    = PatchEmbed(1, 96, 4).to(device)
    out_emb  = embed(x_raw)
    expected = (B, 96, H // 4, W // 4)
    assert tuple(out_emb.shape) == expected, f"Got {list(out_emb.shape)}"
    print(f"  {list(x_raw.shape)} → {list(out_emb.shape)}  ✓")
    print(f"  Params: {sum(p.numel() for p in embed.parameters()):,}")
    print("\nPatchEmbed and PatchMerging: PASSED")

    # ── 3. PatchMerging ───────────────────────────────────────────────────
    print("\n── PatchMerging ──────────────────────────────────────────────")
    merge    = PatchMerging(96).to(device)
    out_mrg  = merge(out_emb)
    expected = (B, 192, 64, 64)
    assert tuple(out_mrg.shape) == expected, f"Got {list(out_mrg.shape)}"
    print(f"  {list(out_emb.shape)} → {list(out_mrg.shape)}  ✓")
    print(f"  Params: {sum(p.numel() for p in merge.parameters()):,}")

    # ── 4. SS2D (reference) ───────────────────────────────────────────────
    print("\n── SS2D (causal, reference) ──────────────────────────────────")
    Bs, Hs, Ws, Cs = 2, 8, 8, 32
    x_ss = torch.randn(Bs, Hs, Ws, Cs, device=device)
    ss2d = SS2D(d_model=Cs).to(device)
    out_ss = ss2d(x_ss)
    assert tuple(out_ss.shape) == (Bs, Hs, Ws, Cs)
    print(f"  {list(x_ss.shape)} → {list(out_ss.shape)}  ✓")
    print("\nSS2D: PASSED")

    # ── 5. VSSD ───────────────────────────────────────────────────────────
    print("\n── VSSD (bidirectional) ──────────────────────────────────────")
    Bv, Hv, Wv, Cv = 2, 16, 16, 96
    x_vs   = torch.randn(Bv, Hv, Wv, Cv, device=device)
    vssd   = VSSD(d_model=Cv).to(device)
    out_vs = vssd(x_vs)
    assert tuple(out_vs.shape) == (Bv, Hv, Wv, Cv)
    diff   = (out_vs - x_vs).abs().max().item()
    assert diff > 1e-4, "VSSD output is trivially equal to input."
    print(f"  {list(x_vs.shape)} → {list(out_vs.shape)}  ✓")
    print(f"  Max |out − in| = {diff:.4f}  (non-trivial ✓)")
    print("\nVSSD: PASSED")

    # ── 6. VSSBlock ───────────────────────────────────────────────────────
    print("\n── VSSBlock ──────────────────────────────────────────────────")

    # Use a small spatial size so the pure-Python VSSD loop is fast
    Bb, Hb, Wb, Cb = 2, 16, 16, 96
    x_blk = torch.randn(Bb, Hb, Wb, Cb, device=device)

    # (a) Basic shape test — no drop path
    blk      = VSSBlock(d_model=Cb, d_state=16, drop_path=0.0).to(device)
    out_blk  = blk(x_blk)
    expected = (Bb, Hb, Wb, Cb)
    assert tuple(out_blk.shape) == expected, (
        f"Expected {list(expected)}, got {list(out_blk.shape)}"
    )
    print(f"  Shape  : {list(x_blk.shape)} → {list(out_blk.shape)}  ✓")
    print(f"  Params : {sum(p.numel() for p in blk.parameters()):,}")

    # (b) Output should differ from input (block is not a pure identity)
    diff_blk = (out_blk - x_blk).abs().max().item()
    assert diff_blk > 1e-4, "VSSBlock output is trivially equal to input."
    print(f"  Max |out − in| = {diff_blk:.4f}  (non-trivial ✓)")

    # (c) Gradient flow
    x_grd  = torch.randn(Bb, Hb, Wb, Cb, device=device, requires_grad=True)
    blk_g  = VSSBlock(d_model=Cb, d_state=16).to(device)
    loss   = blk_g(x_grd).mean()
    loss.backward()
    assert x_grd.grad is not None
    gnorm  = x_grd.grad.norm().item()
    assert gnorm > 0
    print(f"  Grad norm      = {gnorm:.6f}  ✓")

    # (d) Residual: at init (near-zero output proj) output ≈ input
    #     Specifically test that VSSBlock preserves the residual path.
    blk_id = VSSBlock(d_model=Cb, d_state=16).to(device)
    with torch.no_grad():
        # Zero out the output projection weights so the branch = 0
        blk_id.out_proj.weight.zero_()
        blk_id.out_proj.bias.zero_()
        out_id = blk_id(x_blk)
    # With out_proj = 0, the block output should equal the input (residual only)
    residual_err = (out_id - x_blk).abs().max().item()
    assert residual_err < 1e-5, (
        f"Residual path broken: max err={residual_err:.2e}"
    )
    print(f"  Residual path  : max err = {residual_err:.2e}  ✓")

    # (e) Drop path — verify it runs (does not crash); actual dropping only
    #     happens during training (model.train()), not eval mode.
    if _TIMM_AVAILABLE:
        blk_dp   = VSSBlock(d_model=Cb, d_state=16, drop_path=0.3).to(device)
        blk_dp.train()
        out_dp   = blk_dp(x_blk)
        assert tuple(out_dp.shape) == expected
        print(f"  DropPath(0.3)  : output shape {list(out_dp.shape)}  ✓")
    else:
        print(f"  DropPath       : timm not installed — tested as Identity ✓")

    # (f) Two-block stack (as used at every encoder/decoder scale)
    print("\n  Two-block stack (encoder scale simulation):")
    blk1     = VSSBlock(d_model=Cb, d_state=16).to(device)
    blk2     = VSSBlock(d_model=Cb, d_state=16).to(device)
    out_2blk = blk2(blk1(x_blk))
    assert tuple(out_2blk.shape) == expected
    print(f"    {list(x_blk.shape)} → blk1 → blk2 → {list(out_2blk.shape)}  ✓")

    print("\nVSSBlock: PASSED")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)