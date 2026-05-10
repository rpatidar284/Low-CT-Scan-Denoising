"""
models/anatomy_mamba.py

Anatomy-Conditioned Mamba Block for Stage 2 denoiser.
======================================================
Each block receives four inputs and applies conditioning in this order:

  x [B, C, H, W]     feature map (BCHW in, BHWC internally)
  S [B, 7, H, W]     organ probability map at matching spatial scale
  e_a [B, 7, C_anat] per-organ feature embeddings
  t_emb [B, 256]     diffusion timestep embedding

Block steps
-----------
1. adaLN-Zero       — timestep-conditioned LayerNorm (γ, β from t_emb)
2. Spatial FiLM     — organ-specific per-pixel modulation from S
3. VSSD Scan        — bidirectional non-causal 2D scan (global context)
4. Cross-Attention  — pixels query e_a to retrieve organ appearance

Reference: Architecture.pdf — AnatomyMamba_block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.vmamba_blocks import VSSD, LayerNorm2d
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.vmamba_blocks import VSSD, LayerNorm2d


# ═════════════════════════════════════════════════════════════════════════════
# Spatial FiLM — organ-conditioned per-pixel feature modulation
# ═════════════════════════════════════════════════════════════════════════════

class SpatialFiLM(nn.Module):
    """
    Converts organ probability map S into per-pixel γ (scale) and β (shift).

    A liver pixel gets "liver-mode" normalization; a lung pixel gets
    "lung-mode" normalization.  The small ConvNet reads the 7-class S map
    and outputs 2*C parameters for every spatial position.

    Architecture
    ------------
    S [B, 7, H, W]
      → Conv2d(7→32, k=3, p=1) → SiLU
      → Conv2d(32→32, k=3, p=1) → SiLU
      → Conv2d(32→2C, k=1)      → split → γ [B,C,H,W], β [B,C,H,W]

    Output is applied as:  output = (1 + γ) * LayerNorm(x) + β

    Parameters
    ----------
    num_classes : int   Number of organ classes (7).
    channels : int      Feature channels at this UNet scale.
    """

    def __init__(self, num_classes: int = 7, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, channels * 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, H, W]  feature map (after adaLN)
        S : [B, 7, H, W]  organ probability map

        Returns
        -------
        x : [B, C, H, W]  spatially modulated features
        """
        params = self.net(S)                        # [B, 2C, H, W]
        gamma, beta = params.chunk(2, dim=1)        # each [B, C, H, W]
        return x * (1.0 + gamma) + beta


# ═════════════════════════════════════════════════════════════════════════════
# adaLN-Zero — adaptive layer norm (zero-initialized)
# ═════════════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    """
    Timestep-conditioned LayerNorm.

    t_emb → Linear → SiLU → Linear → 6 * channels

    Splits into 6 modulation vectors:
        shift_scan, scale_scan, gate_scan  — for VSSD branch
        shift_attn, scale_attn, gate_attn  — for cross-attention branch

    Zero-initialized output weights → block starts as near-identity.
    """

    def __init__(self, channels: int, time_emb_dim: int = 256):
        super().__init__()
        self.channels = channels
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * channels, bias=True),
        )
        # Zero-initialize
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, t_emb: torch.Tensor):
        """
        Returns (shift_scan, scale_scan, gate_scan, shift_attn, scale_attn, gate_attn),
        each [B, C] for broadcasting over spatial dims.
        """
        out = self.modulation(t_emb)               # [B, 6C]
        return out.chunk(6, dim=1)                  # 6 × [B, C]


# ═════════════════════════════════════════════════════════════════════════════
# Cross-Attention with Anatomy Embeddings
# ═════════════════════════════════════════════════════════════════════════════

class AnatomyCrossAttention(nn.Module):
    """
    Pixels query the 7 organ embeddings (e_a) to retrieve patient-specific
    organ appearance information.

    Q = x (flattened pixels)       [B, H*W, C]
    K = V = e_a                    [B, 7, C_anat]
    output = x + MultiheadAttention(Q, K, V)

    Uses multi-head attention with num_heads = C // 32.

    Parameters
    ----------
    channels : int       Feature dimension C (query).
    anatomy_dim : int    e_a channel dimension (key/value). Default 96.
    """

    def __init__(self, channels: int, anatomy_dim: int = 96):
        super().__init__()
        self.channels = channels
        self.anatomy_dim = anatomy_dim
        self.num_heads = max(1, channels // 32)

        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            kdim=anatomy_dim,
            vdim=anatomy_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : [B, C, H, W]  feature map (BCHW)
        e_a : [B, 7, C_anat]  anatomy embeddings

        Returns
        -------
        x : [B, C, H, W]  BCHW
        """
        B, C, H, W = x.shape
        residual = x

        # BCHW → [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)

        attn_out, _ = self.attn(x_norm, e_a, e_a)
        x_flat = x_flat + attn_out

        # [B, H*W, C] → BCHW
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# AnatomyMambaBlock
# ═════════════════════════════════════════════════════════════════════════════

class AnatomyMambaBlock(nn.Module):
    """
    Complete anatomy-conditioned Mamba block for Stage 2.

    Data format lifecycle
    ---------------------
    Input  : x [B, C, H, W]  BCHW
    Step 1 : adaLN-Zero → shift/scale for norm
    Step 2 : Spatial FiLM from S → BCHW
    Step 3 : BCHW → BHWC → VSSD scan → BHWC
    Step 4 : BHWC → BCHW → Cross-Attention with e_a → BCHW
    Output : [B, C, H, W]  BCHW + residual

    Parameters
    ----------
    channels : int       Feature dimension C at this scale.
    anatomy_dim : int    e_a embedding dimension (default 96).
    d_state : int        VSSD state dimension (default 8).
    time_emb_dim : int   Timestep embedding dimension (default 256).
    """

    def __init__(
        self,
        channels: int,
        anatomy_dim: int = 96,
        d_state: int = 8,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.channels = channels

        # adaLN-Zero
        self.ada_ln = AdaLNZero(channels, time_emb_dim)

        # LayerNorms (BHWC for VSSD, BCHW for SpatialFiLM)
        self.norm_scan = LayerNorm2d(channels, data_format='channels_last')
        self.norm_attn = LayerNorm2d(channels, data_format='channels_first')

        # Core operations
        self.spatial_film = SpatialFiLM(num_classes=7, channels=channels)
        self.vssd = VSSD(d_model=channels, d_state=d_state)
        self.cross_attn = AnatomyCrossAttention(channels, anatomy_dim)

    def forward(
        self,
        x: torch.Tensor,
        S: torch.Tensor,
        e_a: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : [B, C, H, W]  BCHW feature map
        S     : [B, 7, H, W]  organ probability map (same spatial size)
        e_a   : [B, 7, C_anat]  anatomy embeddings
        t_emb : [B, 256]       timestep embedding

        Returns
        -------
        x : [B, C, H, W]  BCHW
        """
        B, C, H, W = x.shape
        residual = x

        # Get adaLN parameters
        shift_scan, scale_scan, gate_scan, \
            shift_attn, scale_attn, gate_attn = self.ada_ln(t_emb)
        # Each: [B, C] → reshape for broadcasting

        # ── VSSD branch (scanned features go through VSSD) ──────────────
        # BCHW → BHWC
        x_scan = x.permute(0, 2, 3, 1).contiguous()              # [B, H, W, C]

        # adaLN: shift + scale
        s_shift = shift_scan.view(B, 1, 1, C)
        s_scale = scale_scan.view(B, 1, 1, C)
        x_scan_norm = self.norm_scan(x_scan)
        x_scan_mod = x_scan_norm * (1.0 + s_scale) + s_shift    # [B, H, W, C]

        # VSSD bidirectional scan
        x_scan_out = self.vssd(x_scan_mod)                        # [B, H, W, C]

        # Gate + BHWC → BCHW
        g_scan = gate_scan.view(B, 1, 1, C)
        x_scan_out = x_scan_out * g_scan
        x = x + x_scan_out.permute(0, 3, 1, 2).contiguous()     # BHWC → BCHW

        # ── Spatial FiLM branch ────────────────────────────────────────
        s_shift_a = shift_attn.view(B, C, 1, 1)
        s_scale_a = scale_attn.view(B, C, 1, 1)

        # adaLN (BCHW norm) + Spatial FiLM
        x_attn_norm = self.norm_attn(x)
        x_attn_mod = x_attn_norm * (1.0 + s_scale_a) + s_shift_a
        x_film = self.spatial_film(x_attn_mod, S)                # [B, C, H, W]

        # ── Cross-Attention branch ─────────────────────────────────────
        x_ca = self.cross_attn(x_film, e_a)                      # [B, C, H, W]

        # Gate
        g_attn = gate_attn.view(B, C, 1, 1)
        x = x + g_attn * x_ca

        return x


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("models/anatomy_mamba.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, C, H, W = 2, 64, 128, 128
    num_classes, anatomy_dim = 7, 96

    x = torch.randn(B, C, H, W, device=device)
    S = torch.randn(B, num_classes, H, W, device=device).softmax(dim=1)
    e_a = torch.randn(B, num_classes, anatomy_dim, device=device)
    t_emb = torch.randn(B, 256, device=device)

    # ── SpatialFiLM ────────────────────────────────────────────────────
    print("── SpatialFiLM ────────────────────────────────────────────────")
    film = SpatialFiLM(num_classes=num_classes, channels=C).to(device)
    out = film(x, S)
    assert out.shape == x.shape, f"Expected {tuple(x.shape)}, got {tuple(out.shape)}"
    assert not torch.allclose(out, x, atol=1e-4), "Should NOT be identity (modified by S)"
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")
    print(f"  Params: {sum(p.numel() for p in film.parameters()):,}")

    # ── AdaLNZero ──────────────────────────────────────────────────────
    print("\n── AdaLNZero ─────────────────────────────────────────────────")
    adaln = AdaLNZero(C).to(device)
    params = adaln(t_emb)
    assert len(params) == 6, f"Expected 6 param tensors, got {len(params)}"
    for i, p in enumerate(params):
        assert p.shape == (B, C), f"Param {i}: expected ({B},{C}), got {tuple(p.shape)}"
    print(f"  t_emb [B,256] → 6 × [B,{C}] ✓")

    # ── AnatomyCrossAttention ──────────────────────────────────────────
    print("\n── AnatomyCrossAttention ─────────────────────────────────────")
    ca = AnatomyCrossAttention(C, anatomy_dim).to(device)
    out = ca(x, e_a)
    assert out.shape == x.shape, f"Expected {tuple(x.shape)}, got {tuple(out.shape)}"
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")
    print(f"  Params: {sum(p.numel() for p in ca.parameters()):,}")

    # ── AnatomyMambaBlock (full) ───────────────────────────────────────
    print("\n── AnatomyMambaBlock (full block) ────────────────────────────")
    block = AnatomyMambaBlock(C, anatomy_dim, d_state=8).to(device)
    out = block(x, S, e_a, t_emb)
    assert out.shape == x.shape, f"Expected {tuple(x.shape)}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")
    print(f"  Params: {sum(p.numel() for p in block.parameters()):,}")

    # Test with gradient
    x_g = torch.randn(B, C, H, W, device=device, requires_grad=True)
    block.zero_grad()
    out_g = block(x_g, S, e_a, t_emb)
    out_g.mean().backward()
    assert x_g.grad is not None and x_g.grad.norm() > 0, "No gradient flow"
    print(f"  Gradient norm: {x_g.grad.norm().item():.4f} ✓")

    # Test multiple scales
    print("\n── Multiple scales ───────────────────────────────────────────")
    for cc, hw in [(64, 128), (128, 64), (256, 32), (512, 16)]:
        xx = torch.randn(1, cc, hw, hw, device=device)
        ss = torch.randn(1, 7, hw, hw, device=device).softmax(dim=1)
        block_s = AnatomyMambaBlock(cc, anatomy_dim, d_state=8).to(device)
        oo = block_s(xx, ss, e_a[:1], t_emb[:1])
        assert oo.shape == xx.shape, f"Scale (C={cc}, H=W={hw}): {tuple(oo.shape)}"
    print(f"  All 4 scales pass ✓")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
