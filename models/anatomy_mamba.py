"""
models/anatomy_mamba.py

Anatomy-conditioned building blocks for Stage 2 denoiser.

Modules:
  ResNetBlock         — Conv + GroupNorm + SiLU (stabilizes training)
  SpatialFiLM         — S → per-pixel γ, β (organ-conditioned modulation)
  VSSDBlock           — adaLN-Zero + VSSD scan + optional cross-attention with e_a
  AnatomyCrossAttention — pixels query e_a to retrieve organ appearance
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
# ResNetBlock — simple conv block for training stability
# ═════════════════════════════════════════════════════════════════════════════

class ResNetBlock(nn.Module):
    """GroupNorm → Conv3×3 → SiLU. Simple but stabilizes deep UNets."""

    def __init__(self, dim, dim_out=None, groups=8):
        super().__init__()
        dim_out = dim_out or dim
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.res_conv(x)


# ═════════════════════════════════════════════════════════════════════════════
# SpatialFiLM — organ-conditioned per-pixel modulation from S
# ═════════════════════════════════════════════════════════════════════════════

class SpatialFiLM(nn.Module):
    """Converts S [B,7,H,W] into per-pixel scale/shift for feature modulation."""

    def __init__(self, num_classes=7, channels=64, hidden=64):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(num_classes, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 1),
        )

    def forward(self, x, S):
        if S.shape[2:] != x.shape[2:]:
            S = F.interpolate(S, size=x.shape[2:], mode='bilinear', align_corners=False)
        params = self.net(S)
        gamma, beta = params.chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


# ═════════════════════════════════════════════════════════════════════════════
# AdaLN-Zero — timestep-conditioned LayerNorm
# ═════════════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    """t_emb → (shift_scan, scale_scan, shift_attn, scale_attn). Zero-initialized."""

    def __init__(self, channels, time_emb_dim=256):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 4 * channels, bias=True),
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, t_emb):
        return self.modulation(t_emb).chunk(4, dim=1)  # 4 × [B, C]


# ═════════════════════════════════════════════════════════════════════════════
# AnatomyCrossAttention — pixels attend to organ embeddings
# ═════════════════════════════════════════════════════════════════════════════

class AnatomyCrossAttention(nn.Module):
    """Flattened pixels query e_a [B,7,96] via MultiheadAttention."""

    def __init__(self, channels, anatomy_dim=96):
        super().__init__()
        self.num_heads = max(1, channels // 32)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, kdim=anatomy_dim, vdim=anatomy_dim,
            num_heads=self.num_heads, batch_first=True,
        )

    def forward(self, x, e_a):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, e_a, e_a)
        x_flat = x_flat + attn_out
        return x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ═════════════════════════════════════════════════════════════════════════════
# VSSDBlock — adaLN-Zero + VSSD scan + optional cross-attention
# ═════════════════════════════════════════════════════════════════════════════

class VSSDBlock(nn.Module):
    """
    Core block: adaLN-Zero → VSSD scan → optional cross-attention with e_a.
    Does NOT include SpatialFiLM (that's applied separately at each UNet scale).
    """

    def __init__(self, channels, anatomy_dim=96, d_state=8, time_emb_dim=256,
                 use_cross_attn=True):
        super().__init__()
        self.channels = channels
        self.use_cross_attn = use_cross_attn

        self.ada_ln = AdaLNZero(channels, time_emb_dim)
        self.norm_scan = LayerNorm2d(channels, data_format='channels_last')
        self.vssd = VSSD(d_model=channels, d_state=d_state)

        if use_cross_attn:
            self.norm_attn = LayerNorm2d(channels, data_format='channels_last')
            self.cross_attn = AnatomyCrossAttention(channels, anatomy_dim)

    def forward(self, x, e_a=None, t_emb=None):
        B, C, H, W = x.shape

        shift_scan, scale_scan, shift_attn, scale_attn = self.ada_ln(t_emb)

        # VSSD branch
        x_v = x.permute(0, 2, 3, 1).contiguous()
        s_sh = shift_scan.view(B, 1, 1, C)
        s_sc = scale_scan.view(B, 1, 1, C)
        x_v = self.norm_scan(x_v) * (1.0 + s_sc) + s_sh
        x_v = self.vssd(x_v)
        x = x + x_v.permute(0, 3, 1, 2).contiguous()

        # Cross-attention branch (optional)
        if self.use_cross_attn and e_a is not None:
            x_ca = x.permute(0, 2, 3, 1).contiguous()
            s_ah = shift_attn.view(B, 1, 1, C)
            s_ac = scale_attn.view(B, 1, 1, C)
            x_ca = self.norm_attn(x_ca) * (1.0 + s_ac) + s_ah
            x_ca = x_ca.permute(0, 3, 1, 2).contiguous()
            x = x + self.cross_attn(x_ca, e_a)

        return x


# ═════════════════════════════════════════════════════════════════════════════
# Compat: AnatomyMambaBlock wraps SpatialFiLM + VSSDBlock
# ═════════════════════════════════════════════════════════════════════════════

class AnatomyMambaBlock(nn.Module):
    """Legacy/compat block with SpatialFiLM built in. Used at each UNet scale."""

    def __init__(self, channels, anatomy_dim=96, d_state=8, time_emb_dim=256):
        super().__init__()
        self.vssd_block = VSSDBlock(channels, anatomy_dim, d_state, time_emb_dim, use_cross_attn=True)
        self.film = SpatialFiLM(num_classes=7, channels=channels)

    def forward(self, x, S, e_a, t_emb):
        x = self.vssd_block(x, e_a, t_emb)
        x = self.film(x, S)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# Self-test
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

    print("── ResNetBlock ──")
    rn = ResNetBlock(C).to(device)
    out = rn(x)
    assert out.shape == x.shape
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")

    print("\n── SpatialFiLM ──")
    film = SpatialFiLM(num_classes=num_classes, channels=C).to(device)
    out = film(x, S)
    assert out.shape == x.shape
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")

    print("\n── VSSDBlock ──")
    block = VSSDBlock(C, anatomy_dim).to(device)
    out = block(x, e_a, t_emb)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")

    print("\n── AnatomyMambaBlock (FiLM+VSSD) ──")
    amb = AnatomyMambaBlock(C).to(device)
    out = amb(x, S, e_a, t_emb)
    assert out.shape == x.shape
    print(f"  {tuple(x.shape)} → {tuple(out.shape)} ✓")
    print(f"  Params: {sum(p.numel() for p in amb.parameters()):,}")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
