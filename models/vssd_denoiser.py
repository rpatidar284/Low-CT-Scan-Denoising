"""
models/vssd_denoiser.py

VSSD Anatomy-Conditioned Denoiser UNet — Stage 2 core.
=========================================================
Full UNet architecture that takes a noisy residual + LDCT reference
and predicts the clean residual, conditioned on:

  S_scales  — organ probability maps at 5 resolutions
  e_a       — per-organ feature embeddings from frozen Stage 1
  t         — diffusion timestep

Architecture
------------
  Input: x_noisy [B,1,H,W], x_ldct [B,1,H,W], t, S_scales, e_a

  init_conv : Conv2d(2, 64, k=7, p=3)   ← concat(x_noisy, x_ldct)
  time_mlp  : sinusoidal(t) → MLP → t_emb [B, 256]

  Encoder (4 scales):
    Scale 1: 2×AnatomyMambaBlock(dim=64)   → skip → downsample → [B,128,H/2,W/2]
    Scale 2: 2×AnatomyMambaBlock(dim=128)  → skip → downsample → [B,256,H/4,W/4]
    Scale 3: 2×AnatomyMambaBlock(dim=256)  → skip → downsample → [B,512,H/8,W/8]
    Scale 4: 2×AnatomyMambaBlock(dim=512)  → skip → downsample → [B,512,H/16,W/16]

  Bottleneck:
    2×AnatomyMambaBlock(dim=512)  → [B,512,H/16,W/16]
    Seg KD head: Conv2d(512,7,k=1) → for L_kd

  Decoder (mirror of encoder):
    Scale 4 up → Scale 3 up → Scale 2 up → Scale 1 up → [B,64,H,W]

  final_conv : Conv2d(64, 1, k=1) → predicted_residual [B,1,H,W]

Reference: Architecture.pdf — Stage 2 (VSSD Anatomy-Conditioned Denoiser)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.anatomy_mamba import AnatomyMambaBlock
    from models.diffusion import TimestepEmbedder
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.anatomy_mamba import AnatomyMambaBlock
    from models.diffusion import TimestepEmbedder


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _downsample(dim_in: int, dim_out: int) -> nn.Module:
    """Strided conv for 2× spatial downsampling."""
    return nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)


def _upsample(dim_in: int, dim_out: int) -> nn.Module:
    """Nearest upsampling 2× + conv for channel adjustment."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
    )


def _build_S_scales(S_512: torch.Tensor, sizes: list) -> list:
    """
    Bilinearly downsample S from full resolution to each required scale.

    Parameters
    ----------
    S_512 : [B, 7, 512, 512]
    sizes : list of (H, W) tuples

    Returns
    -------
    list of [B, 7, H_i, W_i]
    """
    return [F.interpolate(S_512, size=sz, mode='bilinear', align_corners=False) for sz in sizes]


# ═════════════════════════════════════════════════════════════════════════════
# VSSD Denoiser UNet
# ═════════════════════════════════════════════════════════════════════════════

class VSSDDenoiser(nn.Module):
    """
    Anatomy-conditioned VSSD UNet for residual prediction.

    Parameters
    ----------
    in_channels : int         Input CT channels (1 for grayscale).
    base_channels : int       Channel count at scale 1 (default 64).
    channel_mults : tuple     Multiplier per scale (1, 2, 4, 8).
    blocks_per_scale : int    Number of AnatomyMambaBlocks per encoder/decoder scale.
    anatomy_dim : int         e_a embedding dimension (default 96).
    num_classes : int         Number of organ classes (default 7).
    d_state : int             VSSD state dimension (default 8).
    image_size : int          Input image spatial size (square, default 512).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        blocks_per_scale: int = 2,
        anatomy_dim: int = 96,
        num_classes: int = 7,
        d_state: int = 8,
        image_size: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.anatomy_dim = anatomy_dim
        self.image_size = image_size

        # Spatial sizes at each scale
        self.scale_sizes = [
            (image_size, image_size),                   # 512
            (image_size // 2, image_size // 2),         # 256
            (image_size // 4, image_size // 4),         # 128
            (image_size // 8, image_size // 8),         # 64
            (image_size // 16, image_size // 16),       # 32
        ]

        # Channel counts per scale
        ch = [base_channels * m for m in (1,) + channel_mults]  # [64, 64, 128, 256, 512]
        encoder_ch = ch[:-1]                                      # [64, 64, 128, 256]
        mid_ch = ch[-1]                                           # 512

        # ── Input projection ──────────────────────────────────────────────
        # Concat(x_noisy, x_ldct) = 2 channels
        self.init_conv = nn.Conv2d(in_channels * 2, base_channels, kernel_size=7, padding=3)

        # ── Time embedding ────────────────────────────────────────────────
        self.time_mlp = TimestepEmbedder(hidden_size=256)

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_ch = base_channels
        for i, enc_ch_i in enumerate(encoder_ch):
            scale_blocks = nn.ModuleList([
                AnatomyMambaBlock(cur_ch, anatomy_dim, d_state)
                for _ in range(blocks_per_scale)
            ])
            self.encoder_blocks.append(scale_blocks)

            next_ch = ch[i + 1]
            self.downsamples.append(_downsample(cur_ch, next_ch))
            cur_ch = next_ch

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            AnatomyMambaBlock(mid_ch, anatomy_dim, d_state)
            for _ in range(blocks_per_scale)
        ])

        # Seg KD head — small conv to produce organ logits from bottleneck
        # Used during training for L_kd
        self.kd_head = nn.Conv2d(mid_ch, num_classes, kernel_size=1)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.decoder_projections = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        cur_ch = mid_ch  # 512
        for skip_ch in reversed(encoder_ch):  # [256, 128, 64, 64]
            # After upsample + concat, we have (skip_ch + skip_ch) channels
            # Project to skip_ch then process with blocks
            concat_ch = skip_ch * 2

            scale_blocks = nn.ModuleList([
                AnatomyMambaBlock(skip_ch, anatomy_dim, d_state)
                for _ in range(blocks_per_scale)
            ])
            self.decoder_blocks.append(scale_blocks)

            # Project concat channels back down to skip_ch for the blocks
            self.decoder_projections.append(
                nn.Conv2d(concat_ch, skip_ch, kernel_size=1)
            )

            self.upsamples.append(_upsample(cur_ch, skip_ch))
            cur_ch = skip_ch

        # ── Output projection ─────────────────────────────────────────────
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(
        self,
        x_ldct: torch.Tensor,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        S_scales: list,
        e_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_ldct   : [B, 1, H, W]  low-dose CT (constant reference)
        x_noisy  : [B, 1, H, W]  noisy residual at timestep t
        t        : [B]           integer timesteps
        S_scales : list of [B, 7, H_i, W_i]  organ probability maps at 5 scales
        e_a      : [B, 7, C_anat]  anatomy embeddings from Stage 1

        Returns
        -------
        pred_res : [B, 1, H, W]  predicted residual
        kd_logits : [B, 7, H, W] or None  seg KD head output (training only)
        """
        # ── Input projection ─────────────────────────────────────────────
        x_in = torch.cat([x_noisy, x_ldct], dim=1)       # [B, 2, H, W]
        x = self.init_conv(x_in)                          # [B, 64, H, W]

        # ── Time embedding ───────────────────────────────────────────────
        t_emb = self.time_mlp(t)                          # [B, 256]

        # ── Encoder ──────────────────────────────────────────────────────
        skips = []
        for i, (blocks, downsample) in enumerate(zip(self.encoder_blocks, self.downsamples)):
            S_i = S_scales[i]
            for blk in blocks:
                x = blk(x, S_i, e_a, t_emb)
            skips.append(x)
            x = downsample(x)

        # ── Bottleneck ───────────────────────────────────────────────────
        S_bottleneck = S_scales[-1]
        for blk in self.bottleneck:
            x = blk(x, S_bottleneck, e_a, t_emb)

        # KD head logits (upsampled to full resolution for L_kd)
        kd_logits = self.kd_head(x)                       # [B, 7, H/16, W/16]
        kd_logits = F.interpolate(kd_logits, size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False)

        # ── Decoder ──────────────────────────────────────────────────────
        for i, (blocks, proj, upsample) in enumerate(zip(
            self.decoder_blocks, self.decoder_projections, self.upsamples
        )):
            x = upsample(x)                                # 2× upsample first
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)               # concat with skip
            x = proj(x)                                   # project channels
            S_i = S_scales[-(i + 2)]                      # matching S for this scale
            for blk in blocks:
                x = blk(x, S_i, e_a, t_emb)

        # ── Output ───────────────────────────────────────────────────────
        pred_res = self.final_conv(x)                     # [B, 1, H, W]

        return pred_res, kd_logits


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

def build_denoiser(image_size: int = 512, **kwargs) -> VSSDDenoiser:
    """Create a VSSDDenoiser with default config."""
    return VSSDDenoiser(image_size=image_size, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("models/vssd_denoiser.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, C, H, W = 2, 1, 256, 256
    num_classes, anatomy_dim = 7, 96

    # Create model at 256px for fast testing
    model = VSSDDenoiser(image_size=H).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M")

    # Build dummy inputs
    x_ldct = torch.randn(B, C, H, W, device=device)
    x_noisy = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    S = torch.randn(B, num_classes, H, W, device=device).softmax(dim=1)
    S_scales = _build_S_scales(S, model.scale_sizes)

    e_a = torch.randn(B, num_classes, anatomy_dim, device=device)

    # Forward pass
    print(f"\n── Forward pass ───────────────────────────────────────────────")
    pred_res, kd_logits = model(x_ldct, x_noisy, t, S_scales, e_a)

    assert pred_res.shape == (B, C, H, W), f"pred_res: expected {(B,C,H,W)}, got {tuple(pred_res.shape)}"
    assert kd_logits.shape == (B, num_classes, H, W), f"kd_logits: expected {(B,num_classes,H,W)}, got {tuple(kd_logits.shape)}"
    assert torch.isfinite(pred_res).all(), "pred_res contains non-finite values"
    assert torch.isfinite(kd_logits).all(), "kd_logits contains non-finite values"
    print(f"  pred_res : {tuple(pred_res.shape)} ✓")
    print(f"  kd_logits: {tuple(kd_logits.shape)} ✓")

    # Gradient flow
    print(f"\n── Gradient flow ──────────────────────────────────────────────")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = pred_res.mean() + kd_logits.mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "No gradients flowing"
    print(f"  Total gradient norm: {grad_norm:.4f} ✓")

    # S scale sizes
    print(f"\n── S scale sizes ───────────────────────────────────────────────")
    for i, (sz, s_tensor) in enumerate(zip(model.scale_sizes, S_scales)):
        print(f"  Scale {i}: S {tuple(s_tensor.shape)}")

    print(f"\n{'='*60}")
    print("All tests PASSED")
    print("=" * 60)
