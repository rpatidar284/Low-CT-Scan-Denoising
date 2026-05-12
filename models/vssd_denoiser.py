"""
models/vssd_denoiser.py

VSSD Anatomy-Conditioned Denoiser UNet — Stage 2.
Matches the guide architecture: ResNet + VSSDBlock + FiLM at each scale.
Two output heads: pred_res + pred_noise (pred_res_noise objective).

Architecture per scale:
  Encoder: ResNetBlock → VSSDBlock → SpatialFiLM → Downsample
  Bottleneck: ResNetBlock → VSSDBlock → ResNetBlock
  Decoder: Upsample → concat(skip) → ResNetBlock → VSSDBlock → SpatialFiLM

Output: pred_res [B,1,H,W] + pred_noise [B,1,H,W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.anatomy_mamba import ResNetBlock, SpatialFiLM, VSSDBlock
    from models.diffusion import TimestepEmbedder
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.anatomy_mamba import ResNetBlock, SpatialFiLM, VSSDBlock
    from models.diffusion import TimestepEmbedder


class _Upsample(nn.Module):
    """Bilinear 2× upsample + conv."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2),
                          mode='bilinear', align_corners=False)
        return self.conv(x)


def _downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)


def _build_S_scales(S_full, sizes):
    return [F.interpolate(S_full, size=sz, mode='bilinear', align_corners=False) for sz in sizes]


class VSSDDenoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8),
                 anatomy_dim=96, num_classes=7, d_state=8, image_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.image_size = image_size
        ch = [base_channels * m for m in (1,) + channel_mults]  # [64, 64, 128, 256, 512]
        encoder_ch = ch[:-1]  # [64, 64, 128, 256]
        mid_ch = ch[-1]  # 512

        n_enc = len(channel_mults)  # 4 encoder scales
        self.scale_sizes = [(image_size >> i, image_size >> i) for i in range(n_enc + 1)]  # 5 sizes: 512,256,128,64,32

        # Input
        self.init_conv = nn.Conv2d(in_channels * 2, base_channels, 7, padding=3)
        self.time_mlp = TimestepEmbedder(256)

        # Encoder
        self.enc_resnets = nn.ModuleList()
        self.enc_vssd = nn.ModuleList()
        self.enc_films = nn.ModuleList()
        self.downs = nn.ModuleList()

        cur_ch = base_channels
        for i, enc_ch_i in enumerate(encoder_ch):
            self.enc_resnets.append(ResNetBlock(cur_ch, cur_ch))
            self.enc_vssd.append(VSSDBlock(cur_ch, anatomy_dim, d_state))
            self.enc_films.append(SpatialFiLM(num_classes, cur_ch))
            self.downs.append(_downsample(cur_ch, ch[i + 1]))
            cur_ch = ch[i + 1]

        # Bottleneck
        self.mid_resnet1 = ResNetBlock(mid_ch, mid_ch)
        self.mid_vssd = VSSDBlock(mid_ch, anatomy_dim, d_state)
        self.mid_resnet2 = ResNetBlock(mid_ch, mid_ch)

        # KD head
        self.kd_head = nn.Conv2d(mid_ch, num_classes, 1)

        # Decoder
        self.dec_resnets = nn.ModuleList()
        self.dec_vssd = nn.ModuleList()
        self.dec_films = nn.ModuleList()
        self.ups = nn.ModuleList()

        cur_ch = mid_ch
        for skip_ch in reversed(encoder_ch):
            self.ups.append(_Upsample(cur_ch, skip_ch))
            self.dec_resnets.append(ResNetBlock(skip_ch + skip_ch, skip_ch))
            self.dec_vssd.append(VSSDBlock(skip_ch, anatomy_dim, d_state))
            self.dec_films.append(SpatialFiLM(num_classes, skip_ch))
            cur_ch = skip_ch

        # Output heads (TWO for pred_res_noise objective)
        self.final_res_block = ResNetBlock(base_channels * 2, base_channels)
        self.res_head = nn.Conv2d(base_channels, in_channels, 1)
        self.noise_head = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x_ldct, x_noisy, t, S_scales, e_a):
        # Input projection
        x_in = torch.cat([x_noisy, x_ldct], dim=1)
        x = self.init_conv(x_in)
        r = x.clone()

        t_emb = self.time_mlp(t)

        # Encoder
        skips = []
        for i, (rn, vssd_blk, film, down) in enumerate(
            zip(self.enc_resnets, self.enc_vssd, self.enc_films, self.downs)
        ):
            x = rn(x)
            x = vssd_blk(x, e_a, t_emb)
            x = film(x, S_scales[i])
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_resnet1(x)
        x = self.mid_vssd(x, e_a, t_emb)
        x = self.mid_resnet2(x)
        kd_logits = F.interpolate(self.kd_head(x),
                                  size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False)

        # Decoder
        for i, (up, rn, vssd_blk, film) in enumerate(
            zip(self.ups, self.dec_resnets, self.dec_vssd, self.dec_films)
        ):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = rn(x)
            x = vssd_blk(x, e_a, t_emb)
            x = film(x, S_scales[-(i + 2)])

        # Output
        x = torch.cat([x, r], dim=1)
        x = self.final_res_block(x)
        pred_res = self.res_head(x)
        pred_noise = self.noise_head(x)
        return pred_res, pred_noise, kd_logits


def build_denoiser(image_size=512, **kwargs):
    return VSSDDenoiser(image_size=image_size, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Self-test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("models/vssd_denoiser.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, C, H = 2, 1, 128
    model = VSSDDenoiser(image_size=H).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    x_ldct = torch.randn(B, C, H, H, device=device)
    x_noisy = torch.randn(B, C, H, H, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    S = torch.randn(B, 7, H, H, device=device).softmax(dim=1)
    S_scales = _build_S_scales(S, model.scale_sizes)
    e_a = torch.randn(B, 7, 96, device=device)

    pred_res, pred_noise, kd_logits = model(x_ldct, x_noisy, t, S_scales, e_a)

    assert pred_res.shape == (B, C, H, H), f"pred_res: {tuple(pred_res.shape)}"
    assert pred_noise.shape == (B, C, H, H), f"pred_noise: {tuple(pred_noise.shape)}"
    assert kd_logits.shape == (B, 7, H, H), f"kd_logits: {tuple(kd_logits.shape)}"
    assert torch.isfinite(pred_res).all()
    assert torch.isfinite(pred_noise).all()
    print(f"  pred_res: {tuple(pred_res.shape)} ✓")
    print(f"  pred_noise: {tuple(pred_noise.shape)} ✓")
    print(f"  kd_logits: {tuple(kd_logits.shape)} ✓")

    # Gradient check
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    (pred_res.mean() + pred_noise.mean() + kd_logits.mean()).backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "No gradients!"
    print(f"  Grad norm: {grad_norm:.2f} ✓")

    print(f"\n{'='*60}")
    print("All tests PASSED")
    print("=" * 60)
