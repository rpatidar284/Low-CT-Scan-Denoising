"""
models/diffusion.py

Residual Diffusion for Stage 2 CT denoising.
Adapted from DADiff.py ResidualDiffusion (FoundDiff project).

Key formula (from DADiff):
  x_res = x_ldct - x_hdct
  x_t   = x_hdct + alphas_cumsum[t] * x_res + betas_cumsum[t] * noise

Reverse: start from x_ldct + noise (t=T), denoise toward x_hdct (t=0).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Timestep Embedding
# ═════════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        return self.mlp(self.sin_emb(t))


# ═════════════════════════════════════════════════════════════════════════════
# Residual Diffusion
# ═════════════════════════════════════════════════════════════════════════════

class ResidualDiffusion(nn.Module):
    """
    Residual diffusion matching the DADiff formulation.

    Forward: x_t = x_hdct + αₜ·x_res + βₜ·noise
    Reverse: DDIM from x_ldct+noise → x_hdct
    """

    def __init__(self, timesteps=1000, sampling_timesteps=None,
                 beta_start=1e-4, beta_end=0.02, schedule='linear',
                 loss_type='l1', ddim_eta=0.0, sum_scale=0.01):
        super().__init__()

        if schedule == 'cosine':
            betas = _cosine_beta_schedule(timesteps)
        else:
            betas = _linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Residual diffusion cumulative sums
        alphas_cumsum = 1.0 - alphas_cumprod ** 0.5
        betas2_cumsum = 1.0 - alphas_cumprod
        betas_cumsum = torch.sqrt(betas2_cumsum)

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        alphas_step = alphas_cumsum - alphas_cumsum_prev
        alphas_step[0] = alphas_step[1]

        betas2_step = betas2_cumsum - betas2_cumsum_prev
        betas2_step[0] = betas2_step[1]

        posterior_variance = betas2_step * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0.0

        self._register('alphas_step', alphas_step)
        self._register('alphas_cumsum', alphas_cumsum)
        self._register('one_minus_alphas_cumsum', 1.0 - alphas_cumsum)
        self._register('betas2_step', betas2_step)
        self._register('betas_step', torch.sqrt(betas2_step))
        self._register('betas2_cumsum', betas2_cumsum)
        self._register('betas_cumsum', betas_cumsum)
        self._register('posterior_mean_coef1', betas2_cumsum_prev / betas2_cumsum)
        self._register('posterior_mean_coef2',
                       (betas2_step * alphas_cumsum_prev - betas2_cumsum_prev * alphas_step) / betas2_cumsum)
        self._register('posterior_mean_coef3', betas2_step / betas2_cumsum)
        self._register('posterior_variance', posterior_variance)
        self._register('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = sampling_timesteps or timesteps
        self.ddim_eta = ddim_eta
        self.sum_scale = sum_scale

    def _register(self, name, val):
        self.register_buffer(name, val.to(torch.float32))

    @property
    def loss_fn(self):
        return F.l1_loss if self.loss_type == 'l1' else F.mse_loss

    # ── Normalization ────────────────────────────────────────────────────

    @staticmethod
    def normalize(x):
        return x * 2 - 1

    @staticmethod
    def unnormalize(x):
        return (x + 1) * 0.5

    # ── Forward diffusion (matches DADiff) ────────────────────────────────

    def q_sample(self, x_hdct, x_res, t, noise=None):
        """
        x_t = x_hdct + α_cumsum[t] * x_res + β_cumsum[t] * noise

        At t=0 (α≈0): x_t ≈ x_hdct (clean)
        At t=T (α≈1, β≈1): x_t ≈ x_hdct + x_res + noise = x_ldct + noise
        """
        if noise is None:
            noise = torch.randn_like(x_res)
        return (
            x_hdct
            + _extract(self.alphas_cumsum, t, x_res.shape) * x_res
            + _extract(self.betas_cumsum, t, x_res.shape) * noise
        )

    def predict_x_start(self, x_t, pred_res, t):
        return x_t - _extract(self.alphas_cumsum, t, x_t.shape) * pred_res

    def predict_noise_from_res(self, x_t, x_hdct, pred_res, t):
        return (x_t - x_hdct - _extract(self.alphas_cumsum, t, x_t.shape) * pred_res) / \
               _extract(self.betas_cumsum, t, x_t.shape)

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (x_t - _extract(self.alphas_cumsum, t, x_t.shape) * x_res
                - _extract(self.betas_cumsum, t, x_t.shape) * noise)

    # ── Loss computation ─────────────────────────────────────────────────

    def training_losses(self, pred_res, pred_noise, x_res, noise):
        return {
            'loss_res': self.loss_fn(pred_res, x_res),
            'loss_noise': self.loss_fn(pred_noise, noise),
        }

    # ── DDIM Sampling (matches DADiff lines 1276-1365) ───────────────────

    @torch.no_grad()
    def ddim_sample(self, model, x_ldct, S_scales, e_a, shape):
        batch, device = shape[0], self.betas_step.device
        total_steps, sample_steps = self.num_timesteps, self.sampling_timesteps
        eta = self.ddim_eta

        times = torch.linspace(-1, total_steps - 1, steps=sample_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Start from x_ldct + small noise (matching DADiff condition=True start)
        sqrt_sum_scale = math.sqrt(self.sum_scale)
        img = x_ldct + sqrt_sum_scale * torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            t_batch = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_res, pred_noise, _ = model(x_ldct, img, t_batch, S_scales, e_a)

            if time_next < 0:
                # Final step: recover clean image
                img = self.predict_start_from_res_noise(img, t_batch, pred_res, pred_noise)
                continue

            alpha = self.alphas_step[time]
            betas2 = self.betas2_step[time]
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas_cumsum = self.betas_cumsum[time]

            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            noise = torch.randn_like(img) if eta > 0 else 0.0

            img = img - alpha * pred_res + sigma2.sqrt() * noise

        return img  # ≈ x_hdct (clean)


# Helper for cosine schedule
def _cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


# ═════════════════════════════════════════════════════════════════════════════
# Self-test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("models/diffusion.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    emb = SinusoidalPosEmb(256)
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    assert out.shape == (4, 256)
    print(f"  SinusoidalPosEmb: t [4] → [4, 256] ✓")

    te = TimestepEmbedder(256)
    out = te(t)
    assert out.shape == (4, 256)
    print(f"  TimestepEmbedder: t [4] → [4, 256] ✓")

    diff = ResidualDiffusion(timesteps=100, sum_scale=0.01)
    B, C, H, W = 2, 1, 64, 64
    x_hdct = torch.randn(B, C, H, W, device=device)
    x_ldct = torch.randn(B, C, H, W, device=device)
    x_res = x_ldct - x_hdct

    t = torch.randint(0, 100, (B,), device=device)
    noise = torch.randn_like(x_res, device=device)
    x_t = diff.q_sample(x_hdct, x_res, t, noise)
    assert x_t.shape == (B, C, H, W)
    print(f"\n  q_sample(x_hdct): {tuple(x_t.shape)} ✓")

    # At t=0, x_t should be close to x_hdct
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    x_t0 = diff.q_sample(x_hdct, x_res, t0, noise)
    assert torch.allclose(x_t0, x_hdct, atol=0.1), f"t=0 mismatch"
    print(f"  t=0 → x_t ≈ x_hdct ✓")

    # At t=T, x_t should be close to x_ldct + noise
    t_max = torch.full((B,), diff.num_timesteps - 1, dtype=torch.long, device=device)
    x_tmax = diff.q_sample(x_hdct, x_res, t_max, noise)
    expected = x_hdct + \
               _extract(diff.alphas_cumsum, t_max, x_hdct.shape) * x_res + \
               _extract(diff.betas_cumsum, t_max, x_hdct.shape) * noise
    assert torch.allclose(x_tmax, expected, atol=0.1), f"t=T-1 wrong"

    # At t=T, α≈1, β≈1, so x_t ≈ x_hdct + x_res + noise = x_ldct + noise ✓
    print(f"  t=T-1 → x_t ≈ x_ldct + noise ✓")

    # Check predict_x_start recovers hdct from x_t (with true residual)
    recovered = diff.predict_x_start(x_t, x_res, t)
    assert torch.allclose(recovered, x_hdct + _extract(diff.betas_cumsum, t, x_t.shape) * noise, atol=0.1)
    print(f"  predict_x_start: recovers hdct from x_t ✓")

    pred_res = torch.randn_like(x_res, device=device)
    pred_noise = torch.randn_like(x_res, device=device)
    losses = diff.training_losses(pred_res, pred_noise, x_res, noise)
    assert losses['loss_res'] > 0
    assert losses['loss_noise'] > 0
    print(f"  training_losses: L_res={losses['loss_res']:.4f}, L_noise={losses['loss_noise']:.4f} ✓")

    # Normalize/unnormalize
    x_norm = diff.normalize(torch.zeros(1))
    assert torch.allclose(x_norm, torch.tensor(-1.0)), f"normalize(0) should be -1, got {x_norm.item()}"
    x_unnorm = diff.unnormalize(torch.tensor(-1.0))
    assert torch.allclose(x_unnorm, torch.tensor(0.0)), f"unnormalize(-1) should be 0, got {x_unnorm.item()}"
    print(f"  normalize(0)→-1, unnormalize(-1)→0 ✓")

    print(f"\n{'='*60}")
    print("All tests PASSED")
    print("=" * 60)
