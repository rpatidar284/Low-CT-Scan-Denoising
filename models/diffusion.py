"""
models/diffusion.py

Residual Diffusion for Stage 2 CT denoising.
=============================================
Instead of diffusing the full image, we diffuse the residual:
    true_residual = x_ndct - x_ldct

At training: add Gaussian noise to residual at random timestep t,
             train network to predict the residual.

At inference: start from pure noise, iteratively denoise over T steps,
              then x_denoised = x_ldct + predicted_residual.

No dose embedding — fixed 25% dose.

Reference: Architecture.pdf — Stage 2 (Residual Diffusion)
           FoundDiff/src/DADiff.py — adapted residual formulation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract coefficients at timestep t and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as in Improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal Timestep Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal timestep embedding — standard in diffusion models.

    Input:  t [B]  integer timesteps
    Output: emb [B, dim]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding → MLP → timestep conditioning vector."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin_emb(t))


# ─────────────────────────────────────────────────────────────────────────────
# Residual Diffusion
# ─────────────────────────────────────────────────────────────────────────────

class ResidualDiffusion(nn.Module):
    """
    Diffusion on the residual r = x_ldct - x_ndct.

    Forward (training):
        r       = x_ldct - x_ndct                    # residual (noise pattern)
        x_t     = x_ldct - α_cumsum[t] * r + β_cumsum[t] * ε
        model predicts r (or noise ε), loss = MSE

    Reverse (inference):
        Start from x_T = x_ldct + β_cumsum[T-1] * ε  (heavily corrupted)
        Iteratively denoise → x_0 = x_ldct - predicted_residual ≈ x_ndct

    Parameters
    ----------
    timesteps : int     Number of diffusion steps (default 1000).
    beta_start : float  Linear schedule start.
    beta_end : float    Linear schedule end.
    loss_type : str     'l1' or 'l2' (MSE).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = 'linear',
        loss_type: str = 'l2',
        sampling_timesteps: Optional[int] = None,
        ddim_eta: float = 0.0,
    ):
        super().__init__()

        # ── Build noise schedule ──────────────────────────────────────────
        if schedule == 'cosine':
            betas = _cosine_beta_schedule(timesteps)
        else:
            betas = _linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Residual diffusion uses these cumulative sums:
        #   x_t = x_ldct - α_cumsum[t] * r + β_cumsum[t] * ε
        alphas_cumsum = 1.0 - alphas_cumprod ** 0.5
        betas2_cumsum = 1.0 - alphas_cumprod
        betas_cumsum = torch.sqrt(betas2_cumsum)

        # Cache previous-timestep values for posterior
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.0)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.0)

        # Per-step coefficients
        alphas_step = alphas_cumsum - alphas_cumsum_prev
        alphas_step[0] = alphas_step[1]  # avoid zero at t=0

        betas2_step = betas2_cumsum - betas2_cumsum_prev
        betas2_step[0] = betas2_step[1]

        posterior_variance = betas2_step * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0.0

        # Register all buffers
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
        self._register('posterior_log_variance',
                       torch.log(posterior_variance.clamp(min=1e-20)))

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # Sampling config
        self.sampling_timesteps = sampling_timesteps or timesteps
        self.ddim_eta = ddim_eta

    def _register(self, name: str, val: torch.Tensor):
        self.register_buffer(name, val.to(torch.float32))

    @property
    def loss_fn(self):
        return F.l1_loss if self.loss_type == 'l1' else F.mse_loss

    # ── Forward diffusion ─────────────────────────────────────────────────

    def q_sample(self, x_ldct: torch.Tensor, x_res: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: add noise to the residual.

        x_t = x_ldct - α_cumsum[t] * x_res + β_cumsum[t] * ε

        Parameters
        ----------
        x_ldct : [B, 1, H, W]  low-dose (noisy) CT
        x_res  : [B, 1, H, W]  residual = x_ldct - x_ndct
        t      : [B]            integer timesteps
        noise  : [B, 1, H, W]  optional pre-sampled noise

        Returns
        -------
        x_t : [B, 1, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_res)
        return (
            x_ldct
            - _extract(self.alphas_cumsum, t, x_res.shape) * x_res
            + _extract(self.betas_cumsum, t, x_res.shape) * noise
        )

    # ── Model predictions ─────────────────────────────────────────────────

    def predict_x_start(self, x_t: torch.Tensor, x_ldct: torch.Tensor,
                        pred_res: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Recover clean image:  x_ndct = x_ldct - pred_residual"""
        return x_ldct - pred_res

    def predict_noise_from_res(self, x_t: torch.Tensor, x_ldct: torch.Tensor,
                                pred_res: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (
            (x_t - x_ldct + _extract(self.alphas_cumsum, t, x_t.shape) * pred_res)
            / _extract(self.betas_cumsum, t, x_t.shape)
        )

    # ── Posterior (for DDPM sampling) ─────────────────────────────────────

    def q_posterior(self, x_t: torch.Tensor, x_ldct: torch.Tensor,
                    pred_res: torch.Tensor, t: torch.Tensor):
        x_start_pred = self.predict_x_start(x_t, x_ldct, pred_res, t)
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res
            + _extract(self.posterior_mean_coef3, t, x_t.shape) * x_start_pred
        )
        posterior_var = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var = _extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_log_var

    # ── Loss computation ──────────────────────────────────────────────────

    def training_loss(self, pred_res: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        """Compute L_res = MSE/L1 between predicted and true residual."""
        return self.loss_fn(pred_res, x_res)

    # ── DDIM Sampling ─────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, x_ldct: torch.Tensor,
                    S_scales: list, e_a: torch.Tensor,
                    shape: Tuple[int, ...]) -> torch.Tensor:
        """
        DDIM sampling for fast inference.

        Parameters
        ----------
        model : nn.Module     Stage 2 VSSD denoiser
        x_ldct : [B,1,H,W]   low-dose CT (constant throughout sampling)
        S_scales : list       S at each UNet scale [512, 256, 128, 64, 32]
        e_a : [B,7,C_anat]   anatomy embeddings from Stage 1
        shape : tuple         (B, 1, H, W)

        Returns
        -------
        x_denoised : [B, 1, H, W]  denoised CT ≈ NDCT
        """
        batch, device, total_steps, sample_steps, eta = (
            shape[0], self.betas_step.device,
            self.num_timesteps, self.sampling_timesteps, self.ddim_eta,
        )

        # Build timestep sequence: e.g. [999, 949, 899, ..., 99, 49, 0]
        times = torch.linspace(-1, total_steps - 1, steps=sample_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Start from noisy version of LDCT
        img = x_ldct + _extract(self.betas_cumsum, torch.full((batch,), total_steps - 1,
                                  device=device, dtype=torch.long), x_ldct.shape) * torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            t_batch = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_res, _ = model(x_ldct, img, t_batch, S_scales, e_a)

            if time_next < 0:
                img = x_ldct - pred_res  # x_start = x_ldct - residual
                continue

            alpha = self.alphas_step[time]
            betas2 = self.betas2_step[time]
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas_cumsum = self.betas_cumsum[time]

            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            noise = torch.randn_like(img) if eta > 0 else 0.0

            img = img - alpha * pred_res + sigma2.sqrt() * noise

        return img  # ≈ x_ndct


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("models/diffusion.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Sinusoidal embedding ─────────────────────────────────────────────
    print("── SinusoidalPosEmb ────────────────────────────────────────────")
    emb = SinusoidalPosEmb(256)
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print(f"  t [4] → emb [4, 256] ✓")

    # ── TimestepEmbedder ────────────────────────────────────────────────
    print("\n── TimestepEmbedder ───────────────────────────────────────────")
    te = TimestepEmbedder(256)
    out = te(t)
    assert out.shape == (4, 256)
    assert not torch.allclose(out, torch.zeros_like(out)), "Output should not be zero"
    print(f"  t [4] → t_emb [4, 256] ✓")

    # ── ResidualDiffusion ───────────────────────────────────────────────
    print("\n── ResidualDiffusion ──────────────────────────────────────────")
    diff = ResidualDiffusion(timesteps=100)  # small for testing

    B, C, H, W = 2, 1, 64, 64
    x_ldct = torch.randn(B, C, H, W)
    x_ndct = torch.randn(B, C, H, W)
    x_res = x_ldct - x_ndct

    # Forward diffusion
    t = torch.randint(0, 100, (B,))
    noise = torch.randn_like(x_res)
    x_t = diff.q_sample(x_ldct, x_res, t, noise)
    assert x_t.shape == (B, C, H, W)
    print(f"  q_sample: {tuple(x_t.shape)} ✓")

    # At t=0, x_t should be close to x_ldct (minimal noise)
    t0 = torch.zeros(B, dtype=torch.long)
    x_t0 = diff.q_sample(x_ldct, x_res, t0, noise)
    assert torch.allclose(x_t0, x_ldct, atol=0.1), "At t=0, should be ≈ x_ldct"
    print(f"  t=0 → x_t ≈ x_ldct ✓")

    # At t=T-1 (max noise), x_t should differ significantly from x_ldct
    t_max = torch.full((B,), diff.num_timesteps - 1, dtype=torch.long)
    x_tmax = diff.q_sample(x_ldct, x_res, t_max, noise)
    assert not torch.allclose(x_tmax, x_ldct, atol=1.0), "At t=T-1, should differ from x_ldct"
    print(f"  t=T-1 → x_t ≠ x_ldct ✓")

    # Training loss
    pred_res = torch.randn_like(x_res)
    loss = diff.training_loss(pred_res, x_res)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss > 0
    print(f"  training_loss: {loss.item():.4f} ✓")

    # Predict x_start from residual
    x_start_pred = diff.predict_x_start(x_t, x_ldct, pred_res, t)
    assert x_start_pred.shape == (B, C, H, W)
    print(f"  predict_x_start: {tuple(x_start_pred.shape)} ✓")

    # Register buffer shapes
    print(f"\n  Buffers:")
    print(f"    alphas_step:      {diff.alphas_step.shape}")
    print(f"    alphas_cumsum:     {diff.alphas_cumsum.shape}")
    print(f"    betas_cumsum:      {diff.betas_cumsum.shape}")
    print(f"    num_timesteps:     {diff.num_timesteps}")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
