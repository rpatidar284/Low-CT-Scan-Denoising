"""BYOL utility components for Stage-1 training."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorMLP(nn.Module):
    """Two-layer MLP predictor used by BYOL."""

    def __init__(self, dim: int = 768, hidden: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply predictor mapping on latent vectors."""
        return self.net(x)


class EMAUpdater:
    """Exponential moving average updater for target model parameters."""

    def __init__(self, tau: float = 0.996) -> None:
        self.tau = tau

    def update(self, online_model: nn.Module, target_model: nn.Module) -> None:
        """Update target model weights with EMA of online model."""
        with torch.no_grad():
            for online_param, target_param in zip(
                online_model.parameters(), target_model.parameters()
            ):
                target_param.data.mul_(self.tau).add_(online_param.data, alpha=1.0 - self.tau)

    def update_tau(self, step: int, total_steps: int) -> None:
        """Cosine-anneal tau from base value toward 1.0."""
        if total_steps <= 0:
            return
        progress = min(max(step / float(total_steps), 0.0), 1.0)
        self.tau = 1.0 - (1.0 - 0.996) * (math.cos(math.pi * progress) + 1.0) * 0.5


def byol_loss(z_online: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Compute BYOL cosine regression loss."""
    z_online = F.normalize(z_online, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    return 2.0 - 2.0 * (z_online * z_target).sum(dim=-1).mean()

