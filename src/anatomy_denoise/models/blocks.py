import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / (half - 1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNZero(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = self.norm(x)
        return x * (1.0 + gamma) + beta


class SpatialFiLM(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 7):
        super().__init__()
        self.film = nn.Sequential(
            nn.Conv2d(num_classes, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2 * hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(s_feat).chunk(2, dim=1)
        return (1.0 + gamma) * x + beta


class CrossAttentionWithAnatomy(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        heads = max(1, hidden_size // 32)
        self.attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)

    def forward(self, x: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        attended, _ = self.attn(x_flat, e_a, e_a)
        attended = attended.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + attended


class AnatomyMambaBlock(nn.Module):
    """
    Practical stand-in for VSSD block:
    depthwise + pointwise conv keeps spatial global-ish mixing lightweight.
    """

    def __init__(self, channels: int, cond_dim: int, num_classes: int = 7):
        super().__init__()
        self.adaln = AdaLNZero(channels, cond_dim)
        self.film = SpatialFiLM(channels, num_classes)
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.SiLU()
        self.cross = CrossAttentionWithAnatomy(channels)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        s_feat: torch.Tensor,
        e_a: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.adaln(x, cond)
        x = self.film(x, s_feat)
        x = self.pw(self.act(self.dw(x)))
        x = self.cross(x, e_a)
        return x + residual


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.AvgPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

