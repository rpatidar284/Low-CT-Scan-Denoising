from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AnatomyMambaBlock, ConvBlock, Down, Up, sinusoidal_embedding


class Stage2Denoiser(nn.Module):
    def __init__(self, base: int = 64, num_classes: int = 7, t_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.t_dim = t_dim

        # No DA-CLIP: fixed 25% dose only, so we condition on timestep only.
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.init = ConvBlock(2, base)
        self.block1 = AnatomyMambaBlock(base, t_dim, num_classes)
        self.d1 = Down(base, base * 2)
        self.block2 = AnatomyMambaBlock(base * 2, t_dim, num_classes)
        self.d2 = Down(base * 2, base * 4)
        self.block3 = AnatomyMambaBlock(base * 4, t_dim, num_classes)
        self.d3 = Down(base * 4, base * 8)
        self.mid = AnatomyMambaBlock(base * 8, t_dim, num_classes)

        self.seg_kd_head = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base * 4, num_classes, 4, stride=2, padding=1),
        )

        self.u3 = Up(base * 12, base * 4)
        self.ub3 = AnatomyMambaBlock(base * 4, t_dim, num_classes)
        self.u2 = Up(base * 6, base * 2)
        self.ub2 = AnatomyMambaBlock(base * 2, t_dim, num_classes)
        self.u1 = Up(base * 3, base)
        self.ub1 = AnatomyMambaBlock(base, t_dim, num_classes)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        x_ldct: torch.Tensor,
        t: torch.Tensor,
        s_scales: List[torch.Tensor],
        e_a: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cond = self.time_mlp(sinusoidal_embedding(t, self.t_dim))
        x = torch.cat([noisy_residual, x_ldct], dim=1)
        x1 = self.init(x)
        x1 = self.block1(x1, cond, s_scales[0], e_a)
        x2 = self.d1(x1)
        x2 = self.block2(x2, cond, s_scales[1], e_a)
        x3 = self.d2(x2)
        x3 = self.block3(x3, cond, s_scales[2], e_a)
        x4 = self.d3(x3)
        mid = self.mid(x4, cond, s_scales[3], e_a)

        seg_kd = self.seg_kd_head(mid)
        seg_kd = F.interpolate(seg_kd, size=x_ldct.shape[-2:], mode="bilinear", align_corners=False)

        y = self.u3(mid, x3)
        y = self.ub3(y, cond, s_scales[2], e_a)
        y = self.u2(y, x2)
        y = self.ub2(y, cond, s_scales[1], e_a)
        y = self.u1(y, x1)
        y = self.ub1(y, cond, s_scales[0], e_a)
        pred = self.out(y)
        return {"pred_residual": pred, "seg_kd": seg_kd, "mid": mid}

