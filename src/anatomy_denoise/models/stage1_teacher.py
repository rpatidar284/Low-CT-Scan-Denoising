import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, Down, Up


class Stage1Teacher(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 96, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.inc = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 8)
        self.u3 = Up(base * 12, base * 4)
        self.u2 = Up(base * 6, base * 2)
        self.u1 = Up(base * 3, base)
        self.seg_head = nn.Conv2d(base, num_classes, 1)
        self.proj_dim = 256
        self.projector = nn.Sequential(
            nn.Linear(base * 8, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 256)
        )
        self.predictor = nn.Sequential(
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 256)
        )

    def encode_decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        bottleneck = self.mid(x4)
        y = self.u3(bottleneck, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)
        logits = self.seg_head(y)
        s = F.softmax(logits, dim=1)
        return {"logits": logits, "S": s, "decoder_features": y, "F": bottleneck}

    @staticmethod
    def masked_average_pooling(s: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        b, k, h, w = s.shape
        c = feat.shape[1]
        out = []
        for cls in range(k):
            weight = s[:, cls, :, :].unsqueeze(1)
            weighted = weight * feat
            summed = weighted.sum(dim=(2, 3))
            norm = weight.sum(dim=(2, 3)).clamp_min(1e-6)
            out.append(summed / norm)
        return torch.stack(out, dim=1).view(b, k, c)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encode_decode(x)
        e_a = self.masked_average_pooling(out["S"], out["decoder_features"])
        pooled = F.adaptive_avg_pool2d(out["F"], 1).flatten(1)
        z = self.projector(pooled)
        q = self.predictor(z)
        out["e_a"] = e_a
        out["z"] = z
        out["q"] = q
        return out


class Stage1BYOL:
    def __init__(self, online_net: Stage1Teacher):
        self.online = online_net
        self.target = copy.deepcopy(online_net)
        for p in self.target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau: float = 0.996) -> None:
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

