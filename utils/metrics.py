"""PSNR, SSIM, RMSE for Stage 2 evaluation."""

import math
import torch
import torch.nn.functional as F


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item())


def compute_rmse(pred, target):
    return math.sqrt(F.mse_loss(pred, target).item())


def compute_ssim(pred, target, window_size=11):
    """SSIM between two single-channel tensors [1, H, W]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    if pred.dim() == 3:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[1] > 1:
        pred = pred.mean(dim=1, keepdim=True)
        target = target.mean(dim=1, keepdim=True)

    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size * window_size)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()
