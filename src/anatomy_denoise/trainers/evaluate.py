import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from anatomy_denoise.data.dataset import LDCTPairDataset
from anatomy_denoise.models.stage1_teacher import Stage1Teacher
from anatomy_denoise.models.stage2_denoiser import Stage2Denoiser


def psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((max_val * max_val) / mse))


def anatomy_weighted_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    ssim_map = ssim(pred, gt, data_range=gt.max() - gt.min() + 1e-6, full=True)[1]
    weights = np.where(mask > 0, 3.0, 1.0).astype(np.float32)
    return float((weights * ssim_map).sum() / (weights.sum() + 1e-6))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.05) -> float:
    def edges(x: np.ndarray) -> np.ndarray:
        gx = np.gradient(x, axis=1)
        gy = np.gradient(x, axis=0)
        mag = np.sqrt(gx * gx + gy * gy)
        return mag > thresh

    e1 = edges(pred)
    e2 = edges(gt)
    tp = np.logical_and(e1, e2).sum()
    fp = np.logical_and(e1, np.logical_not(e2)).sum()
    fn = np.logical_and(np.logical_not(e1), e2).sum()
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    return float(2 * p * r / (p + r + 1e-6))


def make_s_scales(s: torch.Tensor):
    return [
        F.interpolate(s, size=(256, 256), mode="bilinear", align_corners=False),
        F.interpolate(s, size=(128, 128), mode="bilinear", align_corners=False),
        F.interpolate(s, size=(64, 64), mode="bilinear", align_corners=False),
        F.interpolate(s, size=(32, 32), mode="bilinear", align_corners=False),
    ]


def infer_simple(stage1: Stage1Teacher, stage2: Stage2Denoiser, ldct: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        t1 = stage1(ldct)
        s = t1["S"]
        e_a = t1["e_a"]
        s_scales = make_s_scales(s)
        z = torch.randn_like(ldct)
        steps = 50
        for i in reversed(range(steps)):
            t = torch.full((ldct.size(0),), i * (1000 // steps), device=ldct.device, dtype=torch.long)
            pred = stage2(z, ldct, t, s_scales, e_a)["pred_residual"]
            z = z - pred / float(steps)
        return ldct + z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--val_split", type=Path, default=Path("data/splits/val.txt"))
    parser.add_argument("--stage1_ckpt", type=Path, required=True)
    parser.add_argument("--stage2_ckpt", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LDCTPairDataset(args.data_root, args.val_split)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    stage1 = Stage1Teacher().to(device)
    stage1.load_state_dict(torch.load(args.stage1_ckpt, map_location=device)["model"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False

    stage2 = Stage2Denoiser().to(device)
    stage2.load_state_dict(torch.load(args.stage2_ckpt, map_location=device)["model"])
    stage2.eval()

    sums: Dict[str, float] = {"psnr": 0.0, "aw_ssim": 0.0, "bf1": 0.0}
    n = 0
    for batch in tqdm(dl, desc="Evaluating"):
        ldct = batch["ldct"].to(device)
        ndct = batch["ndct"].to(device)
        mask = batch["mask"].cpu().numpy()[0]

        pred = infer_simple(stage1, stage2, ldct).cpu().numpy()[0, 0]
        gt = ndct.cpu().numpy()[0, 0]

        sums["psnr"] += psnr(pred, gt)
        sums["aw_ssim"] += anatomy_weighted_ssim(pred, gt, mask)
        sums["bf1"] += boundary_f1(pred, gt)
        n += 1

    out = {k: v / max(n, 1) for k, v in sums.items()}
    print(out)


if __name__ == "__main__":
    main()

