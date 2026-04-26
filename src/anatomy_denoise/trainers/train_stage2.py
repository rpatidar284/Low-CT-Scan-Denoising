import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anatomy_denoise.config import TrainConfig
from anatomy_denoise.data.dataset import LDCTPairDataset
from anatomy_denoise.models.stage1_teacher import Stage1Teacher
from anatomy_denoise.models.stage2_denoiser import Stage2Denoiser


def linear_beta_schedule(timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(1e-4, 2e-2, timesteps, device=device)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    a = alphas_cumprod[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise


def parse_batch(
    batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse either dict-style or tuple-style dataset batch outputs."""
    if isinstance(batch, dict):
        ndct = batch["ndct"]
        ldct = batch["ldct"]
        mask = batch.get("pseudo_mask", batch.get("mask"))
        if mask is None:
            raise KeyError("Batch dict must contain 'mask' or 'pseudo_mask'.")
        return ldct, ndct, mask
    if isinstance(batch, tuple) and len(batch) == 3:
        ndct, ldct, mask = batch
        return ldct, ndct, mask
    raise TypeError("Unsupported batch format. Expected dict or 3-tuple.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--train_split", type=Path, default=Path("data/splits/train.txt"))
    parser.add_argument("--stage1_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/stage2"))
    args = parser.parse_args()

    cfg = TrainConfig()
    ds = LDCTPairDataset(args.data_root, args.train_split)
    dl = DataLoader(ds, batch_size=cfg.stage2.batch_size, shuffle=True, num_workers=4, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = Stage1Teacher(base=cfg.stage1.embed_dim, num_classes=cfg.stage1.num_classes).to(device)
    stage1.load_state_dict(torch.load(args.stage1_ckpt, map_location=device)["model"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False

    stage2 = Stage2Denoiser(base=64, num_classes=cfg.stage2.num_classes).to(device)
    opt = torch.optim.AdamW(stage2.parameters(), lr=cfg.stage2.lr, weight_decay=cfg.stage2.weight_decay)

    betas = linear_beta_schedule(cfg.stage2.timesteps, device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    while step < cfg.stage2.steps:
        pbar = tqdm(dl, desc=f"Stage2 step {step}/{cfg.stage2.steps}")
        for batch in pbar:
            if step >= cfg.stage2.steps:
                break
            ldct, ndct, mask = parse_batch(batch)
            ldct = ldct.to(device)
            ndct = ndct.to(device)
            mask = mask.to(device)
            b = ldct.size(0)

            with torch.no_grad():
                t1 = stage1(ldct)
                s = t1["S"]
                e_a = t1["e_a"]

            s_scales = [
                F.interpolate(s, size=(256, 256), mode="bilinear", align_corners=False),
                F.interpolate(s, size=(128, 128), mode="bilinear", align_corners=False),
                F.interpolate(s, size=(64, 64), mode="bilinear", align_corners=False),
                F.interpolate(s, size=(32, 32), mode="bilinear", align_corners=False),
            ]

            true_residual = ndct - ldct
            t = torch.randint(0, cfg.stage2.timesteps, (b,), device=device)
            noise = torch.randn_like(true_residual)
            noisy_x = q_sample(true_residual, t, noise, alphas_cumprod)

            out = stage2(noisy_x, ldct, t, s_scales, e_a)
            pred_noise = out["pred_residual"]
            loss_res = F.mse_loss(pred_noise, noise)

            loss_kd = torch.tensor(0.0, device=device)
            if step >= cfg.stage2.kd_start_step:
                loss_kd = F.cross_entropy(out["seg_kd"], mask, label_smoothing=0.1)

            loss_anatomy = torch.tensor(0.0, device=device)
            if step >= cfg.stage2.anatomy_start_step and step % cfg.stage2.anatomy_every_n_steps == 0:
                x_hat = ldct + pred_noise
                with torch.no_grad():
                    e_pred = stage1(x_hat)["e_a"]
                    e_gt = stage1(ndct)["e_a"]
                loss_anatomy = F.l1_loss(e_pred, e_gt)

            loss = (
                cfg.stage2.lambda_res * loss_res
                + cfg.stage2.lambda_kd * loss_kd
                + cfg.stage2.lambda_anatomy * loss_anatomy
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()), l_res=float(loss_res.detach().cpu()))

            if step % 5000 == 0:
                torch.save({"model": stage2.state_dict(), "step": step}, args.output_dir / f"stage2_step_{step}.pt")


if __name__ == "__main__":
    main()

