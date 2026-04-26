import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anatomy_denoise.config import TrainConfig
from anatomy_denoise.data.dataset import LDCTPairDataset
from anatomy_denoise.models.stage1_teacher import Stage1BYOL, Stage1Teacher


def byol_loss(q: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    return (2.0 - 2.0 * (q * z_target).sum(dim=-1)).mean()


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
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/stage1"))
    args = parser.parse_args()

    cfg = TrainConfig()
    ds = LDCTPairDataset(args.data_root, args.train_split)
    dl = DataLoader(ds, batch_size=cfg.stage1.batch_size, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online = Stage1Teacher(base=cfg.stage1.embed_dim, num_classes=cfg.stage1.num_classes).to(device)
    byol = Stage1BYOL(online)
    byol.target.to(device)
    optimizer = torch.optim.AdamW(online.parameters(), lr=cfg.stage1.lr, weight_decay=cfg.stage1.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    for epoch in range(cfg.stage1.epochs):
        pbar = tqdm(dl, desc=f"Stage1 Epoch {epoch + 1}/{cfg.stage1.epochs}")
        for batch in pbar:
            ldct, ndct, mask = parse_batch(batch)
            ldct = ldct.to(device)
            ndct = ndct.to(device)
            mask = mask.to(device)

            out1 = online(ndct)
            seg_loss = F.cross_entropy(out1["logits"], mask, label_smoothing=0.1)

            with torch.no_grad():
                out2_t = byol.target(ldct)
                out1_t = byol.target(ndct)
            out2_o = online(ldct)
            l_byol = byol_loss(out1["q"], out2_t["z"]) + byol_loss(out2_o["q"], out1_t["z"])

            byol_w = 0.0 if epoch < cfg.stage1.byol_warmup_epochs else cfg.stage1.byol_weight
            loss = cfg.stage1.seg_weight * seg_loss + byol_w * l_byol

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            byol.update_target()
            global_step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()), seg=float(seg_loss.detach().cpu()))

        ckpt_path = args.output_dir / f"stage1_epoch_{epoch + 1:03d}.pt"
        torch.save({"model": online.state_dict(), "epoch": epoch + 1, "step": global_step}, ckpt_path)


if __name__ == "__main__":
    main()

