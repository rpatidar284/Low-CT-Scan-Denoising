"""Stage-1 training script for Mayo 2020 Simulated dataset.

Improvements:
- Resume from checkpoint (--resume)
- Periodic checkpoint saving (--save_every)
- torch.cuda.empty_cache() between epochs
- Proper device placement before trainer
- Cleaner progress bar with ETA
- --train_size cap for quick debug runs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure repo root is on path regardless of cwd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pdf_dataset import build_mayo_pair_paths, PDFDataset         # noqa: E402
from src.stage1_trainer import Trainer, parse_batch                    # noqa: E402


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 anatomy pretraining")
    p.add_argument("--mayo_root",   type=Path, required=True)
    p.add_argument("--mask_dir",    type=Path, required=True)
    p.add_argument("--output_dir",  type=Path, default=Path("outputs/mayo_stage1"))
    p.add_argument("--epochs",      type=int,  default=10)
    p.add_argument("--batch_size",  type=int,  default=1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--lambda_byol", type=float, default=0.1)
    p.add_argument("--byol_warmup", type=int,   default=5000)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--train_size",  type=int,   default=None,
                   help="Cap dataset size (useful for quick debug runs)")
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--amp_dtype",   choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--save_every",  type=int,   default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume",      type=Path,  default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  amp={args.amp}({args.amp_dtype})")

    # ── dataset ───────────────────────────────────────────────────────
    high_paths, low_paths = build_mayo_pair_paths(args.mayo_root)
    dataset = PDFDataset(high_paths, low_paths, mask_dir=args.mask_dir)
    if args.train_size is not None:
        dataset = Subset(dataset, range(min(args.train_size, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    print(f"[train] dataset size={len(dataset)}  steps/epoch={len(loader)}")

    # ── trainer ───────────────────────────────────────────────────────
    total_steps = len(loader) * args.epochs
    config = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_byol=args.lambda_byol,
        byol_warmup_steps=args.byol_warmup,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        total_steps=total_steps,
    )
    trainer = Trainer(config).to(device)

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        print(f"[train] resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        running = {"loss": 0.0, "L_seg": 0.0, "L_byol": 0.0}
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for batch in pbar:
            ldct, ndct, mask = parse_batch(batch)
            ldct = ldct.to(device, non_blocking=True)
            ndct = ndct.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            losses = trainer.train_step(ldct, ndct, mask, global_step)
            global_step += 1

            for k in running:
                running[k] += losses[k]

            pbar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                seg=f"{losses['L_seg']:.4f}",
                byol=f"{losses['L_byol']:.4f}",
                step=global_step,
            )

        # ── epoch summary ─────────────────────────────────────────────
        n = len(loader)
        print(
            f"[epoch {epoch+1}] "
            f"loss={running['loss']/n:.4f}  "
            f"seg={running['L_seg']/n:.4f}  "
            f"byol={running['L_byol']/n:.4f}"
        )

        # ── checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt = args.output_dir / f"stage1_epoch_{epoch+1:03d}.pt"
            trainer.save_checkpoint(ckpt)
            print(f"[train] saved {ckpt}")

        # ── free VRAM between epochs ───────────────────────────────────
        torch.cuda.empty_cache()

    # ── final checkpoint ──────────────────────────────────────────────
    final = args.output_dir / "stage1_final.pt"
    trainer.save_checkpoint(final)
    print(f"[train] training complete. Final checkpoint: {final}")


if __name__ == "__main__":
    main()