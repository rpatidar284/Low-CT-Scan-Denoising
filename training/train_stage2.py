"""
training/train_stage2.py

Stage 2 Training Loop — Anatomy-Conditioned VSSD Denoiser.

Progressive schedule:
  Phase 1  (0 – 50k):      L = L_res
  Phase 2  (50k – 150k):   L = L_res + 0.1 * L_kd
  Phase 3  (150k+):        L = L_res + 0.1 * L_kd + 0.05 * L_anatomy
"""

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models.stage2 import Stage2Model
from losses.stage2_losses import Stage2LossManager
from datapy.dataset import CTSliceDataset, DummyCTDataset


def setup_logger(log_path):
    logger = logging.getLogger('stage2')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for h in [logging.StreamHandler(), logging.FileHandler(log_path, mode='a')]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


class _ProgressBar:
    def __init__(self, total, initial=0):
        self._pbar = None
        if _tqdm is not None:
            self._pbar = _tqdm(total=total, initial=initial, unit='step',
                               desc='stage2', dynamic_ncols=True, mininterval=0.25)

    def step(self, L_res, lr, ep):
        if self._pbar is None: return
        self._pbar.update(1)
        self._pbar.set_postfix({'L_res': f'{L_res:.4f}', 'lr': f'{lr:.1e}', 'ep': str(ep)}, refresh=False)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def _save_checkpoint(path, step, epoch, model, optimizer, best_psnr, config, logger=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': step, 'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr, 'config': config,
    }, path)
    if logger:
        logger.info(f"Checkpoint → {Path(path).name} (best_psnr={best_psnr:.2f})")


def _load_checkpoint(path, model, optimizer, device, logger=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    best_psnr = ckpt.get('best_psnr', 0.0)
    if logger:
        logger.info(f"Resumed from step {step} | best_psnr={best_psnr:.2f}")
    return step, epoch, best_psnr


@torch.no_grad()
def _compute_psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item())


@torch.no_grad()
def _validate(model, val_loader, device, max_batches, step, epoch, logger):
    model.denoiser.eval()
    psnrs = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        ldct = batch['ldct'].to(device)
        ndct = batch['ndct'].to(device)
        S_scales, e_a = model.get_conditioning(ldct)
        shape = (ldct.shape[0], 1, model.image_size, model.image_size)
        x_denoised = model.diffusion.ddim_sample(model.denoiser, ldct, S_scales, e_a, shape)
        psnrs.append(_compute_psnr(x_denoised, ndct))
    model.denoiser.train()
    mean = sum(psnrs) / len(psnrs) if psnrs else 0
    if logger:
        logger.info(f"Step {step:6d} | Val PSNR = {mean:.2f} dB")
    return mean


def train_stage2(
    stage1_checkpoint=None,
    data_root='/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root='/home/teaching/Music/Nigam_51/Project_51/data/masks',
    checkpoint_dir='/home/teaching/Music/Nigam_51/Project_51/outputs/stage2',
    resume_from=None,
    use_dummy_data=False,
    max_steps=None,
    image_size=256,
    batch_size=2,
    total_steps=200_000,
    lr=2e-4,
    weight_decay=0.01,
    warmup_steps=2000,
    log_every=100,
    val_every=2000,
    save_every=5000,
    val_batches=10,
    num_workers=4,
    timesteps=1000,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(Path(checkpoint_dir) / 'training.log'))
    logger.info(f"Stage 2 start | device={device} | steps={total_steps} | batch={batch_size} | lr={lr:.2e}")

    if use_dummy_data:
        train_ds = DummyCTDataset(length=200, image_size=image_size)
        val_ds = DummyCTDataset(length=40, image_size=image_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_ds = CTSliceDataset(data_root=data_root, masks_root=masks_root, split='train',
                                  augment=True, target_size=image_size)
        val_ds = CTSliceDataset(data_root=data_root, masks_root=masks_root, split='val',
                                augment=False, target_size=image_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True,
                                  persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        logger.info(f"Data loaded | train: {len(train_ds.patients)} pts | val: {len(val_ds.patients)} pts")

    batches_per_epoch = len(train_loader)

    model = Stage2Model(
        stage1_checkpoint=stage1_checkpoint,
        denoiser_kwargs={'image_size': image_size},
        diffusion_kwargs={'timesteps': timesteps, 'loss_type': 'l2'},
        image_size=image_size,
    ).to(device)
    logger.info(f"Model | denoiser: {sum(p.numel() for p in model.denoiser.parameters())/1e6:.1f}M params")

    loss_manager = Stage2LossManager()

    trainable = list(model.denoiser.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    def lr_lambda(s):
        if s < warmup_steps: return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, epoch, best_psnr = 0, 0, 0.0
    if resume_from and Path(resume_from).exists():
        step, epoch, best_psnr = _load_checkpoint(resume_from, model, optimizer, device, logger)
    else:
        logger.info("Starting fresh training from step 0")

    model.train()
    model.stage1.eval()
    loader_iter = infinite_loader(train_loader)
    for _ in range(step % len(train_loader)):
        next(loader_iter)

    running_loss_res = 0.0
    running_steps = 0
    t0 = time.time()
    prog = _ProgressBar(total_steps, initial=step)

    try:
        while step < total_steps:
            lr_val = optimizer.param_groups[0]['lr']
            batch = next(loader_iter)
            ldct = batch['ldct'].to(device, non_blocking=True)
            ndct = batch['ndct'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            with torch.no_grad():
                S_scales, e_a = model.get_conditioning(ldct)

            # Precompute e_a_gt from NDCT (for L_anatomy, every 5th step in phase 3)
            e_a_gt = None
            if loss_manager._get_phase(step) >= 3 and step % 5 == 0:
                with torch.no_grad():
                    _, e_a_gt = model.get_conditioning(ndct)

            B = ldct.shape[0]
            x_res = ldct - ndct
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noise = torch.randn_like(x_res)
            x_noisy = model.diffusion.q_sample(ldct, x_res, t, noise)

            optimizer.zero_grad(set_to_none=True)
            pred_res, kd_logits = model.denoiser(ldct, x_noisy, t, S_scales, e_a)

            loss_res = model.diffusion.training_loss(pred_res, x_res)

            e_a_pred = None
            if loss_manager._get_phase(step) >= 3 and step % 5 == 0:
                x_hat = (ldct - pred_res).detach()
                with torch.no_grad():
                    _, e_a_pred = model.get_conditioning(x_hat)

            losses = loss_manager.compute(step, loss_res, kd_logits, mask, e_a_pred, e_a_gt)
            losses['total'].backward()

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss_res += loss_res.item()
            running_steps += 1

            prog.step(L_res=loss_res.item(), lr=lr_val, ep=epoch)

            if step % log_every == 0:
                avg = running_loss_res / running_steps
                elapsed = time.time() - t0
                logger.info(f"Step {step:7d} | ep={epoch:3d} | L_res={avg:.4f} | lr={lr_val:.2e} | {elapsed/running_steps:.2f}s/step")
                running_loss_res = 0.0
                running_steps = 0
                t0 = time.time()

            if step % val_every == 0:
                psnr = _validate(model, val_loader, device, val_batches, step, epoch, logger)
                if psnr > best_psnr:
                    best_psnr = psnr
                    _save_checkpoint(os.path.join(checkpoint_dir, 'stage2_best.pth'), step, epoch, model, optimizer, best_psnr,
                                     {'image_size': image_size, 'timesteps': timesteps}, logger)

            if step % save_every == 0 and step > 0:
                _save_checkpoint(os.path.join(checkpoint_dir, f'stage2_step_{step}.pth'), step, epoch, model, optimizer, best_psnr,
                                 {'image_size': image_size, 'timesteps': timesteps}, logger)
                ckpt_files = sorted(Path(checkpoint_dir).glob('stage2_step_*.pth'), key=lambda f: int(f.stem.split('_')[-1]))
                for f in ckpt_files[:-3]:
                    f.unlink()

            if step % batches_per_epoch == 0:
                epoch += 1

            step += 1

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        prog.close()
        if step > 0:
            _save_checkpoint(os.path.join(checkpoint_dir, f'stage2_step_{step}_final.pth'), step, epoch, model, optimizer, best_psnr,
                             {'image_size': image_size, 'timesteps': timesteps}, logger)

    logger.info(f"Done | steps={step} | best_psnr={best_psnr:.2f} dB")
    return model


if __name__ == '__main__':
    import tempfile
    print("=" * 60)
    print("training/train_stage2.py — smoke test (10 steps, dummy data)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = train_stage2(
            checkpoint_dir=tmpdir, use_dummy_data=True, max_steps=10,
            image_size=64, batch_size=2, num_workers=0,
            timesteps=100, total_steps=10, warmup_steps=3,
            log_every=2, val_every=5, save_every=100,
        )

    assert isinstance(model, Stage2Model)
    model.eval()
    _dev = next(model.parameters()).device
    x_test = torch.randn(1, 1, 64, 64, device=_dev)
    with torch.no_grad():
        out = model(x_test, mode='inference')
    assert out['x_denoised'].shape == (1, 1, 64, 64)
    assert torch.isfinite(out['x_denoised']).all()
    print("\n" + "=" * 60)
    print("Smoke test: PASSED")
    print("=" * 60)
