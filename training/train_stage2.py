"""
training/train_stage2.py

Stage 2 Training — Anatomy-Conditioned Residual Diffusion.
"""

import logging, math, os, sys, time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import deque

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models.stage2 import Stage2Model
from datapy.dataset import CTSliceDataset, DummyCTDataset
from utils.metrics import compute_psnr, compute_ssim, compute_rmse


def setup_logger(log_path):
    logger = logging.getLogger('stage2')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    for h in [logging.StreamHandler(), logging.FileHandler(log_path, 'a')]:
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch


class _ProgressBar:
    def __init__(self, total, initial=0):
        self._p = None
        if _tqdm is not None:
            self._p = _tqdm(total=total, initial=initial, unit='step', desc='stage2',
                            dynamic_ncols=True, mininterval=0.25)

    def step(self, L_res, L_noise, lr, ep):
        if self._p is None: return
        self._p.update(1)
        self._p.set_postfix({'L_res': f'{L_res:.4f}', 'L_n': f'{L_noise:.4f}',
                             'lr': f'{lr:.1e}', 'ep': str(ep)}, refresh=False)

    def close(self):
        if self._p is not None: self._p.close(); self._p = None


def _save_ckpt(path, step, epoch, model, ema_model, optimizer, best_psnr, config, logger=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'step': step, 'epoch': epoch, 'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict() if ema_model else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr, 'config': config}, path)
    if logger: logger.info(f"Checkpoint → {Path(path).name} (PSNR={best_psnr:.1f})")


def _load_ckpt(path, model, ema_model, optimizer, device, logger=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if ema_model and ckpt.get('ema_state_dict'):
        ema_model.load_state_dict(ckpt['ema_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    s, e, bp = ckpt.get('step', 0), ckpt.get('epoch', 0), ckpt.get('best_psnr', 0.0)
    if logger: logger.info(f"Resumed step {s} | best_psnr={bp:.1f}")
    return s, e, bp


class EMAModel(nn.Module):
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.995):
        super().__init__()
        self.module = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach()

    @torch.no_grad()
    def update(self):
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, sd):
        self.shadow = sd['shadow']
        self.decay = sd['decay']


@torch.no_grad()
def _validate(model, ema_model, val_loader, device, max_batches, step, epoch, logger, diffusion):
    """Use EMA model for validation with DDIM sampling."""
    model.eval()

    # Copy EMA weights into model for inference
    if ema_model:
        backup = {n: p.data.clone() for n, p in model.denoiser.named_parameters()
                  if n in ema_model.shadow}
        for n, p in model.denoiser.named_parameters():
            if n in ema_model.shadow:
                p.data.copy_(ema_model.shadow[n])

    psnrs, ssims, rmses = [], [], []

    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        ldct = batch['ldct'].to(device)
        hdct = batch['ndct'].to(device)

        out = model(ldct, mode='inference')
        x_pred = out['x_denoised']

        psnrs.append(compute_psnr(x_pred, hdct))
        ssims.append(compute_ssim(x_pred, hdct))
        rmses.append(compute_rmse(x_pred, hdct))

    # Restore training weights
    if ema_model:
        for n, p in model.denoiser.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    model.train()
    if psnrs:
        logger.info(f"Step {step:6d} | Val PSNR={sum(psnrs)/len(psnrs):.1f} dB | SSIM={sum(ssims)/len(ssims):.4f} | RMSE={sum(rmses)/len(rmses):.4f}")
    return sum(psnrs) / len(psnrs) if psnrs else 0.0


def train_stage2(
    stage1_checkpoint=None,
    data_root='/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root='/home/teaching/Music/Nigam_51/Project_51/data/masks',
    checkpoint_dir='/home/teaching/Music/Nigam_51/Project_51/outputs/stage2',
    resume_from=None, use_dummy_data=False, max_steps=None,
    image_size=128, batch_size=2, total_steps=10000,
    lr=2e-4, weight_decay=0.0, warmup_steps=500,
    log_every=100, val_every=2000, save_every=2000,
    val_batches=8, num_workers=4, timesteps=1000,
    ema_decay=0.995, ema_update_every=10,
    sampling_timesteps=50, res_weight=1.0, noise_weight=1.0,
    grad_clip=0.5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(Path(checkpoint_dir) / 'training.log'))
    logger.info(f"Stage 2 start | device={device} | steps={total_steps} | batch={batch_size} | lr={lr:.2e} | img={image_size}")

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
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        logger.info(f"Data | train: {len(train_ds.patients)} pts | val: {len(val_ds.patients)} pts")

    batches_per_epoch = len(train_loader)

    model = Stage2Model(
        stage1_checkpoint=stage1_checkpoint,
        denoiser_kwargs={'image_size': image_size},
        diffusion_kwargs={'timesteps': timesteps, 'loss_type': 'l1',
                          'sampling_timesteps': sampling_timesteps, 'sum_scale': 0.01},
        image_size=image_size,
    ).to(device)

    ema_model = EMAModel(model.denoiser, decay=ema_decay) if ema_decay > 0 else None
    logger.info(f"Denoiser: {sum(p.numel() for p in model.denoiser.parameters())/1e6:.1f}M | EMA: {ema_decay}")

    trainable = list(model.denoiser.parameters())
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.99),
                                 weight_decay=weight_decay)

    def lr_lambda(s):
        if s < warmup_steps: return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, epoch, best_psnr = 0, 0, 0.0
    nan_count = 0
    if resume_from and Path(resume_from).exists():
        step, epoch, best_psnr = _load_ckpt(resume_from, model, ema_model, optimizer, device, logger)
    else:
        logger.info("Fresh training from step 0")

    model.train()
    model.stage1.eval()
    loader_iter = infinite_loader(train_loader)
    for _ in range(step % len(train_loader)):
        next(loader_iter)

    running_r, running_n, running_steps = 0.0, 0.0, 0
    t0 = time.time()
    prog = _ProgressBar(total_steps, initial=step)

    try:
        while step < total_steps:
            lr_val = optimizer.param_groups[0]['lr']
            batch = next(loader_iter)
            ldct = batch['ldct'].to(device, non_blocking=True)
            hdct = batch['ndct'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(ldct, hdct, mode='train')

            loss = res_weight * out['loss_res'] + noise_weight * out['loss_noise']

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                logger.error(f"Step {step}: loss={loss.item()}, NaN count={nan_count}")
                if nan_count >= 3: raise RuntimeError("Training diverged (3 consecutive NaN)")
                continue
            else:
                nan_count = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            scheduler.step()

            if ema_model and step % ema_update_every == 0:
                ema_model.update()

            running_r += out['loss_res'].item()
            running_n += out['loss_noise'].item()
            running_steps += 1
            step += 1

            prog.step(L_res=out['loss_res'].item(), L_noise=out['loss_noise'].item(),
                      lr=lr_val, ep=epoch)

            if step % log_every == 0:
                avg_r = running_r / running_steps; avg_n = running_n / running_steps
                elapsed = time.time() - t0
                logger.info(f"Step {step:6d} | ep={epoch:2d} | L_res={avg_r:.4f} | L_noise={avg_n:.4f} | lr={lr_val:.2e} | {elapsed/running_steps:.2f}s/step")
                running_r = running_n = 0.0; running_steps = 0; t0 = time.time()

            if step % val_every == 0:
                psnr = _validate(model, ema_model, val_loader, device, val_batches, step, epoch, logger, model.diffusion)
                if psnr > best_psnr:
                    best_psnr = psnr
                    _save_ckpt(os.path.join(checkpoint_dir, 'stage2_best.pth'), step, epoch,
                               model, ema_model, optimizer, best_psnr,
                               {'image_size': image_size, 'timesteps': timesteps}, logger)

            if step % save_every == 0 and step > 0:
                _save_ckpt(os.path.join(checkpoint_dir, f'stage2_step_{step}.pth'), step,
                           epoch, model, ema_model, optimizer, best_psnr,
                           {'image_size': image_size, 'timesteps': timesteps}, logger)
                files = sorted(Path(checkpoint_dir).glob('stage2_step_*.pth'),
                               key=lambda f: int(f.stem.split('_')[-1]))
                for f in files[:-3]: f.unlink()

            if step % batches_per_epoch == 0:
                epoch += 1

    except KeyboardInterrupt:
        logger.info("Interrupted")

    finally:
        prog.close()
        if step > 0:
            _save_ckpt(os.path.join(checkpoint_dir, f'stage2_step_{step}_final.pth'), step,
                       epoch, model, ema_model, optimizer, best_psnr,
                       {'image_size': image_size, 'timesteps': timesteps}, logger)

    logger.info(f"Done | steps={step} | best_psnr={best_psnr:.1f} dB")
    return model


if __name__ == '__main__':
    import tempfile
    print("=" * 60)
    print("training/train_stage2.py — smoke test (5 dummy steps)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        m = train_stage2(checkpoint_dir=tmpdir, use_dummy_data=True, max_steps=5,
                         image_size=32, batch_size=1, num_workers=0, timesteps=50,
                         total_steps=5, warmup_steps=2, log_every=2, val_every=5,
                         save_every=100, ema_decay=0.0, sampling_timesteps=5)
    assert m is not None
    print("\n" + "=" * 60)
    print("Smoke test: PASSED")
    print("=" * 60)
