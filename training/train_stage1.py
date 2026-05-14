
"""
training/train_stage1.py

Stage 1 Training Loop — VM-UNet organ segmentation with BYOL.
=============================================================
Trains the VM-UNet teacher network (Stage1Model) to:
  1. Segment CT organs into 7 classes  (L_seg, always active)
  2. Learn noise-invariant features    (L_byol, active from epoch 5)

After training, the checkpoint is loaded by Stage 2 as a frozen
anatomy conditioning network.

Loss schedule
-------------
  epoch < byol_start_epoch : L = 1.0 * L_seg
  epoch ≥ byol_start_epoch : L = 1.0 * L_seg + 0.1 * L_byol

Optimiser & schedule
--------------------
  AdamW  lr=1e-4  weight_decay=0.01
  Linear warmup over warmup_steps (default 1000)
  Cosine decay from peak lr to 0 over remaining steps

Checkpoints
-----------
  checkpoint_dir/stage1_step_{N}.pth   — every 5000 steps
  checkpoint_dir/stage1_best.pth        — whenever val Dice improves

Reference: Architecture.pdf – Chapters 8 and 9
"""

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    """
    Returns a logger that writes to both console (stdout) and log file.
    Format: [2024-01-15 14:32:01] STEP 1000 | ...
    """
    logger = logging.getLogger('stage1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-run

    fmt = logging.Formatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — appends so resume adds to same log
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Scans checkpoint_dir for stage1_step_*.pth files.
    Returns path of the one with the highest step number, or None.
    Does NOT return stage1_best.pth (that has no step number).
    """
    ckpt_dir = Path(checkpoint_dir)
    pattern = 'stage1_step_*.pth'
    files = list(ckpt_dir.glob(pattern))
    if not files:
        return None
    # Extract step number from filename and return highest
    def _step(f):
        try:
            return int(f.stem.split('_')[-1])
        except (ValueError, IndexError):
            return -1
    valid_files = [f for f in files if _step(f) > 0]
    if not valid_files:
        return None
    return str(max(valid_files, key=_step))


def cleanup_old_checkpoints(checkpoint_dir: str, keep: int = 3, logger: Optional[logging.Logger] = None) -> None:
    """Delete step checkpoints older than the last `keep` ones."""
    files = sorted(
        Path(checkpoint_dir).glob('stage1_step_*.pth'),
        key=lambda f: int(f.stem.split('_')[-1]) if f.stem.split('_')[-1].isdigit() else -1
    )
    for f in files[:-keep]:
        try:
            f.unlink()
            if logger:
                logger.info(f"Deleted old checkpoint: {f.name}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to delete {f.name}: {e}")


def infinite_loader(dataloader):
    """Yields batches forever, reshuffling each epoch."""
    while True:
        for batch in dataloader:
            yield batch


def check_model_weights(model: nn.Module, step: int, logger: logging.Logger) -> bool:
    """
    Returns True if any parameter contains NaN or Inf.
    Call this every 100 steps to catch weight corruption early.
    """
    for name, param in model.named_parameters():
        if param is not None and not torch.isfinite(param).all():
            logger.error(
                f"Step {step} | NaN/Inf detected in weights: {name} "
                f"— training is corrupted. Stop and resume from checkpoint."
            )
            return True
    return False


def _log_train_line(msg: str) -> None:
    """Print without breaking tqdm progress bar when tqdm is installed."""
    if _tqdm is not None:
        _tqdm.write(msg)
    else:
        print(msg)


class _StepProgressBar:
    """Global-step tqdm bar + per-step postfix (instant losses). Falls back to no-op."""

    def __init__(self, total: int, initial: int = 0):
        self._pbar = None
        if _tqdm is not None:
            self._pbar = _tqdm(
                total=total,
                initial=initial,
                unit='step',
                desc='stage1',
                dynamic_ncols=True,
                mininterval=0.25,
                smoothing=0.05,
            )

    def step(
        self,
        *,
        l_seg: float,
        l_byol: float,
        lr: float,
        epoch: int,
        byol_active: bool,
    ) -> None:
        if self._pbar is None:
            return
        self._pbar.update(1)
        post = {
            'L_seg': f'{l_seg:.4f}',
            'lr': f'{lr:.1e}',
            'ep': str(epoch),
        }
        if byol_active:
            post['L_byol'] = f'{l_byol:.4f}'
        else:
            post['BYOL'] = 'off'
        self._pbar.set_postfix(post, refresh=False)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# ── project imports ───────────────────────────────────────────────────────────
# Running `python training/train_stage1.py` puts `training/` first on sys.path,
# not the repo root — ensure root is visible for `models`, `losses`, `datapy`.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models.stage1         import Stage1Model
from losses.stage1_losses  import SegmentationLoss, compute_dice_per_class
from models.byol           import get_ema_tau
from datapy.dataset        import (CTSliceDataset, DummyCTDataset,
                                   create_dataloaders)


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters (used when no YAML config is found)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Training duration
    'total_steps':        100_000,
    'warmup_steps':       1_000,

    # Data
    'batch_size':         4,
    'num_workers':        4,
    'image_size':         512,      # CTSliceDataset bilinear-resizes to H=W if set

    # Model
    'embed_dim':          96,
    'depths':             [2, 2, 2, 2],
    'num_classes':        7,
    'patch_size':         4,
    'd_state':            8,        # VSSD state dim (memory ∝ d_state)
    'drop_path_rate':     0.1,

    # Optimiser
    'lr':                 1e-4,
    'weight_decay':       0.01,

    # Loss weights
    'seg_weight':         1.0,
    'byol_weight':        0.1,
    'byol_start_epoch':   5,     # epoch at which BYOL loss is switched on

    # BYOL EMA
    'ema_tau_start':      0.996,
    'ema_tau_end':        1.0,

    # Logging / checkpointing
    'log_every':          100,
    'val_every':          1000,
    'save_every':         5000,
    'val_batches':        20,    # max val batches per validation run
    'label_smoothing':    0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_yaml_into_cfg(cfg: dict, user: dict) -> None:
    """
    Merge nested configs/stage1_config.yaml sections into the flat dict that
    train_stage1 expects (YAML uses groups: training, model, data, …).
    """
    if not user:
        return

    tr = user.get('training') or {}
    for yaml_k, flat_k in (
        ('total_steps',       'total_steps'),
        ('warmup_steps',      'warmup_steps'),
        ('batch_size',        'batch_size'),
        ('image_size',        'image_size'),
        ('learning_rate',     'lr'),
        ('weight_decay',      'weight_decay'),
        ('loss_seg_weight',   'seg_weight'),
        ('loss_byol_weight',  'byol_weight'),
        ('byol_start_epoch',  'byol_start_epoch'),
        ('label_smoothing',   'label_smoothing'),
    ):
        if yaml_k in tr:
            cfg[flat_k] = tr[yaml_k]

    mo = user.get('model') or {}
    if 'base_channels' in mo:
        cfg['embed_dim'] = mo['base_channels']
    if 'depths' in mo:
        cfg['depths'] = mo['depths']
    if 'num_classes' in mo:
        cfg['num_classes'] = mo['num_classes']
    if 'patch_size' in mo:
        cfg['patch_size'] = mo['patch_size']
    if 'd_state' in mo:
        cfg['d_state'] = mo['d_state']

    da = user.get('data') or {}
    if 'num_workers' in da:
        cfg['num_workers'] = da['num_workers']

    by = user.get('byol') or {}
    if 'ema_tau_start' in by:
        cfg['ema_tau_start'] = by['ema_tau_start']
    if 'ema_tau_end' in by:
        cfg['ema_tau_end'] = by['ema_tau_end']

    lg = user.get('logging') or {}
    if 'log_every_n_steps' in lg:
        cfg['log_every'] = lg['log_every_n_steps']
    if 'eval_every_n_steps' in lg:
        cfg['val_every'] = lg['eval_every_n_steps']
    if 'save_every_n_steps' in lg:
        cfg['save_every'] = lg['save_every_n_steps']


def _load_config(config_path: Optional[str]) -> dict:
    """
    Load YAML config and merge with DEFAULT_CONFIG (defaults fill missing keys).
    Flat nested YAML sections into train_stage1's flat keys.
    """
    cfg = dict(DEFAULT_CONFIG)

    path = Path(config_path) if config_path else None
    if path and path.exists():
        import yaml  # optional dependency
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        _flatten_yaml_into_cfg(cfg, user)
        print(f"[config] Loaded {path}")
    elif config_path:
        print(f"[config] '{config_path}' not found — using built-in defaults.")

    return cfg


def _get_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    """
    Linear warmup then cosine decay schedule.

      0 … warmup_steps        : lr  linearly rises  0 → peak_lr
      warmup_steps … total    : lr  cosine decays   peak_lr → 0
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def _save_checkpoint(
    path:        str,
    step:        int,
    epoch:       int,
    model:       nn.Module,
    optimizer:   torch.optim.Optimizer,
    scheduler:   Optional[torch.optim.lr_scheduler.LambdaLR],
    best_dice:   float,
    config:      dict,
    logger:      Optional[logging.Logger] = None,
    byol_ema_tau: float = 1.0,
) -> None:
    """Save training state to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'step':                step,
            'epoch':               epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_val_dice':       best_dice,
            'config':              config,
            'byol_ema_tau':        byol_ema_tau,
        },
        path,
    )
    if logger:
        logger.info(f"Checkpoint saved → {Path(path).name} (dice={best_dice:.4f})")
    else:
        _log_train_line(f"  [ckpt] Saved → {path}")


def _load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device:    torch.device,
    logger:    Optional[logging.Logger] = None,
):
    """
    Load a checkpoint in-place.  Returns (step, epoch, best_val_dice, byol_ema_tau).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    step       = ckpt.get('step',          0)
    epoch      = ckpt.get('epoch',         0)
    best_dice  = ckpt.get('best_val_dice', 0.0)
    byol_tau   = ckpt.get('byol_ema_tau',  1.0)
    if logger:
        logger.info(f"Resuming from step {step} | best_val_dice={best_dice:.4f}")
    else:
        print(f"  [ckpt] Resumed from {path}  (step={step}, epoch={epoch})")
    return step, epoch, best_dice, byol_tau


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    model:        Stage1Model,
    val_loader:   DataLoader,
    device:       torch.device,
    max_batches:  int,
    step:         int,
    epoch:        int,
    log_fn:       Callable[[str], None] = _log_train_line,
    logger:       Optional[logging.Logger] = None,
) -> float:
    """
    Run segmentation inference on up to *max_batches* validation batches.
    Returns mean Dice across all 7 classes.
    """
    model.eval()

    all_probs   = []
    all_targets = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        ndct = batch['ndct'].to(device)   # [B, 1, H, W]
        mask = batch['mask'].to(device)   # [B, H, W]

        out  = model(ndct, return_byol=False)
        S    = out['S']                   # [B, 7, H, W]

        all_probs.append(S.cpu())
        all_targets.append(mask.cpu())

    if not all_probs:
        return 0.0

    probs   = torch.cat(all_probs,   dim=0)   # [N, 7, H, W]
    targets = torch.cat(all_targets, dim=0)   # [N, H, W]

    dice_scores = compute_dice_per_class(probs, targets)

    msg = (
        f"Step {step:6d} | Val Dice → "
        f"liver={dice_scores['liver_spleen']:.4f} "
        f"kidney={dice_scores['kidney']:.4f} "
        f"lung={dice_scores['lung']:.4f} "
        f"mean={dice_scores['mean']:.4f}"
    )
    
    if logger:
        logger.info(msg)
    else:
        log_fn(
            f"  [val]  step={step:6d}  epoch={epoch:3d}  "
            f"mean_dice={dice_scores['mean']:.4f}  "
            f"liver={dice_scores['liver_spleen']:.4f}  "
            f"kidney={dice_scores['kidney']:.4f}  "
            f"lung={dice_scores['lung']:.4f}"
        )

    model.train()
    return dice_scores['mean']


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_stage1(
    config_path:    Optional[str] = 'configs/stage1_config.yaml',
    data_root:      str  = '/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root:     str  = '/home/teaching/Music/Nigam_51/Project_51/data/masks',
    checkpoint_dir: str  = '/home/teaching/Music/Nigam_51/Project_51/checkpoints/stage1',
    resume_from:    Optional[str]  = None,
    use_dummy_data: bool = False,
    max_steps:      Optional[int]  = None,
):
    """
    Complete Stage 1 training loop.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.  Missing keys are filled from DEFAULT_CONFIG.
    data_root : str
        Root directory that contains patient folders (C002/, C004/, …).
    masks_root : str
        Directory containing TotalSegmentator organ masks.
    checkpoint_dir : str
        Directory where checkpoints are written.
    resume_from : str or None
        Path to a Stage 1 checkpoint to resume from.
    use_dummy_data : bool
        If True, use DummyCTDataset (no files required).  Useful for smoke tests.
    max_steps : int or None
        Stop training after this many gradient steps.  None = run to total_steps.
    """

    # ── Configuration ─────────────────────────────────────────────────────
    cfg = _load_config(config_path)

    total_steps       = cfg['total_steps']
    warmup_steps      = cfg['warmup_steps']
    batch_size        = cfg['batch_size']
    num_workers       = cfg['num_workers']
    peak_lr           = cfg['lr']
    weight_decay      = cfg['weight_decay']
    seg_weight        = cfg['seg_weight']
    byol_weight       = cfg['byol_weight']
    byol_start_epoch  = cfg['byol_start_epoch']
    ema_tau_start     = cfg['ema_tau_start']
    ema_tau_end       = cfg['ema_tau_end']
    log_every         = cfg['log_every']
    val_every         = cfg['val_every']
    save_every        = 1000  # Changed from cfg['save_every']; now hardcoded to 1000
    val_batches       = cfg['val_batches']
    label_smoothing   = cfg['label_smoothing']
    embed_dim         = cfg['embed_dim']
    depths            = cfg['depths']
    num_classes       = cfg['num_classes']
    patch_size        = cfg['patch_size']
    d_state           = cfg['d_state']
    drop_path_rate    = cfg['drop_path_rate']
    image_size        = cfg['image_size']

    # Honour the hard cap from the caller (e.g. smoke test)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Logging setup ─────────────────────────────────────────────────────
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(checkpoint_dir) / 'training.log'
    logger = setup_logger(str(log_path))

    # Log training start
    logger.info(
        f"Training start | device={device} | total_steps={total_steps} | "
        f"batch_size={batch_size} | lr={peak_lr:.2e} | warmup_steps={warmup_steps}"
    )

    # ── Data ──────────────────────────────────────────────────────────────
    if use_dummy_data:
        image_size = cfg.get('image_size', 64)   # small images for fast tests
        train_ds   = DummyCTDataset(length=200,   image_size=image_size)
        val_ds     = DummyCTDataset(length=40,    image_size=image_size)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        logger.info(f"Using DummyCTDataset | image_size={image_size} | train_batches={len(train_loader)} | val_batches={len(val_loader)}")
    else:
        loaders      = create_dataloaders(
            data_root   = data_root,
            masks_root  = masks_root,
            batch_size  = batch_size,
            num_workers = num_workers,
            image_size  = image_size,
        )
        train_loader = loaders['train']
        val_loader   = loaders['val']
        logger.info(f"Loaded real data | train_patients={len(train_loader.dataset)} | val_patients={len(val_loader.dataset)}")

    batches_per_epoch = len(train_loader)
    logger.info(f"Data ready | image_size={image_size} | batches_per_epoch={batches_per_epoch} | log_every={log_every} | val_every={val_every}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = Stage1Model(
        num_classes      = num_classes,
        embed_dim        = embed_dim,
        depths           = depths,
        patch_size       = patch_size,
        d_state          = d_state,
        drop_path_rate   = drop_path_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage1Model initialized | params={n_params/1e6:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────────
    seg_criterion = SegmentationLoss(
        num_classes     = num_classes,
        label_smoothing = label_smoothing,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Exclude BYOL target projector from gradient updates
    # (it has requires_grad=False, so AdamW won't touch it anyway, but
    #  being explicit avoids accidental weight-decay on frozen params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr           = peak_lr,
        weight_decay = weight_decay,
    )

    # LR scheduler (lambda function)
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Optimizer initialized | lr={peak_lr:.2e} | weight_decay={weight_decay}")

    # ── Optional resume ───────────────────────────────────────────────────
    step           = 0
    epoch          = 0
    best_dice      = 0.0
    current_tau    = ema_tau_start

    # Auto-detect resume if no explicit resume_from given
    if resume_from is None:
        resume_from = find_latest_checkpoint(checkpoint_dir)
        if resume_from:
            logger.info(f"Auto-detected checkpoint: {resume_from}")

    if resume_from and Path(resume_from).exists():
        step, epoch, best_dice, current_tau = _load_checkpoint(
            resume_from, model, optimizer, scheduler, device, logger=logger
        )
        # Update step to next step after resume
        step += 1
    else:
        logger.info(f"Starting fresh training from step 0")

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()

    # Infinite loader pattern for proper resume
    loader_iter = infinite_loader(train_loader)

    # Skip batches already processed (fast-forward after resume)
    start_step = step
    if start_step > 0:
        logger.info(f"Fast-forwarding dataloader by {start_step % len(train_loader)} batches...")
        for _ in range(start_step % len(train_loader)):
            next(loader_iter)
        logger.info("Dataloader ready.")

    # Running stats for periodic logging
    running_seg_loss  = 0.0
    running_byol_loss = 0.0
    running_steps     = 0
    t0                = time.time()
    nan_count         = 0  # Track consecutive NaN losses

    logger.info(f"Training start | BYOL active from epoch {byol_start_epoch}")

    stop_training = False
    prog = _StepProgressBar(total_steps, initial=step)

    try:
        while not stop_training:
            # Step is incremented at the END of each iteration, so we use it
            # to check termination and checkpoint save conditions

            # ── LR schedule ───────────────────────────────────────────
            lr = _get_lr(step, total_steps, warmup_steps, peak_lr)
            _set_lr(optimizer, lr)

            # ── Decide whether BYOL is active ─────────────────────────
            byol_active = (epoch >= byol_start_epoch)

            # ── Get next batch ─────────────────────────────────────────
            batch = next(loader_iter)

            # ── Move data to device ───────────────────────────────────
            # Stage 1 trains on the NDCT (clean / HDCT) images.
            ndct = batch['ndct'].to(device, non_blocking=True)  # [B, 1, H, W]
            mask = batch['mask'].to(device, non_blocking=True)   # [B, H, W]

            # ── Forward pass ──────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)

            out = model(ndct, return_byol=byol_active)

            # ── Segmentation loss ─────────────────────────────────────
            l_seg = seg_criterion(out['logits'], mask)
            loss  = seg_weight * l_seg

            # ── BYOL loss (phase 2 only) ──────────────────────────────
            l_byol_val = 0.0
            if byol_active and out['byol_loss'] is not None:
                l_byol     = out['byol_loss']
                loss       = loss + byol_weight * l_byol
                l_byol_val = l_byol.item()

            # ── NaN guard BEFORE backward ──────────────────────────────
            if not torch.isfinite(loss):
                logger.warning(
                    f"Step {step} | Non-finite loss={loss.item():.6f} "
                    f"— skipping batch, zeroing gradients"
                )
                optimizer.zero_grad()
                nan_count += 1
                if nan_count >= 3:
                    logger.error(
                        "3 consecutive NaN losses — weights are corrupted. "
                        "Stopping training. Resume from last checkpoint."
                    )
                    raise RuntimeError("NaN loss: weights corrupted.")
                continue
            nan_count = 0  # reset on healthy step
            # ── end NaN guard ──────────────────────────────────────────

            # ── Backward ──────────────────────────────────────────────
            loss.backward()

            # Clip gradients — prevents exploding gradients from bad batches
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # Log if grad norm is very high (early warning of instability)
            if grad_norm > 10.0:
                logger.warning(
                    f"Step {step} | High grad norm={grad_norm:.2f} — clipped to 1.0"
                )

            optimizer.step()
            scheduler.step()

            # ── EMA update of BYOL target projector ───────────────────
            tau = get_ema_tau(step, total_steps, ema_tau_start, ema_tau_end)
            current_tau = tau
            model.byol.update_target_projector(tau)

            # ── Accumulate stats ──────────────────────────────────────
            running_seg_loss  += l_seg.item()
            running_byol_loss += l_byol_val
            running_steps     += 1

            step += 1   # step is 1-indexed after this point

            prog.step(
                l_seg=l_seg.item(),
                l_byol=float(l_byol_val),
                lr=lr,
                epoch=epoch,
                byol_active=byol_active,
            )

            # ── Periodic logging ──────────────────────────────────────
            if step % log_every == 0:
                avg_seg  = running_seg_loss  / running_steps
                avg_byol = running_byol_loss / running_steps
                elapsed  = time.time() - t0
                sec_per_step = elapsed / max(running_steps, 1)

                byol_str = (
                    f" | L_byol={avg_byol:.4f}"
                    if byol_active else ""
                )
                logger.info(
                    f"Step {step:6d} | ep={epoch:3d} | L_seg={avg_seg:.4f}{byol_str} | "
                    f"lr={lr:.2e} | {sec_per_step:.2f}s/step"
                )

                # Check for weight corruption every 100 steps
                if check_model_weights(model, step, logger):
                    raise RuntimeError(
                        f"Corrupted weights at step {step}. "
                        f"Resume from last checkpoint."
                    )

                # Reset running stats
                running_seg_loss  = 0.0
                running_byol_loss = 0.0
                running_steps     = 0
                t0                = time.time()

            # ── Validation ────────────────────────────────────────────
            if step % val_every == 0:
                mean_dice = _validate(
                    model, val_loader, device, val_batches, step, epoch,
                    logger=logger
                )

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    _save_checkpoint(
                        path      = os.path.join(
                            checkpoint_dir, 'stage1_best.pth'
                        ),
                        step      = step,
                        epoch     = epoch,
                        model     = model,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        best_dice = best_dice,
                        config    = cfg,
                        logger    = logger,
                        byol_ema_tau = current_tau,
                    )
                    logger.info(f"Step {step:6d} | ★ New best val Dice = {best_dice:.4f}")

                model.train()

            # ── Periodic checkpoint (every 1000 steps) ────────────────
            if step % save_every == 0 and step > 0:
                _save_checkpoint(
                    path      = os.path.join(
                        checkpoint_dir, f'stage1_step_{step}.pth'
                    ),
                    step      = step,
                    epoch     = epoch,
                    model     = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    best_dice = best_dice,
                    config    = cfg,
                    logger    = logger,
                    byol_ema_tau = current_tau,
                )
                # Cleanup old checkpoints; keep only last 3
                cleanup_old_checkpoints(checkpoint_dir, keep=3, logger=logger)

            # ── Check epoch boundary ──────────────────────────────────
            if step % batches_per_epoch == 0:
                epoch += 1

            # ── Termination guard ─────────────────────────────────────
            if step >= total_steps:
                stop_training = True
                break

    finally:
        prog.close()

    # ── Final checkpoint ──────────────────────────────────────────────────
    if step > 0:  # Only save if we actually trained
        _save_checkpoint(
            path      = os.path.join(checkpoint_dir, f'stage1_step_{step}_final.pth'),
            step      = step,
            epoch     = epoch,
            model     = model,
            optimizer = optimizer,
            scheduler = scheduler,
            best_dice = best_dice,
            config    = cfg,
            logger    = logger,
            byol_ema_tau = current_tau,
        )

    logger.info(
        f"Training complete | total_steps={step} | total_epochs={epoch} | "
        f"best_val_dice={best_dice:.4f}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST — smoke test (10 gradient steps on dummy data)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile
    import sys

    print("=" * 60)
    print("training/train_stage1.py — smoke test")
    print("=" * 60)
    print("Running 10 gradient steps on dummy data …\n")

    # Use a tiny image size so the smoke test runs in < 30 s on any GPU
    # We patch DEFAULT_CONFIG temporarily to shrink the model and data.
    _SMOKE_CONFIG_OVERRIDES = {
        'total_steps':     10,
        'warmup_steps':    5,
        'batch_size':      2,
        'num_workers':     0,
        'image_size':      64,      # DummyCTDataset will use this
        'embed_dim':       96,
        'depths':          [2, 2, 2, 2],
        'num_classes':     7,
        'lr':              1e-4,
        'weight_decay':    0.01,
        'seg_weight':      1.0,
        'byol_weight':     0.1,
        'byol_start_epoch': 0,      # activate BYOL immediately for smoke test
        'ema_tau_start':   0.996,
        'ema_tau_end':     1.0,
        'log_every':       2,
        'val_every':       5,
        'save_every':      100,     # no checkpoint during 10-step test
        'val_batches':     2,
        'label_smoothing': 0.1,
    }

    # Patch defaults; smoke run passes config_path=None so YAML is not merged on top.
    DEFAULT_CONFIG.update(_SMOKE_CONFIG_OVERRIDES)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_stage1(
                config_path    = None,   # use patched DEFAULT_CONFIG only
                checkpoint_dir = tmpdir,
                use_dummy_data = True,
                max_steps      = 10,
            )

        # Verify the returned model is a valid Stage1Model
        assert isinstance(model, Stage1Model), \
            f"Expected Stage1Model, got {type(model)}"

        # Quick inference check on the trained model (match CUDA vs CPU)
        model.eval()
        _dev = next(model.parameters()).device
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, 64, 64, device=_dev)
            out = model(dummy_x, return_byol=False)
        assert out['S'].shape   == (1, 7, 64, 64)
        assert out['e_a'].shape == (1, 7, 96)
        assert torch.isfinite(out['S']).all()

        print("\n" + "=" * 60)
        print("Smoke test: PASSED")
        print("=" * 60)
        sys.exit(0)

    except Exception as exc:
        import traceback
        print("\n" + "=" * 60)
        print("Smoke test: FAILED")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)