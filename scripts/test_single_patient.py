"""
Test Stage 1 segmentation on a single patient at 256px.
Saves prediction visualizations so the user can judge segmentation quality
before deciding on image_size / training duration tradeoffs.
"""
import os, sys, time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from models.stage1 import Stage1Model
from losses.stage1_losses import SegmentationLoss, compute_dice_per_class
from datapy.dataset import CTSliceDataset

# --- Config ---
PATIENT = "C002"
IMAGE_SIZE = 256
BATCH_SIZE = 4
TOTAL_STEPS = 1000
VAL_EVERY = 200
SAVE_EVERY = 200
OUTPUT_DIR = "/home/teaching/Music/Nigam_51/Project_51/outputs/stage1_test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["background", "liver_spleen", "kidney", "vessel", "lung", "bone", "soft_tissue"]
CLASS_COLORS = [
    [0, 0, 0],        # 0: background - black
    [255, 0, 0],       # 1: liver/spleen - red
    [0, 255, 0],       # 2: kidney - green
    [0, 0, 255],       # 3: vessel - blue
    [255, 255, 0],     # 4: lung - yellow
    [255, 0, 255],     # 5: bone - magenta
    [0, 255, 255],     # 6: soft_tissue - cyan
]


def mask_to_rgb(mask, colors):
    """Convert [H,W] int mask to [H,W,3] uint8 RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(colors):
        rgb[mask == c] = color
    return rgb


def save_overlay(ct_slice, pred_mask, gt_mask, step, slice_idx):
    """Save side-by-side: CT | Ground Truth | Prediction."""
    ct = (ct_slice * 255).astype(np.uint8)
    ct_rgb = np.stack([ct, ct, ct], axis=-1)

    gt_rgb = mask_to_rgb(gt_mask, CLASS_COLORS)
    pred_rgb = mask_to_rgb(pred_mask, CLASS_COLORS)

    # Blend overlay: 50% CT + 50% mask
    gt_overlay = (0.5 * ct_rgb + 0.5 * gt_rgb).astype(np.uint8)
    pred_overlay = (0.5 * ct_rgb + 0.5 * pred_rgb).astype(np.uint8)

    # Horizontal stack: CT | GT Overlay | Pred Overlay
    row = np.concatenate([ct_rgb, gt_overlay, pred_overlay], axis=1)

    out_path = Path(OUTPUT_DIR) / f"step{step:06d}_slice{slice_idx:03d}.png"
    Image.fromarray(row).save(str(out_path))
    return str(out_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Patient: {PATIENT}, Image size: {IMAGE_SIZE}, Steps: {TOTAL_STEPS}")
    print(f"Output: {OUTPUT_DIR}")

    # --- Data: single patient ---
    dataset = CTSliceDataset(
        data_root=str(_root / "data"),
        masks_root=str(_root / "data" / "masks"),
        split="train",
        augment=True,
        target_size=IMAGE_SIZE,
    )
    # Filter to only our patient
    dataset.samples = [(p, s, h, l) for p, s, h, l in dataset.samples if p == PATIENT]
    dataset.patients = [PATIENT]

    if len(dataset) == 0:
        # Try to find the patient in val split
        print(f"{PATIENT} not in train split, checking all patients...")
        all_dataset = CTSliceDataset(
            data_root=str(_root / "data"),
            masks_root=str(_root / "data" / "masks"),
            split="train",
            augment=False,
            target_size=IMAGE_SIZE,
        )
        # Override to include all
        all_patients = sorted([p.name for p in (Path(_root / "data")).iterdir()
                               if p.is_dir() and (p / "HDCT").exists() and (p / "LDCT").exists()
                               and not p.name.startswith('.') and p.name != 'masks'])
        if PATIENT in all_patients:
            dataset.patients = [PATIENT]
            hdct_dir = str(_root / "data" / PATIENT / "HDCT")
            ldct_dir = str(_root / "data" / PATIENT / "LDCT")
            from datapy.dataset import get_sorted_slice_files
            hdct_files = get_sorted_slice_files(hdct_dir)
            ldct_files = get_sorted_slice_files(ldct_dir)

            hdct_by_stem = {Path(f).stem: f for f in hdct_files}
            ldct_by_stem = {Path(f).stem: f for f in ldct_files}
            common = sorted(hdct_by_stem.keys() & ldct_by_stem.keys(),
                          key=lambda s: int(''.join(filter(str.isdigit, s))) if any(c.isdigit() for c in s) else 0)

            dataset.samples = [(PATIENT, i, hdct_by_stem[s], ldct_by_stem[s]) for i, s in enumerate(common)]
            print(f"  Found {len(dataset)} slices for {PATIENT}")
        else:
            raise RuntimeError(f"Patient {PATIENT} not found!")

    print(f"Total slices: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # --- Model ---
    model = Stage1Model(
        num_classes=7, embed_dim=96, depths=[2, 2, 2, 2],
        patch_size=4, d_state=8, drop_path_rate=0.1,
    ).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    seg_loss_fn = SegmentationLoss(num_classes=7, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / 100) * max(0.0, 0.5 * (1.0 + np.cos(np.pi * (s - 100) / max(TOTAL_STEPS - 100, 1))))
    )

    # --- Training loop ---
    model.train()
    loader_iter = iter(loader)
    t0 = time.time()
    best_dice = 0.0
    losses = []

    for step in range(1, TOTAL_STEPS + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        ndct = batch['ndct'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        out = model(ndct, return_byol=False)

        l_seg = seg_loss_fn(out['logits'], mask)
        l_seg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(l_seg.item())

        if step % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            elapsed = time.time() - t0
            print(f"Step {step:5d}/{TOTAL_STEPS} | Loss: {avg_loss:.4f} | {elapsed:.0f}s elapsed")

        if step % VAL_EVERY == 0:
            model.eval()
            all_probs, all_targets = [], []
            # Use first 20 slices for validation
            with torch.no_grad():
                val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                for i, b in enumerate(val_loader):
                    if i >= 5:  # 5 batches = ~20 slices
                        break
                    x_val = b['ndct'].to(DEVICE)
                    m_val = b['mask'].to(DEVICE)
                    pred = model(x_val, return_byol=False)
                    all_probs.append(pred['S'].cpu())
                    all_targets.append(m_val.cpu())

            if all_probs:
                probs = torch.cat(all_probs, dim=0)
                targets = torch.cat(all_targets, dim=0)
                dice = compute_dice_per_class(probs, targets)
                print(f"  Val Dice -> mean={dice['mean']:.4f} "
                      f"liver={dice['liver_spleen']:.4f} "
                      f"kidney={dice['kidney']:.4f} "
                      f"lung={dice['lung']:.4f} "
                      f"bone={dice['bone']:.4f}")

                if dice['mean'] > best_dice:
                    best_dice = dice['mean']
                    torch.save({'model_state_dict': model.state_dict(), 'step': step, 'dice': best_dice},
                               str(Path(OUTPUT_DIR) / "best_model.pth"))

            model.train()

        if step % SAVE_EVERY == 0:
            # Save visualizations for first 4 slices
            model.eval()
            with torch.no_grad():
                for i in range(min(4, len(dataset))):
                    sample = dataset[i]
                    x_vis = sample['ndct'].unsqueeze(0).to(DEVICE)
                    gt = sample['mask'].numpy()
                    pred = model(x_vis, return_byol=False)
                    pred_mask = pred['S'][0].argmax(dim=0).cpu().numpy()
                    ct_np = sample['ndct'][0].numpy()
                    path = save_overlay(ct_np, pred_mask, gt, step, i)
                print(f"  Saved visualizations for step {step}")

            # Save checkpoint
            torch.save({'model_state_dict': model.state_dict(), 'step': step, 'loss': avg_loss},
                       str(Path(OUTPUT_DIR) / f"checkpoint_step{step}.pth"))
            model.train()

    # --- Final: save detailed report ---
    elapsed_hr = (time.time() - t0) / 3600
    steps_per_sec = TOTAL_STEPS / (time.time() - t0)
    est_30k_hours = 30000 / (steps_per_sec * 3600) if steps_per_sec > 0 else 0

    print(f"\nTest complete: {TOTAL_STEPS} steps in {elapsed_hr:.1f}h")
    print(f"Speed: {steps_per_sec:.2f} steps/sec")
    print(f"Estimated 30k steps at 256px: {est_30k_hours:.1f} hours")
    print(f"Best val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    # Write summary
    with open(Path(OUTPUT_DIR) / "summary.txt", "w") as f:
        f.write(f"Single patient test: {PATIENT}\n")
        f.write(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"Training steps: {TOTAL_STEPS}\n")
        f.write(f"Elapsed: {elapsed_hr:.1f} hours\n")
        f.write(f"Speed: {steps_per_sec:.2f} steps/sec\n")
        f.write(f"Estimated 30k steps at 256px: {est_30k_hours:.1f} hours\n")
        f.write(f"Best val Dice: {best_dice:.4f}\n")


if __name__ == "__main__":
    main()
