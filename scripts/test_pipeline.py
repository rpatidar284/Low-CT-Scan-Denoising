"""
scripts/test_pipeline.py

Full inference pipeline: Stage 1 + Stage 2 denoising.
Usage:
  python scripts/test_pipeline.py --patient C002
  python scripts/test_pipeline.py --patient L004 --output results/test_run
  python scripts/test_pipeline.py --patient C002 --save-every 10 --ddim-steps 50
"""

import argparse, os, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models.stage1 import load_stage1_frozen
from models.stage2 import Stage2Model
from models.vssd_denoiser import _build_S_scales
from datapy.dataset import CTSliceDataset, load_slice
from utils.metrics import compute_psnr, compute_ssim, compute_rmse

CLASS_NAMES = ['background', 'liver_spleen', 'kidney', 'vessel', 'lung', 'bone', 'soft_tissue']
COLORS = ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan']


def test_pipeline(patient_id, stage1_path, stage2_path, output_dir, ddim_steps=50,
                  save_every=1, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Full Pipeline Test — Patient: {patient_id}")
    print(f"{'='*60}")

    # ── 1. Load models ──
    print("\n[1/5] Loading models...")
    stage1 = load_stage1_frozen(stage1_path, device=device)
    stage1.eval()
    print(f"  Stage 1 loaded (frozen)")

    denoiser_kwargs = {'image_size': 512}
    diffusion_kwargs = {'timesteps': 1000, 'sampling_timesteps': ddim_steps,
                        'loss_type': 'l1', 'sum_scale': 0.01, 'ddim_eta': 0.0}
    stage2 = Stage2Model(stage1_checkpoint=None, denoiser_kwargs=denoiser_kwargs,
                         diffusion_kwargs=diffusion_kwargs, image_size=512).to(device)
    ckpt = torch.load(stage2_path, map_location=device)
    stage2.load_state_dict(ckpt['model_state_dict'])
    stage2.stage1 = stage1  # Use already-loaded Stage 1
    stage2.eval()
    print(f"  Stage 2 loaded | trained {ckpt['step']} steps | best PSNR={ckpt['best_psnr']:.1f} dB")

    # ── 2. Load patient slices ──
    print(f"\n[2/5] Loading patient {patient_id}...")
    data_root = str(_root / 'data')
    patient_dir = Path(data_root) / patient_id
    hdct_dir = patient_dir / 'HDCT'
    ldct_dir = patient_dir / 'LDCT'

    if not hdct_dir.exists() or not ldct_dir.exists():
        raise FileNotFoundError(f"Patient {patient_id} not found in {data_root}")

    from datapy.dataset import get_sorted_slice_files
    hdct_files = get_sorted_slice_files(str(hdct_dir))
    ldct_files = get_sorted_slice_files(str(ldct_dir))

    hdct_by_stem = {Path(f).stem: f for f in hdct_files}
    ldct_by_stem = {Path(f).stem: f for f in ldct_files}
    common = sorted(hdct_by_stem.keys() & ldct_by_stem.keys(),
                    key=lambda s: int(''.join(c for c in s if c.isdigit())) if any(c.isdigit() for c in s) else s)

    print(f"  HDCT: {len(hdct_files)} slices, LDCT: {len(ldct_files)} slices, Paired: {len(common)}")

    # ── 3. Run inference ──
    print(f"\n[3/5] Running inference ({len(common)} slices, DDIM={ddim_steps} steps)...")
    psnrs, ssims, rmses = [], [], []
    t0 = time.time()
    all_hdct_files = []
    all_ldct_files = []
    all_denoised = []

    for i, stem in enumerate(common):
        hdct_arr = load_slice(hdct_by_stem[stem])
        ldct_arr = load_slice(ldct_by_stem[stem])

        # Normalize to [0,1]
        HU_MIN, HU_MAX = -1000.0, 3000.0
        hdct_t = torch.from_numpy(np.clip(hdct_arr, HU_MIN, HU_MAX)).float()
        ldct_t = torch.from_numpy(np.clip(ldct_arr, HU_MIN, HU_MAX)).float()
        hdct_t = ((hdct_t - HU_MIN) / (HU_MAX - HU_MIN)).unsqueeze(0).unsqueeze(0)
        ldct_t = ((ldct_t - HU_MIN) / (HU_MAX - HU_MIN)).unsqueeze(0).unsqueeze(0)

        # Resize to 512×512
        H, W = hdct_t.shape[-2], hdct_t.shape[-1]
        if H != 512 or W != 512:
            hdct_t = F.interpolate(hdct_t, size=(512, 512), mode='bilinear', align_corners=False)
            ldct_t = F.interpolate(ldct_t, size=(512, 512), mode='bilinear', align_corners=False)

        hdct_t = hdct_t.to(device)
        ldct_t = ldct_t.to(device)

        # Stage 1 → anatomy conditioning from LDCT
        with torch.no_grad():
            S, e_a = stage1.get_anatomy_conditioning(ldct_t)

        # Stage 2 → denoise
        with torch.no_grad():
            out = stage2(ldct_t, mode='inference')
            x_denoised = out['x_denoised']

        # Metrics
        psnrs.append(compute_psnr(x_denoised, hdct_t))
        ssims.append(compute_ssim(x_denoised, hdct_t))
        rmses.append(compute_rmse(x_denoised, hdct_t))

        all_hdct_files.append(hdct_by_stem[stem])
        all_ldct_files.append(ldct_by_stem[stem])
        all_denoised.append(x_denoised.cpu().squeeze().numpy())

        if (i + 1) % save_every == 0 or i == 0:
            elapsed = time.time() - t0
            per_slice = elapsed / (i + 1)
            remaining = per_slice * (len(common) - i - 1)
            print(f"  [{i+1}/{len(common)}] PSNR={psnrs[-1]:.1f}dB SSIM={ssims[-1]:.4f} | {per_slice:.1f}s/slice | {remaining/60:.0f}min remaining")

    total_time = time.time() - t0
    print(f"\n  Total: {total_time:.0f}s ({total_time/len(common):.1f}s/slice)")

    # ── 4. Save outputs ──
    print(f"\n[4/5] Saving outputs to {output_dir}/...")
    npy_dir = os.path.join(output_dir, 'denoised')
    os.makedirs(npy_dir, exist_ok=True)

    for i, stem in enumerate(common):
        np.save(os.path.join(npy_dir, f'{stem}.npy'), all_denoised[i])

    print(f"  Saved {len(common)} denoised slices to {npy_dir}/")

    # ── 5. Summary + Visualization ──
    print(f"\n[5/5] Generating summary...")
    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)
    mean_rmse = np.mean(rmses)

    print(f"\n{'='*60}")
    print(f"RESULTS — Patient {patient_id}")
    print(f"{'='*60}")
    print(f"  Slices processed : {len(common)}")
    print(f"  PSNR (mean±std)  : {mean_psnr:.1f} ± {np.std(psnrs):.1f} dB")
    print(f"  SSIM (mean±std)  : {mean_ssim:.4f} ± {np.std(ssims):.4f}")
    print(f"  RMSE (mean±std)  : {mean_rmse:.4f} ± {np.std(rmses):.4f}")
    print(f"  Inference time   : {total_time:.0f}s ({total_time/len(common):.1f}s/slice)")
    print(f"  DDIM steps       : {ddim_steps}")
    print(f"  Output directory : {output_dir}/")
    print(f"{'='*60}")

    # Save summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Patient: {patient_id}\n")
        f.write(f"Slices: {len(common)}\n")
        f.write(f"PSNR: {mean_psnr:.1f} ± {np.std(psnrs):.1f} dB\n")
        f.write(f"SSIM: {mean_ssim:.4f} ± {np.std(ssims):.4f}\n")
        f.write(f"RMSE: {mean_rmse:.4f} ± {np.std(rmses):.4f}\n")
        f.write(f"DDIM steps: {ddim_steps}\n")
        f.write(f"Inference time: {total_time:.0f}s\n")

    # Visualization — best and worst 3 slices
    best_idx = sorted(range(len(common)), key=lambda i: psnrs[i], reverse=True)
    n_viz = min(3, len(common))

    fig, axes = plt.subplots(n_viz, 3, figsize=(12, 4 * n_viz))
    if n_viz == 1: axes = axes.reshape(1, -1)

    for row, idx in enumerate(best_idx[:n_viz]):
        ldct_img = np.clip(np.load(all_ldct_files[idx]) if all_ldct_files[idx].endswith('.npy')
                           else load_slice(all_ldct_files[idx]), -160, 240)
        hdct_img = np.clip(np.load(all_hdct_files[idx]) if all_hdct_files[idx].endswith('.npy')
                           else load_slice(all_hdct_files[idx]), -160, 240)
        denoised_img = np.clip(all_denoised[idx] * 4000 - 1000, -160, 240)

        for col, (img, title) in enumerate(zip(
            [ldct_img, denoised_img, hdct_img],
            [f'LDCT (noisy)', f'Denoised\nPSNR={psnrs[idx]:.1f}dB', f'HDCT (clean)']
        )):
            axes[row, col].imshow(img, cmap='gray', vmin=-160, vmax=240)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{patient_id}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {output_dir}/{patient_id}_comparison.png")

    return {'psnr': mean_psnr, 'ssim': mean_ssim, 'rmse': mean_rmse, 'n_slices': len(common)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Stage 1+2 denoising pipeline')
    parser.add_argument('--patient', type=str, required=True, help='Patient ID (e.g. C002, L004)')
    parser.add_argument('--stage1', type=str, default='outputs/stage1/stage1_best.pth')
    parser.add_argument('--stage2', type=str, default='outputs/stage2/stage2_best.pth')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--ddim-steps', type=int, default=50, help='DDIM sampling steps (default 50)')
    parser.add_argument('--save-every', type=int, default=20, help='Save progress every N slices')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'results/pipeline_{args.patient}'

    test_pipeline(args.patient, args.stage1, args.stage2, args.output,
                  ddim_steps=args.ddim_steps, save_every=args.save_every, device=args.device)
