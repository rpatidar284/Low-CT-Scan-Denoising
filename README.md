# Anatomy-aware LDCT Denoising (25% dose only)

This project implements the two-stage pipeline from your PDF prompt:

1. Stage 1 teacher (`Stage1Teacher`) learns anatomy segmentation + anatomy embeddings.
2. Stage 2 denoiser (`Stage2Denoiser`) performs anatomy-conditioned residual diffusion denoising.
3. DA-CLIP is intentionally removed and **not used**, because this setup is fixed for 25% dose only.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make scripts executable once:

```bash
chmod +x scripts/*.sh
```

## Expected data layout

```text
data/
  ldct/
    case001_z0000.npy
    ...
  ndct/
    case001_z0000.npy
    ...
  masks_7cls/
    case001_z0000.npy
    ...
  splits/
    train.txt
    val.txt
```

`train.txt` and `val.txt` contain one slice id per line (for example `case001_z0000`).

## Where TotalSegmentator output should be, and format

Use this as the canonical location and format:

- **Raw TotalSegmentator output (per case 3D):** `data/totalseg_raw/<case_id>.nii.gz`
- **Converted training masks (per slice 2D):** `data/masks_7cls/<case_id>_zXXXX.npy`
- **Mask dtype/value format:** integer class ids `0..6` (`np.int16` or `np.int64`)
- **Mask shape per slice:** `[H, W]`, aligned pixel-wise to both `ldct` and `ndct` slice

Run conversion:

```bash
export PYTHONPATH=src
python -m anatomy_denoise.data.prepare_totalseg \
  --totalseg_dir data/totalseg_raw \
  --output_masks_dir data/masks_7cls \
  --manifest_path data/totalseg_manifest.json
```

## Train Stage 1

```bash
bash scripts/train_stage1.sh
```

Checkpoint output: `outputs/stage1/stage1_epoch_XXX.pt`

## Train Stage 2 (no DA-CLIP)

```bash
bash scripts/train_stage2.sh
```

Checkpoint output: `outputs/stage2/stage2_step_XXXXX.pt`

## Inference

```bash
bash scripts/run_infer.sh data/ldct/case001_z0100.npy outputs/preds/case001_z0100_denoised.npy
```

## Important practical notes

- This code is a faithful research scaffold for your prompt, but still lightweight enough to run/modify.
- The `AnatomyMambaBlock` currently uses a practical convolutional proxy for VSSD so it can run without custom kernels. You can replace that block with your preferred Mamba2/VSSD implementation when ready.
- For strict residual diffusion sampling quality, you can later plug in a full DDPM/DDIM solver; current inference is simplified for baseline usability.

## Run full pipeline

```bash
bash scripts/run_all_steps.sh
```

For requirement-to-code traceability, check `IMPLEMENTATION_CHECKLIST.md`.

# Low-CT-Scan-
# Low-CT-Scan-
