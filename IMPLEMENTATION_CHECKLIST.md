# Prompt Implementation Checklist

This file maps the prompt requirements to code in this repo.

## A. Offline TotalSegmentator pseudo-labels

- NDCT-only segmentation (not LDCT): `scripts/run_totalseg_offline.sh`
- Raw mask location: `data/totalseg_raw/<case_id>.nii.gz`
- 104 -> 7 class remap: `src/anatomy_denoise/data/prepare_totalseg.py`
- Slice mask output: `data/masks_7cls/<case_id>_zXXXX.npy`

## B. Stage 1 teacher (VM-UNet style role)

- Model: `src/anatomy_denoise/models/stage1_teacher.py`
- Outputs:
  - soft segmentation `S`
  - anatomy embeddings `e_a`
  - bottleneck feature `F`
- BYOL machinery: `Stage1BYOL` + `train_stage1.py`
- Loss schedule:
  - segmentation CE with label smoothing
  - BYOL auxiliary with warmup

## C. Stage 2 denoiser (anatomy-conditioned residual diffusion)

- Model: `src/anatomy_denoise/models/stage2_denoiser.py`
- Conditioning:
  - Spatial FiLM with `S` at multi-scale
  - Cross-attention with `e_a`
  - adaLN-zero with timestep embedding
- No DA-CLIP (25% dose only): done
- Training:
  - residual diffusion MSE
  - KD segmentation loss after step threshold
  - anatomy feature matching after later threshold

## D. Frozen Stage 1 in Stage 2 training

- `train_stage2.py` sets `stage1.eval()` and `requires_grad=False`
- Stage 1 calls are wrapped in `torch.no_grad()`

## E. Evaluation metrics

- Script: `src/anatomy_denoise/trainers/evaluate.py`
- Includes:
  - PSNR
  - anatomy-weighted SSIM
  - boundary F1

## F. End-to-end execution

- `scripts/run_all_steps.sh` executes the whole sequence.

## G. Known practical approximation

- The current `AnatomyMambaBlock` uses a practical mixer proxy instead of a custom Mamba2 VSSD kernel for immediate reproducibility.
- If you want strict kernel-level VSSD, it can be swapped in the same block entry point.

