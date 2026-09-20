# 🏥 Anatomy-Aware Low-Dose CT Denoising

A two-stage deep learning system that denoises **low-dose CT (LDCT)** images while **preserving anatomical boundaries** — the critical structures radiologists need to see.

> Standard denoisers use pixel-level MSE loss, which blurs organ boundaries. This project tells the denoiser **WHERE** each organ is and **WHAT** each organ looks like, enabling organ-specific denoising strategies.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data](#data)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Implementation Status](#implementation-status)
- [Glossary](#glossary)

---

## Overview

Low-dose CT scans reduce patient radiation exposure but produce noisy images. This project solves the challenge of denoising without sacrificing diagnostic quality:

- **Liver** gets liver-appropriate denoising
- **Lung** gets lung-appropriate denoising
- **Boundaries** get boundary-preserving treatment

The key insight: instead of treating all pixels the same, the model first learns *where* each organ is and *what* each organ looks like, then applies anatomy-conditioned denoising.

---

## System Architecture

The pipeline consists of three parts:

```
Part 0: TotalSegmentator  (pre-trained, runs once offline)
  NDCT 3D volume → organ mask (104 classes → remapped to 7) → saved as .npy

Stage 1: VM-UNet Teacher  (trained first, then frozen forever)
  CT image [B,1,512,512] → S [B,7,512,512] + e_a [B,7,C]

Stage 2: VSSD Denoiser    (trained second, uses frozen Stage 1)
  LDCT + S + e_a → denoised CT [B,1,512,512]
```

**At inference:** Stage 1 runs on LDCT to produce `S` and `e_a`, then Stage 2 denoises. TotalSegmentator is **not** used at inference.

**Why separate stages:** Training them together creates conflicting gradients — segmentation wants sharp boundaries, denoising wants smooth predictions.

### Stage 1 — VM-UNet Teacher

A UNet built with VMamba (Mamba-based) blocks that learns to:
1. Produce **S** `[B,7,512,512]` — soft organ probability map (where each organ is)
2. Produce **e_a** `[B,7,96]` — per-organ feature embeddings (what each organ looks like per patient)

After training, Stage 1 is **frozen forever**.

```
Input [B,1,512,512]
  → PatchEmbed: Conv2d(1,96,k=4,s=4) → [B,96,128,128]

Encoder:
  Scale 1: 2×VSSBlock(96)  → skip1 → PatchMerging → [B,192,64,64]
  Scale 2: 2×VSSBlock(192) → skip2 → PatchMerging → [B,384,32,32]
  Scale 3: 2×VSSBlock(384) → skip3 → PatchMerging → [B,768,16,16]
  Bottleneck: 2×VSSBlock(768) → F [B,768,16,16]

Decoder:
  Scale 3 → Scale 2 → Scale 1 → upsample 4× → [B,96,512,512]
  Seg head: Conv2d(96,7,k=1) → Softmax → S [B,7,512,512]
```

### Stage 2 — VSSD Anatomy-Conditioned Denoiser

Uses **residual diffusion**: predicts the noise residual `(x_ndct - x_ldct)`, then reconstructs the clean image.

The core innovation is the **AnatomyMamba_block**, which combines:

| Component | Inputs | Role |
|-----------|--------|------|
| **adaLN-Zero** | `t_emb` | Timestep-conditioned LayerNorm |
| **Spatial FiLM** | `S` (probability map) | WHERE are the organs? (location-based) |
| **VSSD Scan** | — | Global context via bidirectional 2D scan |
| **Cross-Attention** | `e_a` (embeddings) | WHAT does each organ look like? (content-based) |

---

## Data

**Dataset location:** `/home/teaching/Music/Nigam_51/Project_51/data/`

**Expected structure:**

```
data/
  C002/HDCT/*.dcm     ← clean high-dose CT slices (NDCT)
  C002/LDCT/*.dcm     ← noisy low-dose CT slices (LDCT)
  C004/HDCT/...
  C004/LDCT/...
  ...                 (~70 patients: C002–C296, L004–L266)
  masks/
    C002/0000.npy     ← int8 [512,512], values 0-6 (from TotalSegmentator)
    C002/0001.npy
    ...
```

- **Format:** DICOM (`.dcm`), 512×512 grayscale, Hounsfield Unit values
- **Normalization:** HU values clipped to `[-1000, 3000]` then normalized to `[0, 1]`
- **Pairing:** HDCT/LDCT scans are perfectly aligned (same patient, position, filename ordering)

### 7-Class Organ Mapping

TotalSegmentator outputs 104 labels, remapped to 7 classes with distinct noise characteristics:

| ID | Contents | Why Distinct |
|----|----------|--------------|
| 0 | Background, air | Quantum noise at -1000 HU |
| 1 | Liver, Spleen | Similar HU (40–60), soft-tissue noise |
| 2 | Kidney L+R | Distinct cortex/medulla pattern |
| 3 | Aorta, IVC, Heart | Vessels, motion artifacts |
| 4 | Lung + vessels | Air-filled (–700 HU) |
| 5 | Vertebrae, Ribs | Bone (HU > 400) |
| 6 | Soft tissue, Muscle | Catch-all |

> **Label smoothing (ε=0.1):** TotalSegmentator has 5–15% error. Smooth labels prevent overconfidence on imperfect pseudo-labels.

---

## Project Structure

```
Low-CT-Scan-Denoising/
├── configs/
│   └── stage1_config.yaml          # Stage 1 training configuration
├── datapy/
│   └── dataset.py                  # CTSliceDataset (HDCT/LDCT/mask loading)
├── losses/
│   ├── stage1_losses.py            # SegmentationLoss (CE + smoothing), DiceLoss
│   └── stage2_losses.py            # Stage2LossManager (progressive L_res/L_kd/L_anatomy)
├── mask_checks/                    # Mask inspection utilities
├── models/
│   ├── vmamba_blocks.py            # LayerNorm2d, PatchEmbed, PatchMerging, SS2D, VSSD, VSSBlock
│   ├── vm_unet.py                  # VMUNetEncoder, VMUNetDecoder, VMUNet
│   ├── byol.py                     # ProjectorMLP, PredictorMLP, BYOLModule (EMA)
│   ├── stage1.py                   # Stage1Model, load_stage1_frozen
│   ├── anatomy_mamba.py            # ResNetBlock, SpatialFiLM, VSSDBlock, AnatomyCrossAttention
│   ├── vssd_denoiser.py            # VSSDDenoiser UNet (res + noise heads, KD head)
│   ├── diffusion.py                # ResidualDiffusion (q_sample, losses, DDIM)
│   └── stage2.py                   # Stage2Model (frozen Stage 1 + denoiser + diffusion)
├── scripts/                        # Training/inference scripts
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   └── test_pipeline.py            # Stage 1 + Stage 2 inference + metrics
├── tests/
│   ├── test_stage1_pipeline.py     # Stage 1 integration tests
│   └── test_stage2_pipeline.py     # Stage 2 integration tests
├── training/
│   ├── train_stage1.py             # Stage 1 training loop with BYOL schedule
│   └── train_stage2.py             # Stage 2 training loop with progressive loss schedule
├── utils/
│   ├── generate_masks.py           # TotalSegmentator pipeline
│   ├── verify_masks.py             # Mask validation
│   ├── visualise_masks.py          # Mask overlay visualization
│   ├── explore_data.py             # Data format exploration
│   └── metrics.py                  # PSNR / SSIM / RMSE
├── architecture.md                 # Detailed architecture documentation
├── PROJECT.md                      # Project overview and design decisions
├── result.py                       # Results evaluation
├── cnt.py                          # Count utilities
└── __init__.py
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: ≥16 GB VRAM for 512×512 images)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/rpatidar284/Low-CT-Scan-Denoising.git
cd Low-CT-Scan-Denoising

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pydicom numpy scipy SimpleITK
pip install TotalSegmentator   # for mask generation (offline, one-time)
pip install einops timm
```

---

## Usage

### Step 0 — Generate Pseudo-Labels (one-time)

```bash
python utils/generate_masks.py --data_root /path/to/data --output_dir /path/to/data/masks
```

Verify masks:

```bash
python utils/verify_masks.py
python utils/visualise_masks.py   # visual overlay check
```

### Step 1 — Train Stage 1 (VM-UNet Teacher)

```bash
python training/train_stage1.py --config configs/stage1_config.yaml
```

Expected progress:
- BYOL loss: ~2.0 → ~0.05
- Dice for liver/kidney/lung: > 0.75 by step 50k

### Step 2 — Train Stage 2 (VSSD Denoiser)

```bash
# Uses the frozen Stage 1 checkpoint; progressive L_res → L_kd → L_anatomy schedule
python -c "
from training.train_stage2 import train_stage2
train_stage2(
    stage1_checkpoint='outputs/stage1/stage1_best.pth',
    image_size=128, batch_size=4, total_steps=10000,
    checkpoint_dir='outputs/stage2',
)
"
# or: bash scripts/run_stage2.sh
```

### Inference

```bash
# Full Stage 1 + Stage 2 pipeline + PSNR/SSIM/RMSE + comparison figure
python scripts/test_pipeline.py --patient C002 \
  --stage1 outputs/stage1/stage1_best.pth \
  --stage2 outputs/stage2/stage2_best.pth \
  --output results/pipeline_C002
```

---

## Models

### BYOL — Noise-Invariant Features

Stage 1 is trained on NDCT but runs on LDCT at inference. BYOL forces the network to produce the same organ features regardless of noise level:

```
View 1 (light noise)          View 2 (heavy noise, simulates LDCT)
        ↓                               ↓
  Online Network               Target Network (EMA)
  VM-UNet Encoder              VM-UNet Encoder
  Projector: 768→4096→256      Projector: 768→4096→256
  Predictor: 256→4096→256      z_target.detach()
        ↓                               ↓
  L_byol = 2 - 2 × cosine_sim(q_online, z_target)
```

EMA schedule: τ from 0.996 → 1.0 over training.

### Data Format Convention

| Format | Used by |
|--------|---------|
| **BCHW** (channel-first) | Conv2d, PatchEmbed, PatchMerging, skip connections |
| **BHWC** (channel-last) | VSSBlock, SS2D, VSSD (Mamba operations) |

Permute at every BCHW↔BHWC boundary.

---

## Training

### Stage 1 Loss

```
L_stage1 = 1.0 × L_seg + 0.1 × L_byol

L_seg:   CrossEntropy(logits, pseudo_labels, label_smoothing=0.1)   ← from step 0
L_byol:  BYOL cosine-similarity loss                                 ← from epoch 3
```

### Stage 2 Loss — Progressive Schedule

```
Phase 1 (0–50k steps):      L = L_res
Phase 2 (50k–150k steps):   L = L_res + 0.1 × L_kd
Phase 3 (150k+ steps):      L = L_res + 0.1 × L_kd + 0.05 × L_anatomy
                             (L_anatomy applied every 5th step)
```

| Loss | Description |
|------|-------------|
| **L_res** | MSE(predicted noise, true noise) — primary denoising objective |
| **L_kd** | CrossEntropy from seg head on Stage 2 bottleneck — anatomy-aware features |
| **L_anatomy** | L1 between `e_a` of denoised output vs NDCT — penalizes anatomy distortion |

**Why progressive?** Adding all losses at once creates conflicting gradients. `L_res` first establishes denoising; `L_kd` adds anatomy awareness; `L_anatomy` fine-tunes boundary preservation.

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Input size | 512×512 |
| Base channels | 96 (Stage 1), 64 (Stage 2) |
| Organ classes | 7 |
| Label smoothing | 0.1 |
| BYOL EMA τ start | 0.996 |
| Diffusion timesteps | T (configurable) |

---

## Evaluation Metrics

### Standard Metrics
- **PSNR** — Peak Signal-to-Noise Ratio vs NDCT reference
- **SSIM** — Structural Similarity vs NDCT reference

### Anatomy-Specific Metrics

| Metric | Description |
|--------|-------------|
| **Anatomy Dice Score** | Run TotalSegmentator on denoised output vs NDCT → per-organ Dice |
| **Boundary Preservation Score** | Sobel edges → F1 with 2-pixel tolerance |
| **Anatomy-Weighted SSIM** | Organ pixels weighted 3× vs background |

---

## Implementation Status

### ✅ Completed (Stage 1)

- [x] `models/vmamba_blocks.py` — LayerNorm2d, PatchEmbed, PatchMerging, SS2D, VSSD, VSSBlock
- [x] `models/vm_unet.py` — VMUNetEncoder, VMUNetDecoder, VMUNet (with seg head)
- [x] `models/byol.py` — ProjectorMLP, PredictorMLP, BYOLModule with EMA
- [x] `models/stage1.py` — Stage1Model, load_stage1_frozen
- [x] `losses/stage1_losses.py` — SegmentationLoss, DiceLoss
- [x] `datapy/dataset.py` — CTSliceDataset
- [x] `training/train_stage1.py` — Full training loop with BYOL schedule
- [x] `configs/stage1_config.yaml` — Stage 1 configuration
- [x] `utils/generate_masks.py` — TotalSegmentator pipeline
- [x] `utils/verify_masks.py` — Mask validation
- [x] `utils/visualise_masks.py` — Mask overlay visualization
- [x] `tests/test_stage1_pipeline.py` — Integration tests

### ✅ Completed (Stage 2)

- [x] Stage 2: `AnatomyMamba_block` (SpatialFiLM, CrossAttention, adaLN-Zero) — `models/anatomy_mamba.py`
- [x] Stage 2: Full VSSD UNet denoiser — `models/vssd_denoiser.py` + `models/diffusion.py`
- [x] Stage 2: Training loop with progressive `L_kd` and `L_anatomy` — `training/train_stage2.py`
- [x] Evaluation + inference — `scripts/test_pipeline.py`, `utils/metrics.py`
- [x] Stage 2 integration tests — `tests/test_stage2_pipeline.py`
- [x] Stage 1 training run (on GPU with real data)
- [x] Stage 2 training run (on GPU with real data)

---

## Quick Reference: Key Tensor Shapes

| Variable | Shape | Format | Notes |
|----------|-------|--------|-------|
| Input CT | `[B,1,512,512]` | BCHW | Grayscale HU values |
| After PatchEmbed | `[B,96,128,128]` | BCHW | 4× downsampled |
| Skip 1 | `[B,96,128,128]` | BCHW | Encoder scale 1 |
| Skip 2 | `[B,192,64,64]` | BCHW | Encoder scale 2 |
| Skip 3 | `[B,384,32,32]` | BCHW | Encoder scale 3 |
| F (Bottleneck) | `[B,768,16,16]` | BCHW | For BYOL only |
| S (seg map) | `[B,7,512,512]` | BCHW | Softmax probabilities |
| e_a | `[B,7,96]` | — | Per-organ anatomy embeddings |
| BYOL z, q | `[B,256]` | — | Projected features |
| VSS Block I/O | `[B,H,W,C]` | BHWC | Must permute at boundaries |
| t_emb | `[B,256]` | — | Diffusion timestep embedding |

---

## Glossary

| Term | Meaning |
|------|---------|
| **LDCT / NDCT (HDCT)** | Low-Dose CT (25%, noisy) / Normal-Dose CT (100%, clean) |
| **HU** | Hounsfield Units — CT intensity (-1000 air, 0 water, +400–1000 bone) |
| **S** | Soft segmentation map `[B,7,H,W]` — organ probabilities per pixel |
| **e_a** | Anatomy embeddings `[B,7,C]` — per-organ feature vectors |
| **F** | Bottleneck features `[B,768,H/32,W/32]` — for BYOL only |
| **Pseudo-labels** | Auto-generated organ masks from TotalSegmentator (85–95% accurate) |
| **SS2D** | 2D Selective Scan — causal 4-directional Mamba scan (VMamba1) |
| **VSSD** | Visual State Space Duality — bidirectional non-causal scan (VMamba2) |
| **VSS Block** | Core block: LN→expand→DWConv→scan→gate→residual |
| **PatchMerging** | Downsampling via 4-neighbor gather + project |
| **Spatial FiLM** | Per-pixel feature modulation from S ("where" information) |
| **Cross-Attention** | Pixels query e_a to retrieve organ appearance ("what" information) |
| **adaLN-Zero** | Timestep-conditioned LayerNorm, zero-initialized |
| **BYOL** | Self-supervised noise-invariance via EMA target network |
| **L_seg** | Segmentation cross-entropy loss (Stage 1) |
| **L_byol** | BYOL cosine-similarity loss (Stage 1) |
| **L_res** | Denoising MSE loss (Stage 2) |
| **L_kd** | Knowledge distillation segmentation loss from Stage 2 bottleneck |
| **L_anatomy** | Anatomy feature L1 loss in denoised output space |

---

## License

This project is for research and educational purposes.

---

## Acknowledgements

- [VMamba](https://github.com/MzeroMiko/VMamba) — Visual State Space Model
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) — CT organ segmentation
- [BYOL](https://arxiv.org/abs/2006.07733) — Bootstrap Your Own Latent
