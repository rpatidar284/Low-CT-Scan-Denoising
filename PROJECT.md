# Anatomy-Aware Low-Dose CT Denoising — Project Overview

## What This Project Does

Low-dose CT scans reduce patient radiation exposure but produce noisy images. This project builds a two-stage deep learning system that denoises low-dose CT (LDCT) images while **preserving anatomical boundaries** — the critical structures radiologists need to see.

Standard denoisers use pixel-level MSE loss, which blurs organ boundaries (the network "plays it safe" at uncertain regions by outputting the average). Our approach: tell the denoiser WHERE each organ is and WHAT each organ looks like, so it applies organ-specific denoising strategies. Liver gets liver-appropriate denoising. Lung gets lung-appropriate denoising. Boundaries get boundary-preserving treatment.

---

## System Architecture (3 Parts)

```
Part 0: TotalSegmentator (pre-trained, runs once offline)
  NDCT 3D volume → organ mask (104 classes → remapped to 7) → saved as .npy

Stage 1: VM-UNet Teacher (trained first, then frozen forever)
  CT image [B,1,512,512] → S [B,7,512,512] + e_a [B,7,C]

Stage 2: VSSD Denoiser (trained second, uses frozen Stage 1)
  LDCT + S + e_a → denoised CT [B,1,512,512]
```

**At inference:** Stage 1 runs on LDCT, produces S and e_a, Stage 2 denoises. TotalSegmentator is NOT used at inference.

**Why separate stages:** Training them together creates conflicting gradients — segmentation wants sharp boundaries, denoising wants smooth predictions.

---

## Data

**Location:** `/home/teaching/Music/Nigam_51/Project_51/data/`

**Structure:**
```
data/
  C002/HDCT/*.dcm    ← clean high-dose CT slices (NDCT)
  C002/LDCT/*.dcm    ← noisy low-dose CT slices (LDCT)
  C004/HDCT/...
  C004/LDCT/...
  ... (~70 patients: C002-C296, L004-L266)
  masks/
    C002/0000.npy    ← int8 [512,512], values 0-6 (from TotalSegmentator)
    C002/0001.npy
    ...
```

**Data format:** DICOM (.dcm), 512×512 grayscale, Hounsfield Unit values (clipped to [-1000, 3000] then normalized to [0,1]).

**Key fact:** Paired HDCT/LDCT scans — same patient, same position, same filename ordering. NDCT mask is valid for LDCT (identical anatomy, different noise).

---

## 7-Class Organ Mapping

TotalSegmentator outputs 104 labels. We remap to 7 classes that capture meaningfully different noise characteristics:

| ID | Contents | Why Distinct |
|----|----------|-------------|
| 0 | Background, air | Quantum noise at -1000 HU |
| 1 | Liver, Spleen | Similar HU (40-60), soft-tissue noise |
| 2 | Kidney L+R | Cortex/medulla pattern |
| 3 | Aorta, IVC, Heart | Vessels, motion artifacts |
| 4 | Lung + vessels | Air-filled (-700 HU) |
| 5 | Vertebrae, Ribs | Bone (HU > 400) |
| 6 | Soft tissue, Muscle | Catch-all |

**Label smoothing (ε=0.1):** TotalSegmentator has 5-15% error. Smooth labels prevent overconfidence on imperfect pseudo-labels.

---

## Data Format Convention

- **BCHW** (channel-first): PyTorch default — Conv2d, PatchEmbed, PatchMerging, skip connections
- **BHWC** (channel-last): Mamba operations — VSSBlock, SS2D, VSSD
- Permute at every BCHW↔BHWC boundary

---

## Stage 1: VM-UNet Teacher

### Purpose
Learn to look at a CT image and produce:
1. **S** [B,7,512,512] — soft organ probability map (where each organ is)
2. **e_a** [B,7,96] — per-organ feature embeddings (what each organ looks like in this patient)

After training, Stage 1 is **frozen forever**. Stage 2 uses S and e_a as fixed anatomy conditioning.

### Architecture

```
Input [B,1,512,512]
  → PatchEmbed: Conv2d(1,96,k=4,s=4) → [B,96,128,128]

Encoder (4 scales):
  Scale 1: 2×VSSBlock(96)  → skip1 → PatchMerging → [B,192,64,64]
  Scale 2: 2×VSSBlock(192) → skip2 → PatchMerging → [B,384,32,32]
  Scale 3: 2×VSSBlock(384) → skip3 → PatchMerging → [B,768,16,16]
  Bottleneck: 2×VSSBlock(768) → F [B,768,16,16]

Decoder (3 scales + final upsample):
  Scale 3: upsample 2× → cat(skip3) → 2×VSSBlock → [B,384,32,32]
  Scale 2: upsample 2× → cat(skip2) → 2×VSSBlock → [B,192,64,64]
  Scale 1: upsample 2× → cat(skip1) → 2×VSSBlock → [B,96,128,128]
  Final: upsample 4× → [B,96,512,512]

Seg head: Conv2d(96,7,k=1) → S [B,7,512,512]
```

### Key Components

**PatchEmbed:** Conv2d with stride=4 divides 512×512 into 4×4 patch grid → 128×128 positions with 96-dim features each. Same design as Vision Transformer.

**PatchMerging:** Gathers TL/TR/BL/BR neighbors, concat → [B,4C,H/2,W/2], then Conv1×1 → [B,2C,H/2,W/2]. No information thrown away.

**VSS Block:** The core building block (operates in BHWC):
1. LayerNorm
2. Linear(C→2C) → split into x_main and x_gate
3. DWConv3×3 on x_main (local neighborhood)
4. SS2D/VSSD scan (global context)
5. Gating: x_main * SiLU(x_gate)
6. Linear(C→C) output projection
7. Residual: output = input + DropPath(x_main)

**SS2D (Stage 1):** 4-directional causal 2D scan. Each position sees only previous positions in scan order.

**VSSD (Stage 2):** Bidirectional non-causal scan on both H and W axes. Every position sees all others. 3-4× faster than SS2D.

### Three Outputs

**S [B,7,512,512]:** Softmax over 7 classes per pixel. Soft boundaries preserve uncertainty.

**e_a [B,7,96]:** Masked average pooling — for each organ class k, compute weighted average of decoder features:
```python
for k in range(7):
    weight = S[:, k, :, :]
    e_a_k = (weight * decoder_features).sum([2,3]) / (weight.sum([1,2]) + 1e-8)
```
This is a patient-specific anatomy descriptor — each patient's liver gets its own 96-dim vector.

**F [B,768,16,16]:** Bottleneck features, used for BYOL only. NOT passed to Stage 2.

### BYOL — Noise-Invariant Features

Stage 1 is trained on NDCT, but at inference runs on LDCT. BYOL forces the network to produce the same organ features regardless of noise level.

```
View 1 (light noise)          View 2 (heavy noise, simulates LDCT)
        ↓                               ↓
  Online Network                   Target Network (EMA)
  VM-UNet Encoder                  VM-UNet Encoder
        ↓                               ↓
  Projector: 768→4096→256          Projector: 768→4096→256
        ↓                               ↓
  Predictor: 256→4096→256          z_target.detach()
        ↓                               ↓
  q_online ──────────┬────────────────┘
                     ↓
  L_byol = 2 - 2*cosine_sim(q_online, z_target.detach())
```

EMA update: `target = τ*target + (1-τ)*online`, with τ: 0.996→1.0. Target changes slowly, no gradients flow through it.

Without the predictor+EMA asymmetry, both networks collapse to output constants.

### Stage 1 Loss

```
L_stage1 = 1.0 * L_seg + 0.1 * L_byol

L_seg:  CrossEntropy(logits, pseudo_labels, label_smoothing=0.1) — active from step 0
L_byol: BYOL cosine-similarity loss — active from epoch 3
```

**Training data:** HDCT (clean) images with TotalSegmentator pseudo-labels.

---

## Stage 2: VSSD Anatomy-Conditioned Denoiser

### Residual Diffusion

Instead of predicting the full clean image, Stage 2 predicts the residual:
```
true_residual = x_ndct - x_ldct
```

Diffusion: add Gaussian noise to residual at timestep t → train network to predict noise ε → at inference, start from pure noise, iteratively denoise, then `x_denoised = x_ldct + predicted_residual`.

Fixed dose (25%) — no DA-CLIP dose embedding needed. Only timestep embedding.

### AnatomyMamba_block

Each block receives: `x [B,C,H,W]`, `S [B,7,H,W]`, `e_a [B,7,C_anat]`, `t_emb [B,256]`

**Step 1 — adaLN-Zero:** Timestep-conditioned LayerNorm. γ, β computed from t_emb. Zero-initialized → block starts as identity.

**Step 2 — Spatial FiLM:** S processed by small ConvNet → per-pixel (γ,β). A liver pixel gets liver-mode normalization; a lung pixel gets lung-mode. S is bilinearly downsampled to each UNet scale.

**Step 3 — VSSD Scan:** Bidirectional non-causal 2D scan for global context.

**Step 4 — Cross-Attention with e_a:** Each pixel queries the 7 organ embeddings via MultiheadAttention. Pixels attend to the organ they most resemble, retrieving patient-specific organ appearance.

| Spatial FiLM | Cross-Attention |
|---|---|
| Uses S (probabilities) | Uses e_a (embeddings) |
| WHERE are the organs? | WHAT does each organ look like? |
| Location-based | Content/similarity-based |

### Stage 2 Losses (Progressive)

```
Phase 1 (0-50k steps):      L = L_res
Phase 2 (50k-150k steps):   L = L_res + 0.1 * L_kd
Phase 3 (150k+ steps):      L = L_res + 0.1 * L_kd + 0.05 * L_anatomy
                             (L_anatomy every 5th step)
```

- **L_res:** MSE(predicted_noise, noise) — primary denoising objective
- **L_kd:** CrossEntropy from small seg head on Stage 2 bottleneck → forces anatomy-aware features
- **L_anatomy:** Run frozen Stage 1 on denoised output → compare e_a with NDCT's e_a via L1 loss → directly penalizes anatomy distortion

**Why progressive:** Starting all losses at once creates conflicting gradients. L_res first establishes denoising, L_kd adds anatomy awareness, L_anatomy fine-tunes preservation.

---

## Key Design Decisions

- **VM-UNet for Stage 1:** Mamba gives global receptive field at O(N) cost. Organ segmentation needs global context (is this dark blob a kidney or cyst?).
- **Freeze Stage 1 during Stage 2:** Prevents moving-target problem — S and e_a stay stable, Stage 2 can learn to use them reliably.
- **Soft S instead of hard masks:** Differentiable, no boundary artifacts, preserves uncertainty (S=0.51 vs S=0.99 is meaningful).
- **AdamW over Adam:** Decoupled weight decay, mathematically correct for Mamba/Transformer networks.
- **Bilinear upsampling:** Avoids checkerboard artifacts from transposed convolutions.

---

## Evaluation Metrics

**Standard:** PSNR, SSIM vs NDCT reference.

**Anatomy-specific:**
- **Anatomy Dice:** Run TotalSegmentator on denoised output vs NDCT → per-organ Dice
- **Boundary Preservation:** Sobel edges → F1 with 2-pixel tolerance
- **Anatomy-Weighted SSIM:** Organ pixels weighted 3× vs background

---

## Implementation Status

### Completed (Stage 1)
- [x] `models/vmamba_blocks.py` — LayerNorm2d, PatchEmbed, PatchMerging, SS2D, VSSD, VSSBlock
- [x] `models/vm_unet.py` — VMUNetEncoder, VMUNetDecoder, VMUNet (full model with seg head)
- [x] `models/byol.py` — ProjectorMLP, PredictorMLP, BYOLModule with EMA
- [x] `models/stage1.py` — Stage1Model (backbone + BYOL + e_a), load_stage1_frozen
- [x] `losses/stage1_losses.py` — SegmentationLoss (CE + label smoothing), DiceLoss
- [x] `datapy/dataset.py` — CTSliceDataset with HDCT/LDCT/mask loading
- [x] `training/train_stage1.py` — Complete training loop with BYOL schedule
- [x] `configs/stage1_config.yaml` — Full Stage 1 configuration
- [x] `utils/generate_masks.py` — TotalSegmentator pipeline
- [x] `utils/verify_masks.py` — Mask validation
- [x] `utils/visualise_masks.py` — Mask overlay visualization
- [x] `utils/explore_data.py` — Data format exploration
- [x] `tests/test_stage1_pipeline.py` — Integration tests

### Remaining
- [ ] Stage 1 training run (on GPU with real data)
- [ ] Stage 2: AnatomyMamba_block (SpatialFiLM, CrossAttention, adaLN-Zero)
- [ ] Stage 2: Full VSSD UNet denoiser
- [ ] Stage 2: Training loop with progressive L_kd and L_anatomy
- [ ] Evaluation scripts

---

## Quick Reference: Key Shapes

| Variable | Shape | Format | Notes |
|----------|-------|--------|-------|
| Input CT | [B, 1, 512, 512] | BCHW | Grayscale HU values |
| After PatchEmbed | [B, 96, 128, 128] | BCHW | 4× downsampled |
| Skip 1 | [B, 96, 128, 128] | BCHW | Encoder scale 1 |
| Skip 2 | [B, 192, 64, 64] | BCHW | Encoder scale 2 |
| Skip 3 | [B, 384, 32, 32] | BCHW | Encoder scale 3 |
| F (Bottleneck) | [B, 768, 16, 16] | BCHW | For BYOL |
| decoder_features | [B, 96, 512, 512] | BCHW | For e_a pooling |
| logits | [B, 7, 512, 512] | BCHW | Raw seg scores |
| S | [B, 7, 512, 512] | BCHW | Softmax probs |
| e_a | [B, 7, 96] | — | Anatomy embeddings |
| BYOL z, q | [B, 256] | — | Projected features |
| VSS Block I/O | [B, H, W, C] | BHWC | Must permute |
| t_emb | [B, 256] | — | Diffusion timestep |

## Glossary (project-specific)

| Term | Meaning |
|------|---------|
| **LDCT / NDCT (HDCT)** | Low-Dose CT (25%, noisy) / Normal-Dose CT (100%, clean) |
| **HU** | Hounsfield Units — CT intensity (-1000 air, 0 water, +400-1000 bone) |
| **S** | Soft segmentation map [B,7,H,W] — organ probabilities per pixel |
| **e_a** | Anatomy embeddings [B,7,C] — per-organ feature vectors |
| **F** | Bottleneck features [B,768,H/32,W/32] — for BYOL only |
| **Pseudo-labels** | Auto-generated organ masks from TotalSegmentator (85-95% accurate) |
| **SS2D** | 2D Selective Scan — causal 4-directional Mamba scan (VMamba1) |
| **VSSD** | Visual State Space Duality — bidirectional non-causal scan (VMamba2) |
| **VSS Block** | Core building block: LN→expand→DWConv→scan→gate→residual |
| **PatchMerging** | Downsampling via 4-neighbor gather + project |
| **Spatial FiLM** | Per-pixel feature modulation from S ("where" information) |
| **Cross-Attention** | Pixels query e_a to retrieve organ appearance ("what" information) |
| **adaLN-Zero** | Timestep-conditioned LayerNorm, zero-initialized |
| **BYOL** | Self-supervised noise-invariance via EMA target network |
| **L_seg** | Segmentation cross-entropy loss |
| **L_byol** | BYOL cosine-similarity loss |
| **L_res** | Denoising MSE loss (Stage 2) |
| **L_kd** | Knowledge distillation segmentation loss from Stage 2 bottleneck |
| **L_anatomy** | Anatomy feature L1 loss in denoised output space |
