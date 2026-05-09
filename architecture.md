# Anatomy-Aware Low-Dose CT Denoising — Architecture

## System Overview

Three parts, two trained:

```
Part 0: TotalSegmentator (pre-trained, runs once offline)
  NDCT 3D volume → 3D organ mask (104 classes) → remapped to 7 classes → saved as .npy

Stage 1: VM-UNet Teacher (trained first, then frozen forever)
  CT image [B,1,512,512] → S [B,7,512,512] + e_a [B,7,C] + F [B,768,16,16]

Stage 2: VSSD Denoiser (trained second, uses frozen Stage 1)
  LDCT + S + e_a → denoised CT [B,1,512,512]
```

At inference: Stage 1 runs on LDCT → produces S, e_a → Stage 2 denoises. TotalSegmentator is NOT used at inference.

**Why separate stages:** Training together creates conflicting gradients — segmentation wants sharp boundaries, denoising wants smooth predictions.

---

## Data Format Convention

- PyTorch default: **BCHW** (channel-first) — used by Conv2d, PatchEmbed, PatchMerging
- Mamba operations: **BHWC** (channel-last) — used by VSSBlock, VSSD, SS2D
- Permute at every BCHW↔BHWC boundary; all skip connections, bottleneck, and public outputs stay BCHW

---

## TotalSegmentator Pipeline (Part 0)

1. Stack 2D `.npy` slices per patient → 3D volume `[D, 512, 512]`
2. Convert to NIfTI via SimpleITK
3. Run TotalSegmentator → 3D mask with integers 0-103
4. Remap 104 classes → 7 classes
5. Split back to 2D slices, save as `.npy`

### 7-Class Mapping

| Class ID | Contents | Rationale |
|----------|----------|-----------|
| 0 | Background, air | Quantum noise dominates at -1000 HU |
| 1 | Liver, Spleen | Similar HU range (40-60), similar soft-tissue noise |
| 2 | Kidney left, Kidney right | Distinct cortex/medulla pattern |
| 3 | Aorta, IVC | Blood vessels, motion artifacts |
| 4 | Lung, Lung vessels | Air-filled (-700 HU), different noise statistics |
| 5 | Vertebrae, Ribs | Bone (HU > 400) |
| 6 | Soft tissue, Muscle, rest | Catch-all |

104 classes would be too hard to segment accurately, too much memory, and unnecessary for denoising. 7 classes capture all meaningfully different tissue types.

**NDCT masks apply to LDCT:** Paired images are perfectly aligned (same patient, same position). Run TotalSegmentator on NDCT (clean) for better accuracy; the same mask is valid for the corresponding LDCT slice.

### Label Smoothing (ε = 0.1)

TotalSegmentator has 5-15% error. Smooth labels prevent overconfidence:
```
smooth[k] = 0.9 if k == true_class else 0.1/6 ≈ 0.014
```

---

## Stage 1: VM-UNet Teacher

### Architecture

```
Input [B,1,512,512]
  → PatchEmbed: Conv2d(1,96,k=4,s=4) → [B,96,128,128]

Encoder:
  Scale 1: 2×VSSBlock(96)  → skip1 [B,96,128,128]   → PatchMerging → [B,192,64,64]
  Scale 2: 2×VSSBlock(192) → skip2 [B,192,64,64]     → PatchMerging → [B,384,32,32]
  Scale 3: 2×VSSBlock(384) → skip3 [B,384,32,32]     → PatchMerging → [B,768,16,16]
  Bottleneck: 2×VSSBlock(768) → F [B,768,16,16]

Decoder:
  Scale 3: upsample 2× + Linear(768→384) → cat(skip3) → 2×VSSBlock → [B,384,32,32]
  Scale 2: upsample 2× + Linear(384→192) → cat(skip2) → 2×VSSBlock → [B,192,64,64]
  Scale 1: upsample 2× + Linear(192→96)  → cat(skip1) → 2×VSSBlock → [B,96,128,128]
  Final: upsample 4× → [B,96,512,512]

Segmentation head: Conv2d(96,7,k=1) → logits → Softmax → S [B,7,512,512]
```

**PatchMerging:** Gathers TL/TR/BL/BR sub-grids, concatenates to [B,4C,H/2,W/2], then Conv1×1 → [B,2C,H/2,W/2]. No information is thrown away.

**Upsampling:** Bilinear interpolation (avoids transposed conv checkerboard artifacts) + Linear projection to adjust channels.

### Three Outputs

**S — Soft Segmentation Map [B,7,512,512]:** Organ probabilities per pixel (softmax over class dim). Soft boundaries encode uncertainty — e.g., a boundary pixel gets `[0.30, 0.55, 0.05, ...]` which Stage 2 uses for blended conditioning.

**e_a — Anatomy Embeddings [B,7,C]:** Masked average pooling of decoder features (before seg head) weighted by S:
```python
for k in range(7):
    weight = S[:, k, :, :]                        # [B, H, W]
    e_a_k = (weight * decoder_features).sum([2,3]) / (weight.sum([1,2]) + 1e-8)
```
This is a patient-specific per-organ feature vector. Each patient's liver looks different; e_a captures this.

**F — Bottleneck Features [B,768,16,16]:** Used only for BYOL training. NOT passed to Stage 2.

### VSS Block Internals (BHWC in/out)

```
x [B,H,W,C]
  → LayerNorm
  → Linear(C→2C) → split → x_main, x_gate
  → x_main: permute→BCHW → DWConv3×3 → permute→BHWC  (local detail)
  → x_main: SS2D (Stage 1) or VSSD (Stage 2)          (global context)
  → x_main * SiLU(x_gate)                               (gating)
  → Linear(C→C)
  → residual: output = x + DropPath(x_main)
```

### SS2D vs VSSD

- **SS2D (VMamba1):** 4-directional causal scan. When processing position i, only positions 0..i are visible. Used in Stage 1.
- **VSSD (VMamba2):** Bidirectional non-causal scan on both H and W axes. Every position sees all others. 3-4× faster. Used in Stage 2.

### Stage 1 Losses

```
L_stage1 = 1.0 * L_seg + 0.1 * L_byol
```

**L_seg:** CrossEntropy(S, pseudo_labels, label_smoothing=0.1). Primary objective — active from step 0.

**L_byol:** BYOL cosine-similarity loss. Makes features noise-invariant (NDCT vs LDCT). Active from epoch 3 onward.

---

## BYOL — Noise-Invariant Features

Stage 1 is trained on NDCT but at inference runs on LDCT. BYOL ensures the same organ produces the same e_a regardless of noise level.

### Architecture

```
View 1 (NDCT + small noise)          View 2 (LDCT-level noise)
        ↓                                     ↓
  Online Network                         Target Network (EMA)
  VM-UNet Encoder                        VM-UNet Encoder
        ↓                                     ↓
  Projector MLP                          Projector MLP
  768→4096→256                           768→4096→256
        ↓                                     ↓
  Predictor MLP                          z_target [B,256]
  256→4096→256                                 ↓
        ↓                                  .detach()
  q_online [B,256]                            │
        └──────────────┬──────────────────────┘
                       ↓
  L_byol = 2 - 2*cosine_sim(q_online, z_target.detach())
  (symmetric: also compute reverse direction)
```

**EMA update:** `target = τ*target + (1-τ)*online`, with τ: 0.996 → 1.0 over training. Target changes slowly, no gradients flow through it.

**Why the asymmetry (predictor only in online):** Prevents representational collapse — without it, both networks learn to output constant vectors.

---

## Stage 2: VSSD Anatomy-Conditioned Denoiser

### Residual Diffusion

Instead of predicting the full clean image, Stage 2 predicts the residual:
```
true_residual = x_ndct - x_ldct
```

Diffusion process: add Gaussian noise to the residual at timestep t, train network to predict the noise ε. At inference: start from pure noise, iteratively denoise over T steps, then `x_denoised = x_ldct + predicted_residual`.

**No DA-CLIP:** Since you use a fixed dose (25%), the dose embedding `e_d` is removed. Only timestep embedding `t_emb` conditions the diffusion process.

### AnatomyMamba_block — Core Innovation

Each block receives: `x [B,C,H,W]`, `S [B,7,H,W]`, `e_a [B,7,C_anat]`, `t_emb [B,256]`

**Step 1 — adaLN-Zero:** Timestep-conditioned LayerNorm. γ and β are computed from `t_emb` via Linear. Initialized to output zeros → block starts as identity. At different timesteps, the block behaves differently (aggressive denoising at high t, subtle refinement at low t).

**Step 2 — Spatial FiLM:** Organ-specific spatial conditioning. A small ConvNet processes S → per-pixel (γ, β) pairs:
```
film_params = Conv2d(7→32,k=3) → SiLU → Conv2d(32→2C,k=1)
gamma, beta = chunk(film_params)
output = (1+gamma) * LayerNorm(x) + beta
```
A liver pixel gets "liver-mode" normalization; a lung pixel gets "lung-mode" normalization. S is bilinearly downsampled to match each UNet scale.

**Step 3 — VSSD Scan:** Bidirectional non-causal 2D scan. Full global context after spatial conditioning.

**Step 4 — Cross-Attention with e_a:** Each pixel queries the 7 organ embeddings:
```
Q = x (flattened pixels)     [B, H*W, C]
K = V = e_a                  [B, 7, C_anat]
output = x + MultiheadAttention(Q, K, V)
```
Pixels attend to the organ embedding they most resemble, retrieving patient-specific organ appearance information. Uses multi-head attention with `num_heads = C // 32`.

| Spatial FiLM | Cross-Attention |
|---|---|
| Uses S (probability map) | Uses e_a (feature embeddings) |
| Answers: WHERE are the organs? | Answers: WHAT does each organ look like? |
| Location-based | Content/similarity-based |

### Stage 2 Architecture

```
Inputs:
  x_ldct [B,1,512,512]  — original noisy CT
  x_noisy [B,1,512,512] — noisy residual at timestep t
  t — diffusion timestep
  S_scales — [S_512, S_256, S_128, S_64, S_32]
  e_a [B,7,C_anatomy]

init_conv: Conv2d(2, 64, k=7, p=3)  — concat(x_noisy, x_ldct) as input
time_mlp: sinusoidal(t) → Linear(64,256) → SiLU → Linear(256,256)

Encoder:
  Scale 1: AnatomyMamba_block × N (dim=64)   → skip → downsample → [B,128,256,256]
  Scale 2: AnatomyMamba_block × N (dim=128)  → skip → downsample → [B,256,128,128]
  Scale 3: AnatomyMamba_block × N (dim=256)  → skip → downsample → [B,512,64,64]
  Scale 4: AnatomyMamba_block × N (dim=512)  → skip → downsample → [B,512,32,32]

Bottleneck:
  AnatomyMamba_block (dim=512) → [B,512,32,32]
  Seg KD head: Conv2d(512,7,k=1) → upsample → [B,7,512,512]  (for L_kd)

Decoder (mirror of encoder with skip connections):
  Scale 4 up → Scale 3 up → Scale 2 up → Scale 1 up → [B,64,512,512]

final_conv: Conv2d(64, 1, k=1) → [B,1,512,512]  (predicted noise/residual)
```

### Stage 2 Losses — Progressive Schedule

```
Phase 1 (0-50k steps):      L = L_res
Phase 2 (50k-150k steps):   L = L_res + 0.1 * L_kd
Phase 3 (150k+ steps):      L = L_res + 0.1 * L_kd + 0.05 * L_anatomy
                             (L_anatomy applied every 5th step)
```

**L_res:** MSE between predicted and true noise/residual. Primary denoising objective. Active from step 0. Weight = 1.0.

**L_kd:** CrossEntropy from Seg KD head (small decoder on Stage 2 bottleneck → organ class predictions). Forces Stage 2's bottleneck features to be anatomy-interpretable, making cross-attention with e_a meaningful. Weight = 0.1.

**L_anatomy:** Run frozen Stage 1 on predicted denoised image → e_a_pred. Compare with e_a from NDCT via L1 loss. Directly penalizes anatomy distortion in the denoised output. Weight = 0.05.

**Why progressive:** Adding all losses at once creates conflicting gradients before the network has learned anything. L_res first establishes basic denoising, L_kd makes features anatomy-aware, L_anatomy fine-tunes anatomy preservation.

---

## Complete Data Flow (One Stage 2 Training Step)

```python
# Step A: Anatomy conditioning from frozen Stage 1 (no gradients)
with torch.no_grad():
    S, e_a = frozen_stage1(x_ldct)
    S_scales = [interpolate(S, size) for size in [512, 256, 128, 64, 32]]

# Step B: Diffusion forward process
true_residual = x_ndct - x_ldct
t = randint(0, T)
noise = randn_like(true_residual)
noisy_residual = q_sample(true_residual, t, noise)

# Step C: Time embedding
t_emb = time_mlp(sinusoidal_embedding(t))

# Step D: Stage 2 forward
predicted_noise = stage2_unet(
    x=noisy_residual, x_ldct=x_ldct, t=t_emb,
    S_scales=S_scales, e_a=e_a
)

# Step E: Loss computation
L_res = MSE(predicted_noise, noise)
if step > 50000:
    L_kd = CrossEntropy(kd_head(bottleneck), pseudo_labels)
if step > 150000 and step % 5 == 0:
    x_hat = x_ldct + predicted_noise
    e_a_pred = frozen_stage1(x_hat).e_a
    e_a_gt   = frozen_stage1(x_ndct).e_a    # can be precomputed
    L_anatomy = L1(e_a_pred, e_a_gt)

# Step F: Backward (Stage 2 only)
total_loss.backward()
optimizer.step()
```

---

## Design Decisions

**Why VM-UNet for Stage 1:** Regular UNet convolutions have local receptive fields. Organ segmentation requires global context (is this dark blob a kidney or cyst? → look at surrounding structures). Mamba's 4-directional scan gives global context at O(N) cost.

**Why freeze Stage 1 during Stage 2 training:** If Stage 1 were updated, S and e_a would change every step — Stage 2 would learn from a moving target and never converge properly.

**Why soft S instead of hard masks:** (1) Differentiable — can't backprop through threshold, (2) avoids boundary artifacts from hard organ transitions, (3) preserves uncertainty information (S=0.51 vs S=0.99 are meaningfully different), (4) damps TotalSegmentator errors at boundaries.

**Why AdamW over Adam:** AdamW applies weight decay directly to weights (separate from gradient update), which is mathematically correct for Mamba/Transformer networks.

**Why bilinear upsampling:** Transposed convolutions create checkerboard artifacts. Bilinear is smooth and artifact-free.

---

## Evaluation Metrics

**Standard:** PSNR, SSIM (measured against NDCT reference).

**Anatomy-specific:**
- **Anatomy Dice Score:** Run TotalSegmentator on denoised output vs NDCT → per-organ Dice. Measures boundary preservation.
- **Boundary Preservation Score:** Sobel edge detection → F1 with 2-pixel tolerance between denoised and NDCT edge maps.
- **Anatomy-Weighted SSIM:** Up-weight organ pixels 3× relative to background pixels.

---

## Implementation Order

### Milestone 0: Pseudo-Label Generation
- Install TotalSegmentator, stack 2D slices → 3D volume, convert to NIfTI
- Run on all NDCT patients, remap 104→7 classes, save 2D masks
- Verify: visualize mask overlay on CT slice

### Milestone 1: VM-UNet Implementation
- PatchEmbed (1→96, stride 4), PatchMerging, VSSBlock, VMUNetEncoder, VMUNetDecoder, VMUNet
- Test: `[2,1,512,512] → S [2,7,512,512], e_a [2,7,96], F [2,768,16,16]`

### Milestone 2: Stage 1 Training (L_seg only)
- Dataset loader with mask loading, CrossEntropy + label_smoothing=0.1
- Target: Dice > 0.75 for liver, kidney, lung by step 50k

### Milestone 3: BYOL
- EMA target network, Projector/Predictor MLPs, two-view noise augmentation
- L_byol weight 0.0 first 5 epochs, then 0.1
- Monitor: BYOL loss from ~2.0 → ~0.05

### Milestone 4: AnatomyMamba_block
- SpatialFiLM, CrossAttentionWithAnatomy, adaLN-Zero, VSSD_Block
- Test with dummy tensors

### Milestone 5: Stage 2 Training (L_res only)
- Load frozen Stage 1, S interpolation to all scales, train with L_res
- Verify PSNR ≥ baseline

### Milestones 6-7: L_kd and L_anatomy
- Seg KD head, L_kd at step 50k, L_anatomy at step 150k (every 5th step)
- Monitor: PSNR and Dice should both improve

---

## Glossary (project-specific terms only)

| Term | Meaning |
|------|---------|
| **LDCT / NDCT** | Low-Dose CT (25% dose, noisy) / Normal-Dose CT (100%, clean) |
| **HU** | Hounsfield Units — CT intensity scale (-1000 air, 0 water, +400-1000 bone) |
| **S** | Soft segmentation map [B,7,H,W] — organ probabilities per pixel |
| **e_a** | Anatomy embeddings [B,7,C] — per-organ feature vectors |
| **F** | Bottleneck features [B,768,16,16] — used for BYOL only |
| **Pseudo-labels** | Auto-generated organ masks from TotalSegmentator |
| **SS2D** | 2D Selective Scan — causal 4-directional Mamba (VMamba1) |
| **VSSD** | Visual State Space Duality — bidirectional non-causal 2D scan (VMamba2) |
| **VSS Block** | VMamba building block: LN→expand→DWConv→scan→gate→residual |
| **PatchMerging** | Downsampling: gather 4 neighbors, concat, project → halve H,W, double C |
| **Spatial FiLM** | Per-pixel feature modulation from S (where are organs) |
| **Cross-Attention** | Pixels query e_a to retrieve organ appearance (what do organs look like) |
| **adaLN-Zero** | Timestep-conditioned LayerNorm, zero-initialized for stable training |
| **BYOL** | Self-supervised noise-invariance training via EMA target network |
| **L_seg** | Segmentation CE loss (Stage 1) |
| **L_byol** | BYOL cosine-similarity loss (Stage 1) |
| **L_res** | Denoising MSE loss (Stage 2) |
| **L_kd** | Knowledge distillation seg loss from bottleneck (Stage 2) |
| **L_anatomy** | Anatomy feature L1 loss in output space (Stage 2) |
