# Stage 2 Agent Prompting Script
## Anatomy-Aware Low-Dose CT Denoising — Residual Diffusion Student Network

> **How to use this script:**
> - Send each numbered prompt to your coding agent **one at a time**
> - Run the verification test after each step before proceeding
> - Stage 1 must be fully trained and `stage1_best.pth` must exist before starting
> - Reference files: `DADiff.py` (diffusion architecture), `denoising_diffusion_pytorch.py` (base DDPM)

---

## PREREQUISITES

```
Before starting Stage 2, confirm:
  ✓ Stage 1 training complete
  ✓ Checkpoint exists: /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage1/stage1_best.pth
  ✓ Val Dice (mean) ≥ 0.50 from Stage 1 training log
  ✓ conda environment: ldct_mamba
  ✓ Project root: /home/teaching/Music/Nigam_51/Project_51/
```

---

## EXECUTION ORDER OVERVIEW

```
STAGE 2 — Anatomy-Conditioned Residual Diffusion  (train after Stage 1 is frozen)

  PART A  — Denoising UNet with VSSD + Anatomy Conditioning
    STEP 1   Stage 2 config file
    STEP 2   AnatomyConditionedUNet (diffusion backbone)
    STEP 3   ResidualDiffusion process (forward/reverse/losses)

  PART B  — Dataset, Training Loop, Evaluation
    STEP 4   Stage 2 dataset  (LDCT input + HDCT target pairs)
    STEP 5   Stage 2 training loop
    STEP 6   Evaluation + inference script

  PART C  — Integration & Final Test
    STEP 7   Integration test + full pipeline verification
```

---

## ARCHITECTURE OVERVIEW

```
Stage 2 inputs per training step:
  x_ldct  [B, 1, 256, 256]  ← Low-dose CT (noisy input)
  x_hdct  [B, 1, 256, 256]  ← High-dose CT (clean target)

Frozen Stage 1 provides anatomy conditioning:
  S   [B, 7, 256, 256]   segmentation map  (soft spatial weights)
  e_a [B, 7, 96]         anatomy embeddings (per-organ feature vectors)

Residual diffusion formulation (from DADiff.py):
  x_res = x_ldct - x_hdct               ← residual to remove
  x_t   = x_hdct + α_t·x_res + β_t·ε   ← noisy sample at timestep t
  UNet predicts: (x_res, ε)             ← pred_res_noise objective

  At inference:
    start from x_ldct + small noise
    denoise T steps → x_hdct_pred

The AnatomyConditionedUNet replaces DADiff's Dose-CLIP conditioning with:
  - S  injected via spatial FiLM modulation at every decoder scale
  - e_a injected via cross-attention in VSSD blocks
```

---

## PART A — DENOISING UNET

---

## STEP 1 — Stage 2 Config File

```
Create configs/stage2_config.yaml with the following content.
This config controls the diffusion UNet, training, and anatomy conditioning.

# ─────────────────────────────────────────────────────────────
# Stage 2: Anatomy-Conditioned Residual Diffusion
# CT Denoising: LDCT → HDCT
# ─────────────────────────────────────────────────────────────

stage1:
  checkpoint: /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage1/stage1_best.pth
  embed_dim: 96
  num_classes: 7
  freeze: true           # Stage 1 weights are never updated in Stage 2

model:
  image_size: 256
  in_channels: 1
  base_dim: 64           # UNet base channel count (doubles at each scale)
  dim_mults: [1, 2, 4, 8]  # → channels: 64, 128, 256, 512
  # VSSD settings (same as Stage 1 but lighter — diffusion UNet is deeper)
  vssd_d_state: 8
  # Anatomy conditioning dimensions
  num_classes: 7         # from Stage 1
  anatomy_embed_dim: 96  # from Stage 1 e_a

diffusion:
  timesteps: 1000
  sampling_timesteps: 100   # DDIM accelerated sampling (10× faster than DDPM)
  objective: pred_res_noise  # predict residual AND noise simultaneously
  beta_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
  ddim_eta: 0.0             # deterministic DDIM (eta=0)
  sum_scale: 0.01           # residual diffusion scale factor

training:
  batch_size: 4
  total_steps: 50000
  warmup_steps: 1000
  lr: 2.0e-4
  weight_decay: 0.0
  adam_betas: [0.9, 0.99]
  grad_clip: 1.0
  ema_decay: 0.995
  ema_update_every: 10
  log_every: 100
  val_every: 1000
  save_every: 2000
  keep_checkpoints: 3

data:
  data_root: /home/teaching/Music/Nigam_51/Project_51/data
  masks_root: /home/teaching/Music/Nigam_51/Project_51/data/masks
  num_workers: 4
  pin_memory: true
  val_split: 0.1

checkpointing:
  checkpoint_dir: /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2

loss:
  res_weight: 1.0     # weight on residual prediction loss
  noise_weight: 1.0   # weight on noise prediction loss

metrics:
  # Computed on val set every val_every steps
  # PSNR target: > 35 dB, SSIM target: > 0.92
  compute_psnr: true
  compute_ssim: true
  compute_rmse: true
```

**Verify:** `import yaml; cfg = yaml.safe_load(open('configs/stage2_config.yaml')); print(cfg['model']['base_dim'])` → prints 64
```

---

## STEP 2 — AnatomyConditionedUNet

```
Create models/anatomy_unet.py

This is the denoising backbone for Stage 2.
It is adapted from the Unet class in DADiff.py with three key changes:
  1. Dose-CLIP conditioning is REPLACED by anatomy conditioning (S, e_a)
  2. SS2D blocks are REPLACED by VSSD (bidirectional, from models/vmamba_blocks.py)
  3. Anatomy is injected at TWO levels:
       - e_a via cross-attention inside VSSDBlock (global per-organ context)
       - S   via spatial FiLM at each UNet scale (local pixel-level guidance)

═══════════════════════════════════════════════════════════════
MODULE 1 — SinusoidalPosEmb
═══════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
  """Standard sinusoidal timestep embedding. Input: [B] long. Output: [B, dim]"""
  Same as DADiff.py SinusoidalPosEmb.


═══════════════════════════════════════════════════════════════
MODULE 2 — AnatomyFiLM  (spatial FiLM conditioning from S)
═══════════════════════════════════════════════════════════════

class AnatomyFiLM(nn.Module):
  """
  Spatial FiLM modulation using segmentation map S.

  S [B, num_classes, H, W] is a soft spatial mask (sums to 1 over classes).
  We compute per-pixel scale and shift from S and apply them to feature maps.

  Architecture:
    1. seg_proj: Conv2d(num_classes, hidden, 1)  → [B, hidden, H, W]
    2. to_scale: Conv2d(hidden, out_channels, 1) → [B, C, H, W]
    3. to_shift: Conv2d(hidden, out_channels, 1) → [B, C, H, W]
    4. output: x * (1 + scale) + shift

  S must be resized to match x spatial dims using F.interpolate(mode='bilinear').

  Parameters:
    num_classes (int): 7
    out_channels (int): channel dim of the feature map being modulated
    hidden (int): 64 (intermediate projection width)

  Forward:
    x: [B, C, H, W]    feature map from UNet
    S: [B, 7, H0, W0]  segmentation map (any spatial size, will be resized)
    Returns: [B, C, H, W]
  """


═══════════════════════════════════════════════════════════════
MODULE 3 — VSSDBlock  (replaces Mamba_block from DADiff.py)
═══════════════════════════════════════════════════════════════

class VSSDBlock(nn.Module):
  """
  VSSD-based block with timestep + anatomy embedding conditioning.
  Replaces DADiff's Mamba_block (which used SS2D + adaLN-Zero + CrossAttention).

  Adapts the adaLN-Zero conditioning pattern from DADiff.Mamba_block but:
    - Uses VSSD (from models/vmamba_blocks.py) instead of SS2D
    - Anatomy e_a is injected via cross-attention after VSSD scan
    - Timestep t is injected via adaLN-Zero modulation

  Architecture:
    norm1: LayerNorm(hidden_size)
    vssd:  VSSD(d_model=hidden_size, d_state=vssd_d_state)   ← from vmamba_blocks.py
    norm2: LayerNorm(hidden_size, elementwise_affine=False)
    cross_attn: CrossAttention(query_dim=hidden_size, context_dim=num_classes*embed_dim)
    adaLN_modulation: nn.Sequential(SiLU, Linear(time_dim → 4*hidden_size))
                      → produces (shift_msa, scale_msa, shift_ca, scale_ca)

  Forward(x, t_emb, e_a):
    x:     [B, C, H, W]  NCHW feature map
    t_emb: [B, time_dim] timestep embedding
    e_a:   [B, 7, 96]    anatomy embeddings from Stage 1

    Steps:
      1. x_bhwc = x.permute(0,2,3,1)           → [B, H, W, C]
      2. shift_msa, scale_msa, shift_ca, scale_ca = adaLN_modulation(t_emb).chunk(4)
      3. x_mod = x_bhwc * (1 + scale_msa) + shift_msa   (adaLN)
      4. x_vssd = vssd(norm1(x_mod))            → [B, H, W, C]
      5. x_bhwc = x_bhwc + x_vssd              (residual)
      6. context = e_a.reshape(B, 7*96)        → [B, 672] for cross-attn
         (or use e_a.flatten(1) directly)
      7. x_ca_in = x_bhwc.permute(0,3,1,2)    → [B, C, H, W]
      8. x_ca_in_mod = x_ca_in * (1 + scale_ca.view) + shift_ca.view
      9. Apply cross_attn: queries from x, keys/values from context
     10. Return x_bhwc.permute(0,3,1,2)        → [B, C, H, W]

  Initialize adaLN_modulation final layer weights/bias to zero (adaLN-Zero).

  Parameters:
    hidden_size (int): channel dim C
    time_dim (int):    timestep embedding dim (base_dim * 4)
    num_classes (int): 7
    embed_dim (int):   96
    vssd_d_state (int): 8
  """


═══════════════════════════════════════════════════════════════
MODULE 4 — AnatomyConditionedUNet   (main backbone)
═══════════════════════════════════════════════════════════════

class AnatomyConditionedUNet(nn.Module):
  """
  Denoising UNet for Stage 2 residual diffusion.
  Adapted from DADiff.Unet but with anatomy conditioning replacing Dose-CLIP.

  Input:
    x_in:  [B, 2, H, W]   concat of (x_noisy, x_ldct)  — same as DADiff condition=True
    time:  [B]             diffusion timestep indices
    S:     [B, 7, H, W]   segmentation map from frozen Stage 1
    e_a:   [B, 7, 96]     anatomy embeddings from frozen Stage 1

  Output:
    pred_res:   [B, 1, H, W]   predicted residual  (x_ldct - x_hdct)
    pred_noise: [B, 1, H, W]   predicted noise

  Architecture:
    Encoder (downs):  4 scales, channels = [64, 128, 256, 512]
      Each scale:
        ResnetBlock(dim_in → dim_in)
        VSSDBlock(dim_in, time_dim, num_classes=7, embed_dim=96)
        AnatomyFiLM(num_classes=7, out_channels=dim_in)
        Downsample(dim_in → dim_out)   [except last scale: Conv2d 3×3]

    Bottleneck (mid):
        ResnetBlock(512 → 512)
        VSSDBlock(512, time_dim)
        ResnetBlock(512 → 512)

    Decoder (ups):  4 scales (reversed)
        ResnetBlock(dim_out + dim_in → dim_out)   [skip connection concat]
        VSSDBlock(dim_out, time_dim)
        AnatomyFiLM(num_classes=7, out_channels=dim_out)
        Upsample(dim_out → dim_in)   [except last scale: Conv2d 3×3]

    Output heads (TWO heads for pred_res_noise objective):
        final_res_block:   ResnetBlock(base_dim*2 → base_dim)
        res_head:          Conv2d(base_dim → 1, 1)   → pred_res
        noise_head:        Conv2d(base_dim → 1, 1)   → pred_noise

  Timestep embedding:
    time_mlp: SinusoidalPosEmb(base_dim) → Linear → GELU → Linear → [B, base_dim*4]

  init_conv: Conv2d(2, base_dim, 7, padding=3)   [input: noisy + ldct concatenated]

  Note: DADiff uses self.condition=True and concatenates x_input to x_noisy.
  We follow the same pattern: x_in = cat([x_t, x_ldct], dim=1) → 2 channels.

  Parameters:
    base_dim (int):      64
    dim_mults (tuple):   (1, 2, 4, 8)
    in_channels (int):   1
    num_classes (int):   7
    anatomy_embed_dim (int): 96
    vssd_d_state (int):  8
  """
  def __init__(self, base_dim=64, dim_mults=(1,2,4,8),
               in_channels=1, num_classes=7,
               anatomy_embed_dim=96, vssd_d_state=8):
      ...

  def forward(self, x_in, time, S, e_a):
      """
      x_in: [B, 2, H, W]   (x_t concat x_ldct)
      time: [B]
      S:    [B, 7, H, W]
      e_a:  [B, 7, 96]
      Returns: (pred_res [B,1,H,W], pred_noise [B,1,H,W])
      """
      ...


Add self-test __main__ block:
  model = AnatomyConditionedUNet()
  x_in  = torch.randn(1, 2, 256, 256)
  t     = torch.randint(0, 1000, (1,))
  S     = torch.softmax(torch.randn(1, 7, 256, 256), dim=1)
  e_a   = torch.randn(1, 7, 96)

  pred_res, pred_noise = model(x_in, t, S, e_a)

  assert pred_res.shape   == (1, 1, 256, 256), f"pred_res: {pred_res.shape}"
  assert pred_noise.shape == (1, 1, 256, 256), f"pred_noise: {pred_noise.shape}"
  assert torch.isfinite(pred_res).all()
  assert torch.isfinite(pred_noise).all()
  print(f"AnatomyConditionedUNet: PASSED")
  print(f"  pred_res   : {list(pred_res.shape)}")
  print(f"  pred_noise : {list(pred_noise.shape)}")
  n_params = sum(p.numel() for p in model.parameters()) / 1e6
  print(f"  Parameters : {n_params:.1f}M")
```

**Verify:** `python models/anatomy_unet.py` → prints `AnatomyConditionedUNet: PASSED`

---

## STEP 3 — ResidualDiffusion Process

```
Create models/residual_diffusion.py

Implements the forward diffusion process, reverse sampling, and training losses.
Adapted directly from DADiff.ResidualDiffusion. Read DADiff.py carefully —
the math here mirrors lines 908-1499 of that file.

KEY FORMULAS from DADiff (copy these exactly):

  Forward process (q_sample):
    x_res = x_ldct - x_hdct                                  ← residual
    x_t   = x_hdct + alphas_cumsum[t] * x_res
                   + betas_cumsum[t]  * noise                ← noisy sample
    (DADiff line 1382-1388)

  Training objective (pred_res_noise):
    Model predicts (pred_res, pred_noise) from (x_t, x_ldct, t, S, e_a)
    loss = L1(pred_res, x_res) + L1(pred_noise, noise)
    (DADiff lines 1444-1481)

  DDIM reverse step:
    img = img - alpha * pred_res + sigma2.sqrt() * noise_term
    (DADiff lines 1343-1349)

  Noise schedule (linear, convert_to_ddim=True path from DADiff lines 948-976):
    betas = linspace(0.0001, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = cumprod(alphas)
    alphas_cumsum  = 1 - alphas_cumprod ** 0.5
    betas2_cumsum  = 1 - alphas_cumprod
    betas_cumsum   = sqrt(betas2_cumsum)

═══════════════════════════════════════════════════════════════
CLASS — ResidualDiffusion
═══════════════════════════════════════════════════════════════

class ResidualDiffusion(nn.Module):
  """
  Residual diffusion process for CT denoising.

  Wraps AnatomyConditionedUNet with:
    - Forward noising (q_sample)
    - Loss computation (p_losses) — pred_res_noise objective
    - DDIM reverse sampling (ddim_sample)
    - DDPM reverse sampling (p_sample_loop) for reference

  The anatomy conditioning (S, e_a) comes from frozen Stage 1.
  It is passed THROUGH this module to the UNet at every forward call.

  Parameters:
    model (AnatomyConditionedUNet): the denoising UNet
    image_size (int): 256
    timesteps (int): 1000
    sampling_timesteps (int): 100  (DDIM steps, much faster)
    ddim_eta (float): 0.0  (deterministic)
    sum_scale (float): 0.01
    loss_res_weight (float): 1.0
    loss_noise_weight (float): 1.0

  Registered buffers (all from noise schedule):
    alphas, alphas_cumsum, betas2, betas2_cumsum,
    betas_cumsum, posterior_variance, posterior_mean_coef1/2/3

  Methods:

  q_sample(x_hdct, x_res, t, noise=None) → x_t
    Add noise at timestep t. Follows DADiff lines 1382-1388.

  p_losses(x_hdct, x_ldct, S, e_a) → dict
    Sample random t, compute x_t, predict (pred_res, pred_noise),
    compute L1 losses.
    Returns: {'loss': total_loss, 'loss_res': L_res, 'loss_noise': L_noise}

  ddim_sample(x_ldct, S, e_a, shape) → x_hdct_pred
    Run DDIM reverse chain from x_ldct + noise → x_hdct.
    Use sampling_timesteps steps (100 by default).
    Follows DADiff lines 1276-1365.
    Returns: [B, 1, H, W] denoised image

  forward(x_hdct, x_ldct, S, e_a) → dict
    Calls p_losses. Used during training.

  predict(x_ldct, S, e_a) → [B, 1, H, W]
    Calls ddim_sample. Used during inference.
    Normalizes input to [-1,1] before diffusion, unnormalizes output.

  Helper methods (adapt from DADiff):
    predict_start_from_res_noise(x_t, t, x_res, noise)
    predict_noise_from_res(x_t, t, x_input, pred_res)
    model_predictions(x_ldct, x_t, t, S, e_a)
    q_posterior(pred_res, x_start, x_t, t)

  Normalization:
    normalize:   img * 2 - 1  (to [-1, 1])
    unnormalize: (img + 1) * 0.5  (back to [0, 1])
    Apply normalize to both x_hdct and x_ldct before diffusion.
    Apply unnormalize to final output.
  """

Add self-test __main__ block:
  from models.anatomy_unet import AnatomyConditionedUNet

  unet = AnatomyConditionedUNet()
  diff = ResidualDiffusion(model=unet, image_size=256,
                           sampling_timesteps=10)  # 10 steps for fast test

  B = 1
  x_hdct = torch.rand(B, 1, 256, 256)
  x_ldct = torch.rand(B, 1, 256, 256)
  S      = torch.softmax(torch.randn(B, 7, 256, 256), dim=1)
  e_a    = torch.randn(B, 7, 96)

  # Training forward
  out = diff(x_hdct, x_ldct, S, e_a)
  assert 'loss' in out and torch.isfinite(out['loss'])
  print(f"  p_losses: loss={out['loss'].item():.4f}  ✓")

  # Inference (DDIM 10 steps)
  with torch.no_grad():
      x_pred = diff.predict(x_ldct, S, e_a)
  assert x_pred.shape == (B, 1, 256, 256)
  assert torch.isfinite(x_pred).all()
  print(f"  ddim_sample: shape={list(x_pred.shape)}  ✓")
  print("ResidualDiffusion: PASSED")
```

**Verify:** `python models/residual_diffusion.py` → prints `ResidualDiffusion: PASSED`

---

## PART B — DATASET, TRAINING, EVALUATION

---

## STEP 4 — Stage 2 Dataset

```
Create data/stage2_dataset.py

Stage 2 needs PAIRED slices: LDCT (input) and HDCT (target).
Unlike Stage 1 which only used HDCT, Stage 2 uses both.

class CTDenoisingDataset(torch.utils.data.Dataset):
  """
  Paired LDCT / HDCT dataset for Stage 2 denoising training.

  Directory structure expected:
    data_root/
      {PATIENT}/
        HDCT/   ← clean target, .dcm or .npy files
        LDCT/   ← noisy input, .dcm or .npy files
      masks/
        {PATIENT}/
          {SLICE_IDX:04d}.npy   ← segmentation masks from Stage 1 Part 0

  Each __getitem__ returns a dict:
    {
      'ldct':   [1, H, W]  float32 tensor  ← noisy input
      'hdct':   [1, H, W]  float32 tensor  ← clean target
      'mask':   [H, W]     int8 tensor     ← segmentation label (for reference)
      'patient': str
      'slice_idx': int
    }

  Data loading:
    - Reuse the same HDCT/LDCT loading logic from data/dataset.py
    - MUST load BOTH HDCT and LDCT slices for the same patient + slice index
    - Normalise slices to [0, 1] using the same window/level as Stage 1:
        window: [-1000, 3000] HU range → divide by 4000 and shift
        OR: use the min/max normalisation from data/dataset.py (be consistent)
    - Skip slices where the mask file does not exist

  Splits:
    Use the same patient-level train/val/test split as Stage 1.
    Call create_stage2_dataloaders(data_root, masks_root, batch_size, val_split=0.1)
    which returns (train_loader, val_loader, test_loader).

  DummyCTDenoisingDataset:
    Returns random tensors of the right shapes.
    Used for smoke testing the training loop without real data.
    Implement identically to data/dataset.py DummyCTDataset pattern.
```

**Verify:**
```python
from data.stage2_dataset import CTDenoisingDataset, DummyCTDenoisingDataset
ds = DummyCTDenoisingDataset(n_samples=8)
batch = ds[0]
assert batch['ldct'].shape == (1, 256, 256)
assert batch['hdct'].shape == (1, 256, 256)
print("Stage2 Dataset: PASSED")
```

---

## STEP 5 — Stage 2 Training Loop

```
Create training/train_stage2.py

Implements the complete Stage 2 training loop.
Read training/train_stage1.py for the logging, checkpoint, and resume patterns —
replicate them exactly (same logger, same find_latest_checkpoint, same NaN guards).

def train_stage2(
    config_path: str,
    data_root:   str  = '/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root:  str  = '/home/teaching/Music/Nigam_51/Project_51/data/masks',
    checkpoint_dir: str = '/home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2',
    resume_from: str  = None,
    use_dummy_data: bool = False,
    max_steps: int = None,
):

TRAINING STEP (each iteration):
  1. Load batch: ldct, hdct (paired)
  2. Get anatomy conditioning from FROZEN Stage 1:
       with torch.no_grad():
           stage1_out = stage1_model(hdct, return_byol=False)
           S   = stage1_out['S']    # [B, 7, 256, 256]
           e_a = stage1_out['e_a']  # [B, 7, 96]
     NOTE: Use HDCT (clean) for Stage 1 inference at training time,
           because Stage 1 was trained on HDCT.
           At inference time (LDCT denoising), Stage 1 will receive LDCT.
  3. Forward diffusion:
       out = diffusion(hdct, ldct, S, e_a)
       loss = out['loss_res'] * cfg_res_weight + out['loss_noise'] * cfg_noise_weight
  4. NaN guard BEFORE backward (same pattern as Stage 1)
  5. loss.backward()
  6. clip_grad_norm_(model.parameters(), max_norm=1.0)
  7. optimizer.step()
  8. EMA update: ema.update() every ema_update_every steps

OPTIMIZER: Adam(lr=2e-4, betas=(0.9, 0.99))
LR SCHEDULE: linear warmup (1000 steps) then cosine decay

LOGGING (every 100 steps):
  step, epoch, loss_res, loss_noise, total_loss, lr, s/step

VALIDATION (every 1000 steps):
  Run DDIM sampling on up to 10 val batches (no_grad + EMA model)
  Compute: PSNR, SSIM, RMSE
  Save checkpoint if PSNR improved

  Helper functions (implement in training/train_stage2.py):
    compute_psnr(pred, target) → float   (peak signal-to-noise ratio, dB)
    compute_ssim(pred, target) → float   (structural similarity)
    compute_rmse(pred, target) → float   (root mean square error)

CHECKPOINT FORMAT:
    {
      'step': step,
      'epoch': epoch,
      'model_state_dict': model.state_dict(),    ← diffusion model (not EMA)
      'ema_state_dict': ema.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'best_psnr': best_psnr,
      'config': cfg,
    }
  Save:
    stage2_step_{N}.pth    every save_every steps
    stage2_best.pth        whenever val PSNR improves

STAGE 1 LOADING (at top of train_stage2):
    from models.stage1 import load_stage1_frozen
    stage1 = load_stage1_frozen(cfg['stage1']['checkpoint'], device=device)
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)
    logger.info(f"Stage 1 loaded and frozen from {cfg['stage1']['checkpoint']}")

EMA MODEL (from ema_pytorch package):
    from ema_pytorch import EMA
    ema = EMA(diffusion_model, beta=cfg['training']['ema_decay'],
              update_every=cfg['training']['ema_update_every'])
    Use ema.ema_model for validation inference.

LOG FORMAT examples:
  [2024-01-16 09:00:01] Training start | device=cuda | steps=50000 | stage1=frozen
  [2024-01-16 09:01:45] Step   100 | ep=0 | L_res=0.0821 | L_noise=0.3142 | lr=2.0e-05
  [2024-01-16 10:00:00] Step  1000 | Val PSNR=28.42 dB | SSIM=0.847 | RMSE=0.0231
  [2024-01-16 10:00:01] Checkpoint saved → stage2_best.pth (PSNR=28.42)

ALSO copy the following from train_stage1.py:
  - setup_logger()
  - find_latest_checkpoint()
  - check_model_weights()
  - cleanup_old_checkpoints()
  - infinite_loader()
  - NaN guard pattern (nan_count, 3× NaN → raise RuntimeError)

if __name__ == '__main__':
  # Smoke test: 10 steps dummy data
  import tempfile
  with tempfile.TemporaryDirectory() as tmpdir:
      train_stage2(config_path='configs/stage2_config.yaml',
                   use_dummy_data=True, max_steps=10,
                   checkpoint_dir=tmpdir)
  print("Stage 2 smoke test: PASSED")
```

**Verify:** `python training/train_stage2.py` → prints `Stage 2 smoke test: PASSED`

---

## STEP 6 — Evaluation + Inference Script

```
Create utils/evaluate_stage2.py

This script runs full evaluation on the TEST set and generates denoised images.
It is run AFTER training to measure final performance.

Functions to implement:

─────────────────────────────────────────────────────────────
evaluate_stage2(
    checkpoint_path: str,
    config_path: str,
    data_root: str,
    masks_root: str,
    output_dir: str,
    sampling_timesteps: int = 100,
    save_images: bool = True,
):
─────────────────────────────────────────────────────────────

  1. Load config + build models
  2. Load Stage 1 frozen
  3. Load Stage 2 from checkpoint (use EMA weights if available)
  4. Run on test_loader (all test patients)

  For each test batch:
    a. Get S, e_a from Stage 1 with LDCT input:
         with torch.no_grad():
             out = stage1(ldct, return_byol=False)  ← LDCT at inference!
             S   = out['S']
             e_a = out['e_a']

    b. Denoise:
         x_pred = diffusion.predict(ldct, S, e_a)

    c. Compute metrics:
         psnr = compute_psnr(x_pred, hdct)
         ssim = compute_ssim(x_pred, hdct)
         rmse = compute_rmse(x_pred, hdct)

    d. If save_images: save as .npy to output_dir/{patient}/{slice:04d}.npy

  Print summary:
    Mean PSNR / SSIM / RMSE across all test slices
    Per-patient breakdown

─────────────────────────────────────────────────────────────
denoise_single_slice(
    ldct_slice: np.ndarray,   [H, W] float32 CT slice
    stage1_model,
    diffusion_model,
    device: str = 'cuda',
    sampling_timesteps: int = 20,   # fast inference
) -> np.ndarray:  [H, W]
─────────────────────────────────────────────────────────────

  Convenience function for denoising a single 2D slice.
  Used by researchers who want to denoise one image at a time.

  Steps:
    1. Normalize slice to [0, 1]
    2. Add batch dim → [1, 1, H, W]
    3. Stage 1: get S, e_a
    4. Diffusion: predict → [1, 1, H, W]
    5. Remove batch dim, return as numpy

Add __main__ block for quick evaluation:
  python utils/evaluate_stage2.py \
    --checkpoint /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2/stage2_best.pth \
    --config configs/stage2_config.yaml \
    --output /home/teaching/Music/Nigam_51/Project_51/results/stage2

Use argparse for all paths.
```

**Verify:** `python utils/evaluate_stage2.py --help` → shows all arguments

---

## PART C — INTEGRATION TEST

---

## STEP 7 — Integration Test + Full Pipeline

```
Create tests/test_stage2_pipeline.py

Run a complete end-to-end test of the Stage 2 pipeline using dummy data.
All tests must pass before real training begins.

import torch
import tempfile
import os
from models.anatomy_unet   import AnatomyConditionedUNet
from models.residual_diffusion import ResidualDiffusion
from models.stage1         import Stage1Model
from data.stage2_dataset   import DummyCTDenoisingDataset

B = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("Stage 2 Integration Tests")
print("=" * 60)

─────────────────────────────────────────────────────────────
TEST 1 — UNet shape flow
─────────────────────────────────────────────────────────────
unet = AnatomyConditionedUNet().to(device)
x_in = torch.randn(B, 2, 256, 256, device=device)
t    = torch.randint(0, 1000, (B,), device=device)
S    = torch.softmax(torch.randn(B, 7, 256, 256, device=device), dim=1)
e_a  = torch.randn(B, 7, 96, device=device)
pred_res, pred_noise = unet(x_in, t, S, e_a)
assert pred_res.shape   == (B, 1, 256, 256)
assert pred_noise.shape == (B, 1, 256, 256)
print("  PASSED ✓  UNet shape flow")

─────────────────────────────────────────────────────────────
TEST 2 — Diffusion loss computation
─────────────────────────────────────────────────────────────
diff = ResidualDiffusion(model=unet, image_size=256, sampling_timesteps=5)
x_hdct = torch.rand(B, 1, 256, 256, device=device)
x_ldct = torch.rand(B, 1, 256, 256, device=device)
out = diff(x_hdct, x_ldct, S, e_a)
assert torch.isfinite(out['loss'])
assert out['loss_res'].item() >= 0
assert out['loss_noise'].item() >= 0
print(f"  PASSED ✓  Loss computation (loss={out['loss'].item():.4f})")

─────────────────────────────────────────────────────────────
TEST 3 — DDIM sampling (5 steps for speed)
─────────────────────────────────────────────────────────────
with torch.no_grad():
    x_pred = diff.predict(x_ldct, S, e_a)
assert x_pred.shape == (B, 1, 256, 256)
assert torch.isfinite(x_pred).all()
print(f"  PASSED ✓  DDIM sampling shape={list(x_pred.shape)}")

─────────────────────────────────────────────────────────────
TEST 4 — Stage 1 → Stage 2 conditioning flow
─────────────────────────────────────────────────────────────
stage1 = Stage1Model().to(device)
stage1.eval()
for p in stage1.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    s1_out = stage1(x_hdct, return_byol=False)
    S_real   = s1_out['S']    # [B, 7, 256, 256]
    e_a_real = s1_out['e_a']  # [B, 7, 96]

out2 = diff(x_hdct, x_ldct, S_real, e_a_real)
assert torch.isfinite(out2['loss'])
print(f"  PASSED ✓  Stage1 → Stage2 conditioning flow")

─────────────────────────────────────────────────────────────
TEST 5 — Gradient flow (Stage 1 frozen, Stage 2 trains)
─────────────────────────────────────────────────────────────
unet2 = AnatomyConditionedUNet().to(device)
diff2 = ResidualDiffusion(model=unet2, image_size=256, sampling_timesteps=5)
out3  = diff2(x_hdct, x_ldct, S_real.detach(), e_a_real.detach())
out3['loss'].backward()

# Stage 2 has gradients
has_grad = any(p.grad is not None for p in unet2.parameters())
assert has_grad, "Stage 2 UNet should have gradients"

# Stage 1 has no gradients
stage1_has_grad = any(p.grad is not None for p in stage1.parameters())
assert not stage1_has_grad, "Stage 1 should NOT have gradients"
print(f"  PASSED ✓  Gradient flow (Stage1 frozen)")

─────────────────────────────────────────────────────────────
TEST 6 — Dataset compatibility
─────────────────────────────────────────────────────────────
ds    = DummyCTDenoisingDataset(n_samples=4)
batch = ds[0]
assert batch['ldct'].shape == (1, 256, 256)
assert batch['hdct'].shape == (1, 256, 256)
print(f"  PASSED ✓  Dataset compatibility")

─────────────────────────────────────────────────────────────
Print summary
─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("All Stage 2 integration tests PASSED")
print("=" * 60)
print()
print("Ready to start Stage 2 training:")
print("  python -c \"")
print("  from training.train_stage2 import train_stage2")
print("  train_stage2(config_path='configs/stage2_config.yaml')")
print("  \"")
```

**Verify:** `python tests/test_stage2_pipeline.py` → All 6 tests PASSED

---

## FINAL CHECKLIST

```
Before starting real Stage 2 training, confirm ALL of these:

python models/anatomy_unet.py          ← AnatomyConditionedUNet PASSED
python models/residual_diffusion.py    ← ResidualDiffusion PASSED
python training/train_stage2.py        ← smoke test 10 steps PASSED
python tests/test_stage2_pipeline.py   ← all 6 integration tests PASSED

Checkpoint exists:
  ls /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage1/stage1_best.pth

Create checkpoint dir:
  mkdir -p /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2
```

---

## START STAGE 2 TRAINING

```bash
mkdir -p /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2

python -c "
from training.train_stage2 import train_stage2
train_stage2(
    config_path='configs/stage2_config.yaml',
    data_root='/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root='/home/teaching/Music/Nigam_51/Project_51/data/masks',
    checkpoint_dir='/home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2',
)
"
```

Monitor:
```bash
tail -f /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage2/training.log
```

## EXPECTED METRICS

| Steps  | PSNR (dB) | SSIM  | Status |
|--------|-----------|-------|--------|
| 1,000  | 25–28     | 0.75–0.85 | Learning noise structure |
| 5,000  | 28–31     | 0.85–0.90 | Anatomy conditioning kicking in |
| 15,000 | 31–34     | 0.90–0.93 | Good denoising |
| 30,000 | 33–36     | 0.92–0.95 | Strong — anatomy-aware |
| **50,000** | **35–38** | **0.93–0.96** | ✓ Target |

**Target for publication-quality results: PSNR ≥ 35 dB, SSIM ≥ 0.92**
