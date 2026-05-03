# NaN Prevention Implementation — Complete Summary

## Changes Made

### 1. training/train_stage1.py

#### Added helper function (line ~105-120):
```python
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
```

#### Modified training loop (line ~650-750):
- Added `nan_count = 0` initialization before loop
- Added NaN guard BEFORE backward pass:
  ```python
  # ── NaN guard BEFORE backward ──────────────────────────────
  if not torch.isfinite(loss):
      logger.warning(f"Step {step} | Non-finite loss={loss.item():.6f} — skipping batch...")
      optimizer.zero_grad()
      nan_count += 1
      if nan_count >= 3:
          logger.error("3 consecutive NaN losses — weights are corrupted...")
          raise RuntimeError("NaN loss: weights corrupted.")
      continue
  nan_count = 0  # reset on healthy step
  ```

- Replaced simple gradient clipping with enhanced version:
  ```python
  # Clip gradients — prevents exploding gradients from bad batches
  grad_norm = torch.nn.utils.clip_grad_norm_(
      model.parameters(), max_norm=1.0
  )
  
  # Log if grad norm is very high (early warning of instability)
  if grad_norm > 10.0:
      logger.warning(f"Step {step} | High grad norm={grad_norm:.2f} — clipped to 1.0")
  ```

- Added weight corruption check in logging block:
  ```python
  if step % log_every == 0:
      # ... logging ...
      
      # Check for weight corruption every 100 steps
      if check_model_weights(model, step, logger):
          raise RuntimeError(f"Corrupted weights at step {step}. Resume from checkpoint.")
  ```

### 2. models/vmamba_blocks.py

#### Modified _scan_chunk_fwd method (line ~340-360):
- Added delta clamping (line ~344):
  ```python
  delta = F.softplus(self.delta_proj(xc))
  delta = delta.clamp(max=10.0)  # Prevent delta from becoming too large
  ```

- Added A_cumsum clamping (line ~357-360):
  ```python
  A_cumsum = torch.cumsum(log_dA, dim=1)
  
  # Clamp A_cumsum to prevent exp() overflow in long sequences
  # max=0.0 because log(dA) should always be ≤ 0 (dA ≤ 1, decay)
  # min=-20.0 prevents underflow (exp(-20) ≈ 2e-9, effectively zero)
  A_cumsum = A_cumsum.clamp(min=-20.0, max=0.0)
  ```

### 3. configs/stage1_config.yaml

#### Changed hyperparameters (line ~115-117):
- **learning_rate**: `1.0e-4` → `3.0e-5` (3x lower for stability)
- **warmup_steps**: `500` → `2000` (4x longer warmup for safer LR ramp)

## Why These Changes Work

### Gradient Clipping (FIX 1)
- **Problem**: Mamba's recurrent state updates produce very large gradients (often 100-10000x normal)
- **Solution**: Clip gradient norm to 1.0 before optimizer.step()
- **Detection**: Log warning when grad_norm > 10.0 to identify problem batches

### NaN Guard (FIX 2)
- **Problem**: Loss can become NaN/Inf during forward pass (bad data, numerical overflow)
- **Solution**: Check loss BEFORE backward() and skip bad batches
- **Safety**: Stop training after 3 consecutive NaN losses (weights are corrupted)

### Weight Check (FIX 3)
- **Problem**: Weights can become NaN/Inf even after gradient clipping (sign of critical instability)
- **Solution**: Check all parameters every 100 steps
- **Impact**: Early detection prevents wasting GPU hours on corrupted training

### VSSD Clamping (FIX 4)
- **Problem**: State space cumsum can explode for long sequences (H×W = 262K tokens)
  - exp(A_cumsum) with A_cumsum ≈ -500 → overflow
  - Large exponentials generate huge gradients on backprop
- **Solution**: 
  - Clamp delta (step size) to [0, 10] to prevent dA overflow
  - Clamp A_cumsum to [-20, 0] so exp() stays in safe range [2e-9, 1]

### Lower LR + Longer Warmup (CONFIG CHANGE)
- **Problem**: Learning rate 1e-4 is too aggressive for Mamba networks with large gradients
- **Solution**: 
  - Reduce peak LR to 3e-5 (3x lower)
  - Extend warmup to 2000 steps (slow burn-in = safer)

## Expected Behavior

### Good Signs (training is stable):
```
[Step 100] High grad norm=523.45 — clipped to 1.0    ← Normal, clipping is working
[Step 200] Step 200 | ep=0 | L_seg=4.2341 | lr=1.00e-05 | 2.65s/step
[Step 500] ★ New best val Dice = 0.44               ← Improving
[Step 5000] Step 5000 | ★ New best val Dice = 0.58  ← Still improving
```

### Warning Signs (data issues, not critical):
```
[Step 512] Non-finite loss=nan — skipping batch, zeroing gradients
[Step 678] Non-finite loss=nan — skipping batch, zeroing gradients
```
→ Happens rarely (< 1 per 500 steps), probably bad data batch
→ Monitor but not critical if infrequent

### Fatal Errors (stop immediately, resume from checkpoint):
```
RuntimeError: 3 consecutive NaN losses — weights are corrupted.
RuntimeError: Corrupted weights at step 5000. Resume from checkpoint.
```
→ Weights are corrupted
→ Resume from latest checkpoint

## Testing

### Smoke Test (10 steps on dummy data):
```bash
python3 training/train_stage1.py
```
Expected: `Smoke test: PASSED` + high grad norm warnings

### First 500 Real Steps:
Check:
- L_seg stays in range [0.3, 10.0]
- High grad norm warnings appear frequently (normal)
- No NaN losses or weight corruption

### Verify Changes:
```bash
grep -n "High grad norm" training/train_stage1.py      # Line ~738
grep -n "def check_model_weights" training/train_stage1.py  # Line ~105
grep -n "A_cumsum.clamp" models/vmamba_blocks.py       # Line ~360
grep -n "delta.clamp" models/vmamba_blocks.py          # Line ~344
```

## Key Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `training/train_stage1.py` | NaN guard, grad clipping, weight check | ~105-120, ~640-750 |
| `models/vmamba_blocks.py` | A_cumsum & delta clamping | ~344, ~360 |
| `configs/stage1_config.yaml` | Lower LR, longer warmup | ~115-117 |

## References

- **State Space Models**: Gu et al. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752)
- **Gradient Clipping**: Pascanu et al. "On the difficulty of training RNNs" (ICML 2013)
- **Mamba in Vision**: Liu et al. "Vision Mamba: Efficient Visual Representation Learning with State Space Model" (arXiv:2401.09417)
