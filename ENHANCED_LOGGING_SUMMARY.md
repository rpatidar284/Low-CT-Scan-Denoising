# Enhanced Logging Summary

## Overview
The training logger has been enhanced to display all critical metrics in the log file, enabling better monitoring and debugging of the Stage 1 VM-UNet training process.

## Periodic Logging (every 100 steps)

### Format
```
Step   100/30000 (  0.3%) | ep=  0 | L_seg=7.8007 | L_byol=-- | grad_norm=0.85 | lr=2.00e-05 | 2.67s/step
```

### Metrics Displayed

| Metric | Meaning | Range | Notes |
|--------|---------|-------|-------|
| `Step X/TOTAL (%)` | Current step / total steps | 0-100% | Progress towards completion |
| `ep=` | Current epoch | 0+ | Resets when cycling through data |
| `L_seg=` | Segmentation loss (CE + Dice) | 0+ | Should decrease over time |
| `L_byol=` | BYOL auxiliary loss | 0+ | Shows "--" when BYOL inactive (epochs 0-2) |
| `grad_norm=` | Gradient norm before clipping | 0+ | >10.0 triggers warning; clipped to 1.0 |
| `lr=` | Learning rate at current step | varies | Follows warmup + cosine schedule |
| `s/step` | Seconds per training step | varies | Should stabilize after warmup |

### What This Tells You

**Healthy Training:**
- `L_seg` steadily decreases from ~8.0 to <0.5
- `grad_norm` stays mostly in range 0.5-2.0
- Progress % increases linearly
- `s/step` stabilizes around 2-3 seconds

**Warning Signs:**
- `grad_norm > 10.0` → High gradient activity, may need investigation
- `L_seg` spikes or plateaus → Learning rate or data quality issue
- `s/step` increasing → GPU memory pressure or computational bottleneck

---

## Validation Logging (every 500 steps by default)

### Format
```
Step   500 | Val Dice → bg=0.98 liver=0.52 kidney=0.42 vessel=0.35 lung=0.65 bone=0.28 soft=0.40 | mean=0.47
```

### Metrics Displayed

All 7 class Dice coefficients:

| Class | Key | Abbreviation | Typical Range | Notes |
|-------|-----|--------------|---|-------|
| Background | `background` | `bg=` | 0.8-1.0 | Should be high (large background area) |
| Liver/Spleen | `liver_spleen` | `liver=` | 0.3-0.7 | Mixed class, harder to segment |
| Kidney | `kidney` | `kidney=` | 0.2-0.6 | Gets class weight boost (2.0) |
| Vessel | `vessel` | `vessel=` | 0.1-0.4 | Very small, challenging class |
| Lung | `lung` | `lung=` | 0.3-0.7 | Relatively easy, large connected region |
| Bone | `bone` | `bone=` | 0.1-0.5 | Small scattered regions |
| Soft Tissue | `soft_tissue` | `soft=` | 0.2-0.5 | Rare, challenging class |
| **Mean** | `mean` | After `\|` | 0.4-0.7 | Determines checkpoint saving |

### What This Tells You

**Balanced Learning:**
- All organs showing improvement over time
- Mean Dice steadily increasing
- No single class drastically underperforming

**Imbalance Issues (pre-class weights):**
- `bg=` extremely high, organs low → Background dominates
- `kidney=` much lower than others → Class imbalance
- `vessel=` or `soft=` near 0.0 → Underfitting rare classes

**With Class Weights Applied:**
- `kidney=` should improve 2-8% (weight=2.0)
- `vessel=` and `soft=` may show modest gains
- `bg=` may slightly decrease (weight=0.1)

---

## Example Log File Output

```
2024-01-15 10:23:45 INFO     Starting training…
2024-01-15 10:23:48 INFO     Loading checkpoint: stage1_step_5000.pth
2024-01-15 10:23:52 INFO     Step  100/30000 (  0.3%) | ep=  0 | L_seg=7.8007 | L_byol=-- | grad_norm=1.23 | lr=2.00e-05 | 2.67s/step
2024-01-15 10:24:15 INFO     Step  200/30000 (  0.7%) | ep=  0 | L_seg=7.1234 | L_byol=-- | grad_norm=0.92 | lr=4.00e-05 | 2.61s/step
2024-01-15 10:24:38 INFO     Step  300/30000 (  1.0%) | ep=  0 | L_seg=6.8901 | L_byol=-- | grad_norm=0.85 | lr=6.00e-05 | 2.64s/step
2024-01-15 10:25:00 INFO     Step  400/30000 (  1.3%) | ep=  0 | L_seg=6.5234 | L_byol=-- | grad_norm=0.78 | lr=8.00e-05 | 2.59s/step
2024-01-15 10:25:22 INFO     Step  500/30000 (  1.7%) | ep=  0 | L_seg=6.2001 | L_byol=-- | grad_norm=0.91 | lr=1.00e-04 | 2.63s/step
2024-01-15 10:25:22 INFO     Step   500 | Val Dice → bg=0.98 liver=0.12 kidney=0.08 vessel=0.02 lung=0.18 bone=0.05 soft=0.03 | mean=0.21
2024-01-15 10:25:44 INFO     Step  600/30000 (  2.0%) | ep=  0 | L_seg=5.8234 | L_byol=-- | grad_norm=0.87 | lr=1.00e-04 | 2.65s/step
…
2024-01-15 10:47:23 INFO     Step 2600/30000 (  8.7%) | ep=  1 | L_seg=0.6782 | L_byol=0.0234 | grad_norm=15.42 | lr=5.00e-05 | 2.71s/step
2024-01-15 10:47:23 WARNING  Step 2600 | High grad norm=15.42 — clipped to 1.0
2024-01-15 10:47:24 ERROR    NaN loss detected at step 2600! Skipping batch (2/3 NaN batches)…
```

---

## NaN Prevention Integration

Enhanced logging works in tandem with NaN prevention:

1. **High Grad Norm Detection**: If `grad_norm > 10.0`, a warning is logged immediately
   - Indicates potential gradient explosion
   - Value is clipped to 1.0 automatically
   - May precede NaN loss issues

2. **NaN Guard Before Backward**: 
   - If loss is NaN, logged as ERROR with batch count
   - Training continues if <3 consecutive NaN batches
   - Weight corruption check triggered on NaN

3. **Weight Corruption Check** (every 100 steps):
   - Scans model parameters for NaN/Inf values
   - Raises RuntimeError if found (forces checkpoint resume)

---

## Class Weights Effects on Logging

With class weights `[0.1, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0]`:

**Before Class Weights:**
- `Step 500 | Val Dice → bg=0.98 liver=0.12 kidney=0.08 vessel=0.02 lung=0.18 bone=0.05 soft=0.03 | mean=0.21`
- Kidney Dice very low, background dominant

**After Class Weights (expected):**
- `Step 500 | Val Dice → bg=0.95 liver=0.18 kidney=0.15 vessel=0.04 lung=0.22 bone=0.07 soft=0.05 | mean=0.27`
- Kidney Dice improved ~2x (from 0.08 to 0.15)
- Overall mean improved ~28%
- Background slightly lower (expected due to weight=0.1)

---

## How to Monitor Training Effectively

### First 500 steps (Warmup phase):
- Watch `L_seg` decrease from ~8.0 to ~6.0
- `grad_norm` should be in range 0.7-2.0
- `lr` increasing linearly (warmup schedule)
- Validation shows very low organ Dice (expected)

### Steps 500-10000 (Main learning):
- `L_seg` decreasing towards 0.5-1.0
- `grad_norm` mostly stable <2.0
- Validation organ Dice steadily improving
- BYOL activates at step ~8000 (epoch 3+)

### Steps 10000+ (Fine-tuning):
- `L_seg` should be <0.5
- `grad_norm` very stable <1.0
- Validation Dice plateauing at optimal level
- `L_byol` contributing to loss (visible in log)

### Debugging NaNs:
1. Check if `grad_norm` exceeded 10.0 before NaN
2. Look for "Weight Corruption" warnings
3. Check data quality if NaNs appear suddenly mid-training
4. Consider lower learning rate if NaNs recurrent

---

## Configuration

The following configuration values control logging:

**In `configs/stage1_config.yaml`:**
```yaml
log_every_n_steps: 100        # Periodic logging frequency
val_every_n_steps: 500        # Validation logging frequency
total_steps: 30000            # Total training steps
```

**In `training/train_stage1.py`:**
```python
logger = setup_logger(log_path)  # Dual console + file logging
```

The logger writes to both:
- **Console (stdout)**: Real-time progress
- **File (`training.log`)**: Permanent record for analysis

---

## Summary

Enhanced logging provides **real-time visibility** into:
- ✅ Training convergence (L_seg trends)
- ✅ Gradient stability (grad_norm spikes)
- ✅ Validation performance (all 7 class Dice)
- ✅ Training progress (step % completion)
- ✅ Learning rate schedule (lr trends)
- ✅ Computational efficiency (s/step)
- ✅ NaN detection (warnings + errors)

This enables quick diagnosis of training issues and confidence in training stability.
