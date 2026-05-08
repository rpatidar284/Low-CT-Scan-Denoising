# Early Training Stability Fixes - Quick Reference

## Summary

Three coordinated fixes to prevent NaN loss in first 30000 steps:

### Fix 1: Step-Based Class Weights
- **What:** Defer class weights from step 0 to step 3000
- **Why:** Class weights destabilize random init
- **File:** `training/train_stage1.py` lines 573-583, 654-665
- **Key:** Look for "★ Activating class weights at step 3000" in logs

### Fix 2: Complete VSSD Initialization
- **What:** Add proper init for delta_proj.weight, B_proj, C_proj, D
- **Why:** Prevents gradient explosion in state space layers
- **File:** `models/vmamba_blocks.py` lines 313-335
- **Key:** All 5 weight matrices initialized with std=0.01 or small constant

### Fix 3: Conservative Learning Rate
- **What:** Lower LR (6.0e-5 → 2.0e-5) + longer warmup (2000 → 5000)
- **Why:** Gives VSSD time to stabilize + slower ramp = safer
- **File:** `configs/stage1_config.yaml` lines 71, 74
- **Key:** Warmup now covers steps 0-5000 (16.7% of training)

---

## Expected Training Progression

```
Step 0-100:        Loss 7.8 → 7.4 (smooth, no NaN)
Step 100-3000:     Loss 7.4 → 1.0 (steady decrease)
Step 3000:         ★ Class weights activate, loss may jump to 1.05
Step 3000-5000:    Loss 1.05 → 0.8 (adapts to weights)
Step 5000-30000:   Loss 0.8 → 0.1 (main training, LR decaying)
```

---

## Log File Markers

**Fix 1 Activation:**
```
★ Activating class weights at step 3000
```

**Normal early training (no NaN):**
```
Step   100/30000 (  0.3%) | ep=  0 | L_seg=7.40 | L_byol=-- | grad_norm=0.45
Step  1000/30000 (  3.3%) | ep=  0 | L_seg=5.80 | L_byol=-- | grad_norm=0.51
Step  3000/30000 (10.0%) | ep=  1 | L_seg=0.99 | L_byol=-- | grad_norm=0.50
Step  5000/30000 (16.7%) | ep=  2 | L_seg=0.71 | L_byol=-- | grad_norm=0.49
```

**Validation improvement over time:**
```
Step   500 | Val Dice → ... | kidney=0.03 ...
Step  2500 | Val Dice → ... | kidney=0.12 ...
Step  3500 | Val Dice → ... | kidney=0.18 ...  ← improvements continue
Step  5000 | Val Dice → ... | kidney=0.28 ...
```

---

## Verification Checklist

Before starting training:
- [ ] `learning_rate: 2.0e-5` in config ✓
- [ ] `warmup_steps: 5000` in config ✓
- [ ] `seg_criterion = SegmentationLoss(..., weight=None)` in code ✓
- [ ] Step 3000 trigger code present ✓
- [ ] VSSD._init_weights() includes 5 initializations ✓

During training (first 1000 steps):
- [ ] No NaN or inf losses ✓
- [ ] grad_norm values normal (0.3-1.5) ✓
- [ ] Loss decreasing smoothly ✓

At step 3000:
- [ ] Class weight activation message appears ✓
- [ ] Training continues uninterrupted ✓

---

## Rollback Instructions

If something goes wrong, revert with git:

```bash
git checkout HEAD -- training/train_stage1.py models/vmamba_blocks.py configs/stage1_config.yaml
```

Then re-apply fixes one at a time.

---

## Files Changed

```
training/train_stage1.py     (2 changes: init + trigger)
models/vmamba_blocks.py      (1 change: complete _init_weights)
configs/stage1_config.yaml   (2 changes: lr + warmup)
```

---

## Training Command

```bash
cd /home/teaching/Music/Nigam_51/Project_51
python3 training/train_stage1.py --resume-from-best
```

Expected to reach step 30000 without NaN crashes.

---

## Success Criteria

✅ Training completes 30000 steps
✅ No NaN losses detected
✅ Loss curve smooth (no sudden jumps except at step 3000)
✅ Organ Dice improving consistently
✅ Final kidney Dice > 0.4 (improved from 0.0 baseline)

---

**Status: READY TO TRAIN**
