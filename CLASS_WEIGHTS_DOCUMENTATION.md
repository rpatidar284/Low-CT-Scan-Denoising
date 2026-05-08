# Class Weights for Stage 1 Segmentation Loss

## Overview

Class weights have been added to the `SegmentationLoss` in Stage 1 training to handle **class imbalance**. The background class dominates the dataset, so it is down-weighted while organ classes are up-weighted to ensure balanced learning.

## Class Weight Configuration

### Weight Vector
```python
class_weights = torch.tensor([0.1, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0])
```

### Class Mapping
| Index | Class Name | Weight | Reasoning |
|-------|-----------|--------|-----------|
| 0 | background | 0.1 | Down-weighted (dominates dataset) |
| 1 | liver_spleen | 1.5 | Up-weighted (often underfitted) |
| 2 | kidney_L | 2.0 | Up-weighted (small, hard to segment) |
| 3 | kidney_R | 1.5 | Up-weighted (small organs) |
| 4 | spleen | 1.5 | Up-weighted (often missed) |
| 5 | bone | 1.5 | Up-weighted (sparse and important) |
| 6 | lung | 1.0 | Baseline weight (common, easy to segment) |

## Why Class Weighting?

### The Problem: Class Imbalance
In CT scans with a 512×512 pixel resolution:
- **Background:** ~80-90% of pixels
- **Organs combined:** ~10-20% of pixels

Without class weighting, the loss function heavily penalizes background errors (which are rare) and less penalizes organ errors (which contribute less to overall loss).

### The Solution: Weighted Cross-Entropy
The weighted cross-entropy loss is:
```
L = -Σ_c weight[c] * p_true[c] * log(p_pred[c])
```

Where:
- `weight[c]` = class weight for class c
- `p_true[c]` = true probability of class c
- `p_pred[c]` = predicted probability of class c

With class weights:
- Background errors are penalized less (weight=0.1)
- Organ errors are penalized more (weights=1.5-2.0)
- This forces the network to learn organ boundaries precisely

## Implementation

### Location
**File:** `training/train_stage1.py` (lines 600-612)

```python
# Class weights to handle class imbalance (background dominates)
class_weights = torch.tensor([0.1, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0])
class_weights = class_weights.to(device)

seg_criterion = SegmentationLoss(
    num_classes     = num_classes,
    label_smoothing = label_smoothing,
    weight          = class_weights,
)

logger.info(f"SegmentationLoss configured with class weights: {class_weights.tolist()}")
```

### Loss Function
**File:** `losses/stage1_losses.py`

The `SegmentationLoss` class accepts an optional `weight` parameter:

```python
class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes:     int   = 7,
        label_smoothing: float = 0.1,
        weight:          torch.Tensor = None,  # ← Class weights
    ):
        # ...
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None
    
    def forward(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            weight          = self.weight,      # ← Passed to F.cross_entropy
            label_smoothing = self.label_smoothing,
            reduction       = 'mean',
        )
```

## Expected Training Behavior

### With Class Weights
```
[Step 100] L_seg=4.23 | organ pixels penalized more heavily
[Step 200] L_seg=3.87 | background errors cost less
[Step 300] Val Dice → liver=0.58 kidney=0.42 lung=0.65
           ↑ Organ metrics improve faster
```

### Key Observations
1. **Loss may not decrease as smoothly** — background pixels contribute less to loss
2. **Organ Dice scores improve faster** — small organs (kidney) get more gradient signal
3. **Background accuracy may slightly decrease** — acceptable trade-off for better organ segmentation
4. **Training is more stable** — balanced gradients from all 7 classes

## Weight Tuning Guide

If training shows imbalance issues:

### Too Much Background (Val Dice for organs too low)
- Reduce background weight to 0.05
- Example: `[0.05, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0]`

### Too Much Organ (Background accuracy poor, organs noisy)
- Increase background weight to 0.3
- Example: `[0.3, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0]`

### Kidney Still Underfitting
- Increase kidney weights to 2.5
- Example: `[0.1, 1.5, 2.5, 2.5, 1.5, 1.5, 1.0]`

### Current Weights (Default)
```python
[0.1, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0]
```
Recommended starting point — balances organ vs background learning.

## Monitoring in Logs

### Look For
```
[2026-05-03 17:39:27] SegmentationLoss configured with class weights: [0.1, 1.5, 2.0, 1.5, 1.5, 1.5, 1.0]
```

### In training.log
```
[Step 100] L_seg=4.23 | lr=1.00e-05
[Step 500] Val Dice → liver=0.52 kidney=0.38 lung=0.61 mean=0.50
          Organ-specific metrics show impact of class weights
```

## Validation Metric Interpretation

With class weights enabled:

| Metric | Interpretation |
|--------|---|
| L_seg ↓ slowly but organs improve | ✓ Class weighting is working |
| Organ Dice ↑↑ rapidly | ✓ Organs are learning faster |
| Background accuracy ↓ slightly | ✓ Expected (acceptable trade-off) |
| Mean Dice ↑ | ✓ Overall improvement |

## References

- **Weighted Cross-Entropy:** Standard technique in imbalanced classification
- **PyTorch Docs:** `torch.nn.CrossEntropyLoss(weight=...)` 
- **Architecture.pdf:** Chapter 8 (Stage 1 Loss Functions)

## Status: ✓ Implemented

Class weights are now active during Stage 1 training. They will improve organ segmentation quality, especially for small organs like kidneys.

**Next:** Monitor first 500 training steps and verify organ Dice scores improve more rapidly than background accuracy decreases.
