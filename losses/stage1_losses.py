
"""
losses/stage1_losses.py

Stage 1 Loss Functions — Segmentation losses for VM-UNet training.
===================================================================
Implements the segmentation loss components used during Stage 1 training.

Loss function hierarchy
-----------------------
L_stage1 = 1.0 * L_seg + 0.1 * L_byol   (L_byol added at epoch 5)

L_seg  : SegmentationLoss  — cross-entropy with label smoothing (primary)
L_dice : DiceLoss          — optional auxiliary to cross-entropy

Why label smoothing (ε = 0.1)?
-------------------------------
TotalSegmentator pseudo-labels have 5-15% error rate at organ boundaries.
Training with hard labels (0 or 1) teaches the network to be overconfident
about sometimes-wrong labels. Label smoothing prevents this:

  Hard label for "liver" pixel  : [0, 1, 0, 0, 0, 0, 0]
  Smooth label for "liver" pixel: [0.014, 0.914, 0.014, 0.014, 0.014, 0.014, 0.014]

  Formula: smooth[k] = (1 - ε) if k == true_class else ε / (num_classes - 1)
           = 0.9 for the true class
           = 0.1 / 6 ≈ 0.014 for other classes

This tells the network: "be mostly confident about liver, but acknowledge
there's slight uncertainty" — appropriate for pseudo-labels.

NOTE on logits vs S
--------------------
SegmentationLoss takes LOGITS (raw scores before softmax), NOT S.
F.cross_entropy applies log-softmax internally, which is numerically
more stable than computing softmax then log separately.

DiceLoss takes PROBS (S, after softmax) because Dice requires
probabilities in [0, 1] for the soft intersection computation.

Reference: Architecture.pdf — Chapter 8 (Stage 1 Loss Functions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CLASS NAMES — consistent with utils/masking.py
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'background',    # 0
    'liver_spleen',  # 1
    'kidney',        # 2
    'vessel',        # 3
    'lung',          # 4
    'bone',          # 5
    'soft_tissue',   # 6
]


# ─────────────────────────────────────────────────────────────────────────────
# SegmentationLoss
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for organ segmentation.

    Wraps F.cross_entropy with the label_smoothing parameter.
    Label smoothing is crucial for training on TotalSegmentator pseudo-labels
    which have ~5-15% error at organ boundaries.

    Formula (for a pixel with true class k*):
        smooth[k] = (1 - ε)          if k == k*   (true class)
                  = ε / (C - 1)      otherwise    (other classes)

    With ε=0.1, C=7:
        True class  receives target probability: 0.9
        Other classes receive target probability: 0.1 / 6 ≈ 0.0167

    Loss computation (via F.cross_entropy with label_smoothing):
        L = -Σ_k smooth[k] * log(softmax(logits)[k])

    Parameters
    ----------
    num_classes : int
        Number of organ segmentation classes. Default: 7.
    label_smoothing : float
        Smoothing factor ε ∈ [0, 1). Default: 0.1.
        0.0 = standard hard-label cross-entropy.
        0.1 = recommended for pseudo-label training.
    weight : torch.Tensor or None
        Optional [num_classes] tensor for class-frequency weighting.
        Useful when some organ classes are much rarer than others.

    Forward
    -------
    logits  : [B, num_classes, H, W]  ← raw scores from segmentation head
    targets : [B, H, W]               ← integer class labels 0 to (num_classes-1)

    Returns
    -------
    scalar loss (0-dim tensor)

    Notes
    -----
    * F.cross_entropy applies log-softmax internally — numerically stable.
    * Use logits (not S) for the loss. Cross-entropy applies softmax internally.
    * targets must be long (int64) dtype.

    Examples
    --------
    >>> loss_fn = SegmentationLoss()
    >>> logits  = torch.randn(2, 7, 512, 512)
    >>> targets = torch.randint(0, 7, (2, 512, 512))
    >>> loss    = loss_fn(logits, targets)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        num_classes:     int   = 7,
        label_smoothing: float = 0.1,
        weight:          torch.Tensor = None,
    ):
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}."
            )
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}."
            )

        self.num_classes     = num_classes
        self.label_smoothing = label_smoothing

        # Register weight as a buffer so it moves with .to(device) calls.
        # If no weight is provided, register None (F.cross_entropy handles None).
        if weight is not None:
            if weight.shape != (num_classes,):
                raise ValueError(
                    f"weight must have shape ({num_classes},), "
                    f"got {tuple(weight.shape)}."
                )
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Parameters
        ----------
        logits  : [B, num_classes, H, W]  raw class scores (NOT softmax output)
        targets : [B, H, W]               integer labels in [0, num_classes-1]

        Returns
        -------
        scalar loss
        """
        # ── Input validation ──────────────────────────────────────────────
        if logits.dim() != 4:
            raise ValueError(
                f"logits must be 4-D [B, C, H, W], got {tuple(logits.shape)}."
            )
        if targets.dim() != 3:
            raise ValueError(
                f"targets must be 3-D [B, H, W], got {tuple(targets.shape)}."
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits has {logits.shape[1]} classes, "
                f"expected {self.num_classes}."
            )
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits B={logits.shape[0]}, "
                f"targets B={targets.shape[0]}."
            )

        # Ensure targets are the correct dtype (long/int64)
        targets = targets.long()

        # ── F.cross_entropy with label_smoothing ──────────────────────────
        # F.cross_entropy(input, target, ...) expects:
        #   input  : [B, C, *]   (C = num_classes)
        #   target : [B, *]      (integer class labels)
        # label_smoothing is supported from PyTorch ≥ 1.10.
        return F.cross_entropy(
            logits,
            targets,
            weight          = self.weight,
            label_smoothing = self.label_smoothing,
            reduction       = 'mean',
        )

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"label_smoothing={self.label_smoothing}, "
            f"weighted={self.weight is not None}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DiceLoss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation — optional auxiliary to cross-entropy.

    Dice coefficient measures the overlap between prediction and ground truth:
        Dice(pred, target) = (2 * |pred ∩ target| + smooth) /
                             (|pred| + |target| + smooth)

    Dice loss = 1 - Dice coefficient.

    The "soft" variant uses softmax probabilities directly (not hard thresholds),
    making it fully differentiable and therefore usable during training.

    Formula (per class k, for batch image b):
        intersection_k = Σ_{h,w} probs[b,k,h,w] * target_one_hot[b,k,h,w]
        pred_sum_k     = Σ_{h,w} probs[b,k,h,w]
        target_sum_k   = Σ_{h,w} target_one_hot[b,k,h,w]
        dice_k         = (2 * intersection_k + smooth) / (pred_sum_k + target_sum_k + smooth)
        loss_k         = 1 - dice_k

    Final loss = mean over all classes (including background).

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant to prevent division by zero.
        Default: 1.0. Prevents the loss from being undefined when both
        pred and target are zero for a class (absent organ).

    Forward
    -------
    probs   : [B, num_classes, H, W]  softmax probabilities S (NOT logits)
    targets : [B, H, W]               integer class labels 0 to (num_classes-1)

    Returns
    -------
    scalar loss in [0, 1]
        0 = perfect overlap for all classes
        1 = zero overlap for all classes

    Notes
    -----
    * Takes PROBS (after softmax), not logits.
    * Background class (0) is included in the mean.
    * Smooth=1.0 means Dice is bounded away from 0 for absent classes,
      preventing instability when a class doesn't appear in the batch.

    Examples
    --------
    >>> probs   = torch.softmax(torch.randn(2, 7, 64, 64), dim=1)
    >>> targets = torch.randint(0, 7, (2, 64, 64))
    >>> dice_fn = DiceLoss()
    >>> loss    = dice_fn(probs, targets)
    >>> 0 <= loss.item() <= 1
    True
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        if smooth < 0:
            raise ValueError(f"smooth must be >= 0, got {smooth}.")
        self.smooth = smooth

    def forward(
        self,
        probs:   torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft Dice loss.

        Parameters
        ----------
        probs   : [B, num_classes, H, W]  softmax probabilities in [0, 1]
        targets : [B, H, W]               integer labels

        Returns
        -------
        scalar Dice loss (mean over classes)
        """
        # ── Input validation ──────────────────────────────────────────────
        if probs.dim() != 4:
            raise ValueError(
                f"probs must be 4-D [B, C, H, W], got {tuple(probs.shape)}."
            )
        if targets.dim() != 3:
            raise ValueError(
                f"targets must be 3-D [B, H, W], got {tuple(targets.shape)}."
            )

        B, C, H, W = probs.shape
        targets = targets.long()

        # ── One-hot encode targets: [B, H, W] → [B, C, H, W] ─────────────
        # F.one_hot returns [B, H, W, C], so we permute to [B, C, H, W].
        target_one_hot = F.one_hot(targets, num_classes=C)   # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)  # [B, C, H, W]
        target_one_hot = target_one_hot.float()

        # ── Soft Dice per class ───────────────────────────────────────────
        # Flatten spatial dimensions: [B, C, H, W] → [B, C, H*W]
        probs_flat  = probs.reshape(B, C, -1)           # [B, C, H*W]
        target_flat = target_one_hot.reshape(B, C, -1)  # [B, C, H*W]

        # Intersection: Σ_{h,w} pred[b,k,h,w] * target[b,k,h,w]
        # Shape: [B, C]
        intersection = (probs_flat * target_flat).sum(dim=2)

        # Sum of predictions and targets per class
        pred_sum   = probs_flat.sum(dim=2)    # [B, C]
        target_sum = target_flat.sum(dim=2)   # [B, C]

        # Dice coefficient per (batch, class): [B, C]
        dice = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )

        # Dice loss = 1 - Dice, averaged over batch and classes
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def extra_repr(self) -> str:
        return f"smooth={self.smooth}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_dice_per_class
# ─────────────────────────────────────────────────────────────────────────────

def compute_dice_per_class(
    probs:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 7,
) -> dict:
    """
    Compute Dice score per organ class. Used for evaluation and logging ONLY.

    Dice(pred_k, target_k) = (2 * |pred_k ∩ target_k| + smooth) /
                              (|pred_k| + |target_k| + smooth)

    Uses hard thresholded predictions (argmax over class dimension) to compute
    binary per-class masks, then evaluates Dice against ground truth binary masks.

    This differs from DiceLoss (which uses soft probabilities) — here we want
    the actual hard-segmentation performance for logging/monitoring.

    Parameters
    ----------
    probs : torch.Tensor  [B, num_classes, H, W]
        Softmax probabilities S from VM-UNet.
    targets : torch.Tensor  [B, H, W]
        Integer ground truth labels (0 to num_classes-1).
    num_classes : int
        Number of classes. Default: 7.

    Returns
    -------
    dict : {class_name: float, ..., 'mean': float}
        Per-class Dice scores and their mean.
        Class names: 'background', 'liver_spleen', 'kidney', 'vessel',
                     'lung', 'bone', 'soft_tissue', 'mean'.

    Notes
    -----
    * Uses smooth=1.0 for numerical stability (same as DiceLoss).
    * Hard predictions (argmax) are used, not soft probabilities.
    * When a class is absent from both prediction AND ground truth,
      Dice is computed as (0 + smooth) / (0 + 0 + smooth) = 1.0
      (both are empty, so perfect agreement).
    * Returns Python floats (not tensors) for easy logging.

    Examples
    --------
    >>> probs   = torch.softmax(torch.randn(2, 7, 64, 64), dim=1)
    >>> targets = torch.randint(0, 7, (2, 64, 64))
    >>> scores  = compute_dice_per_class(probs, targets)
    >>> 'mean' in scores and 'liver_spleen' in scores
    True
    """
    smooth = 1.0

    # Hard predicted class at each pixel: [B, H, W]
    pred_classes = probs.argmax(dim=1)

    dice_scores = {}
    class_dice_list = []

    for k in range(num_classes):
        # Binary mask for class k: True where argmax == k
        pred_k   = (pred_classes == k).float()  # [B, H, W]
        target_k = (targets == k).float()        # [B, H, W]

        # Flatten to compute global Dice across entire batch
        pred_flat   = pred_k.reshape(-1)    # [B*H*W]
        target_flat = target_k.reshape(-1)  # [B*H*W]

        intersection = (pred_flat * target_flat).sum()
        pred_sum     = pred_flat.sum()
        target_sum   = target_flat.sum()

        dice_k = (2.0 * intersection + smooth) / (
            pred_sum + target_sum + smooth
        )

        dice_val = dice_k.item()
        class_name = CLASS_NAMES[k]
        dice_scores[class_name] = dice_val
        class_dice_list.append(dice_val)

    # Mean Dice across all classes
    dice_scores['mean'] = sum(class_dice_list) / len(class_dice_list)

    return dice_scores


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch.nn.functional as F  # noqa: F811 (already imported above)

    print("=" * 60)
    print("losses/stage1_losses.py — self-test")
    print("=" * 60)

    B, C, H, W = 2, 7, 64, 64   # use 64×64 to keep tests fast

    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    # ── SegmentationLoss ──────────────────────────────────────────────────
    print("\n── SegmentationLoss ──────────────────────────────────────────")

    seg_loss = SegmentationLoss()
    loss_val = seg_loss(logits, targets)

    assert loss_val.shape == (), \
        f"Expected scalar, got shape {loss_val.shape}"
    assert loss_val > 0, \
        f"Expected positive loss, got {loss_val.item()}"

    print(f"  Output shape : {tuple(loss_val.shape)} (scalar) ✓")
    print(f"  Loss value   : {loss_val.item():.4f} ✓")
    print(f"SegmentationLoss: PASSED (loss={loss_val:.4f})")

    # Gradient flows through loss → logits
    logits_grd = torch.randn(B, C, H, W, requires_grad=True)
    loss_grd = SegmentationLoss()(logits_grd, targets)
    loss_grd.backward()
    assert logits_grd.grad is not None, "Gradient did not reach logits."
    gnorm = logits_grd.grad.norm().item()
    assert gnorm > 0, "Gradient norm is zero."
    print(f"  Gradient norm : {gnorm:.6f} ✓")

    # label_smoothing=0.0 should give standard cross-entropy
    seg_no_smooth = SegmentationLoss(label_smoothing=0.0)
    loss_no_smooth = seg_no_smooth(logits, targets)
    loss_manual    = F.cross_entropy(logits, targets.long(), reduction='mean')
    assert torch.allclose(loss_no_smooth, loss_manual, atol=1e-5), \
        "label_smoothing=0.0 should match standard F.cross_entropy."
    print(f"  label_smoothing=0.0 matches F.cross_entropy ✓")

    # Class weight support
    weight = torch.tensor([0.5, 2.0, 2.0, 3.0, 2.0, 1.5, 1.0])
    seg_weighted = SegmentationLoss(weight=weight)
    loss_w = seg_weighted(logits, targets)
    assert loss_w.item() > 0
    print(f"  Weighted loss : {loss_w.item():.4f} ✓")

    # ── DiceLoss ──────────────────────────────────────────────────────────
    print("\n── DiceLoss ──────────────────────────────────────────────────")

    probs    = F.softmax(logits, dim=1)
    dice_fn  = DiceLoss()
    dice_val = dice_fn(probs, targets)

    assert dice_val.shape == (), \
        f"Expected scalar, got shape {dice_val.shape}"
    assert 0 <= dice_val.item() <= 1, \
        f"Dice loss must be in [0, 1], got {dice_val.item()}"

    print(f"  Output shape : {tuple(dice_val.shape)} (scalar) ✓")
    print(f"  Loss value   : {dice_val.item():.4f} ✓")
    print(f"DiceLoss: PASSED (loss={dice_val:.4f})")

    # Perfect prediction should give Dice loss ≈ 0
    # (all probability mass on the correct class)
    probs_perfect = torch.zeros(B, C, H, W)
    for b in range(B):
        for k in range(C):
            probs_perfect[b, k][targets[b] == k] = 1.0
    # Normalise (in case some pixels have all zeros due to no target)
    probs_perfect = probs_perfect + 1e-8
    probs_perfect = probs_perfect / probs_perfect.sum(dim=1, keepdim=True)
    dice_perfect = dice_fn(probs_perfect, targets)
    assert dice_perfect.item() < 0.05, \
        f"Near-perfect predictions should give Dice loss < 0.05, got {dice_perfect.item():.4f}"
    print(f"  Near-perfect prediction Dice loss: {dice_perfect.item():.6f} ≈ 0 ✓")

    # Gradient flows through Dice loss (probs are non-leaf: grad lives on logits)
    logits_dice = torch.randn(B, C, H, W, requires_grad=True)
    probs_grd   = F.softmax(logits_dice, dim=1)
    dice_grd    = DiceLoss()(probs_grd, targets)
    dice_grd.backward()
    assert logits_dice.grad is not None, "Gradient did not reach logits."
    gnorm_dice = logits_dice.grad.norm().item()
    assert gnorm_dice > 0, "Dice gradient norm is zero."
    print(f"  Gradient norm (via logits→softmax→Dice) : {gnorm_dice:.6f} ✓")

    # ── compute_dice_per_class ────────────────────────────────────────────
    print("\n── compute_dice_per_class ────────────────────────────────────")

    per_class = compute_dice_per_class(probs, targets)

    assert 'mean' in per_class, "Missing 'mean' key."
    assert 'liver_spleen' in per_class, "Missing 'liver_spleen' key."
    assert len(per_class) == C + 1, \
        f"Expected {C + 1} entries (7 classes + mean), got {len(per_class)}."

    # All Dice scores should be floats in [0, 1]
    for name, score in per_class.items():
        assert isinstance(score, float), \
            f"{name}: expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, \
            f"{name}: Dice score {score:.4f} out of range [0, 1]."

    # Mean should equal average of per-class scores
    expected_mean = sum(
        v for k, v in per_class.items() if k != 'mean'
    ) / C
    assert abs(per_class['mean'] - expected_mean) < 1e-6, \
        f"Mean mismatch: {per_class['mean']:.6f} vs {expected_mean:.6f}"

    print(f"  Keys         : {sorted(per_class.keys())} ✓")
    print(f"  Mean Dice    : {per_class['mean']:.4f} ✓")
    print(f"  liver_spleen : {per_class['liver_spleen']:.4f} ✓")
    print(f"  All scores in [0,1] ✓")
    print(f"compute_dice_per_class: PASSED (mean dice={per_class['mean']:.4f})")

    # ── Edge cases ───────────────────────────────────────────────────────
    print("\n── Edge cases ────────────────────────────────────────────────")

    # All pixels belong to class 0 — other classes should be absent
    targets_all0 = torch.zeros(B, H, W, dtype=torch.long)
    per_class_0 = compute_dice_per_class(probs, targets_all0)
    # Background Dice should be > 0 (probs[:,0] has some mass there)
    assert 'background' in per_class_0
    print(f"  All-class-0 targets: background Dice = {per_class_0['background']:.4f} ✓")

    # Batch size = 1
    logits_b1  = torch.randn(1, C, H, W)
    targets_b1 = torch.randint(0, C, (1, H, W))
    loss_b1 = SegmentationLoss()(logits_b1, targets_b1)
    assert loss_b1.item() > 0
    print(f"  B=1 works: loss={loss_b1.item():.4f} ✓")

    # Wrong input dimension raises ValueError
    try:
        SegmentationLoss()(torch.randn(B, C, H), targets)
        assert False, "Should have raised ValueError."
    except ValueError as e:
        print(f"  3-D logits raises ValueError ✓")

    try:
        DiceLoss()(torch.randn(B, C, H), targets)
        assert False, "Should have raised ValueError."
    except ValueError as e:
        print(f"  3-D probs raises ValueError ✓")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)