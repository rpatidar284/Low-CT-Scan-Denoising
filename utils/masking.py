"""
utils/masking.py

Masked Average Pooling — anatomy embedding extraction.
======================================================
Computes per-organ feature embeddings e_a from the VM-UNet decoder
features and the soft segmentation map S.

This is the bridge between Stage 1 (segmentation) and Stage 2 (denoising):
  e_a = masked_average_pooling(decoder_features, S)   → [B, 7, 96]

e_a is then passed to Stage 2 as anatomy conditioning through the
CrossAttentionWithAnatomy modules inside each AnatomyMamba_block.

Physical meaning of e_a
-----------------------
e_a[b, k, :] = the average feature vector of organ k in image b,
               weighted by how confident the model is that each pixel
               belongs to organ k.

Example:
  e_a[0, 1, :]  ← average liver/spleen feature vector for image 0
  e_a[0, 4, :]  ← average lung feature vector for image 0

Because the weighting uses S (soft probabilities) rather than hard masks:
  * Boundary pixels contribute partially to both neighbouring organs.
  * Uncertain pixels (flat S distribution) contribute little to any organ.
  * This produces smooth, noise-tolerant embeddings.

Reference: Architecture.pdf – Chapter 8 (Stage 1 outputs / e_a computation)
"""

import torch


# ─────────────────────────────────────────────────────────────────────────────
# CLASS NAMES — for documentation and visualisation
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
# masked_average_pooling
# ─────────────────────────────────────────────────────────────────────────────

def masked_average_pooling(
    decoder_features: torch.Tensor,
    S:                torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-organ feature embeddings using S as soft spatial attention weights.

    For each organ class k ∈ {0, …, num_classes-1}:

        weight[b, h, w]        = S[b, k, h, w]           ∈ [0, 1]
        weighted_feat[b, c]    = Σ_{h,w} weight[b,h,w] · features[b,c,h,w]
        total_weight[b]        = Σ_{h,w} weight[b,h,w]   (+ ε for stability)
        e_a_k[b, c]            = weighted_feat[b,c] / total_weight[b]

    All K classes are computed simultaneously using batched tensor operations
    (no Python loop) for efficiency.

    Why soft weighting instead of hard masks?
    -----------------------------------------
    Hard threshold (S[:,k] > 0.5):
      - Not differentiable → cannot backpropagate through e_a.
      - Amplifies TotalSegmentator errors at boundaries.
      - Loses information: S[liver]=0.51 treated identically to S[liver]=0.99.

    Soft weighting:
      - Fully differentiable → gradients flow back into the decoder and
        the segmentation head during Stage 1 training.
      - Boundary pixels contribute proportionally to both neighbouring organs.
      - Naturally handles TotalSegmentator uncertainty.

    Parameters
    ----------
    decoder_features : torch.Tensor  [B, C, H, W]
        Full-resolution decoder output just before the segmentation head.
        C = embed_dim = 96 in the default configuration.
        Must be in BCHW (channel-first) format.

    S : torch.Tensor  [B, num_classes, H, W]
        Soft segmentation probability map output by VMUNet.
        S[:, k, h, w] ∈ [0, 1] is the probability that pixel (h, w)
        belongs to organ class k.
        S.sum(dim=1) == 1 at every spatial position (softmax output).

    Returns
    -------
    e_a : torch.Tensor  [B, num_classes, C]
        Per-organ anatomy embeddings.
        e_a[b, k, :] is the soft-weighted average feature vector of
        organ k in image b.

    Shapes (default config)
    -----------------------
    decoder_features : [B,  96, 512, 512]
    S                : [B,   7, 512, 512]
    e_a              : [B,   7,  96]

    Numerical notes
    ---------------
    * eps = 1e-8 prevents division-by-zero when a class is completely absent
      from the image (e.g. lung class in an abdominal-only scan).
    * When a class is absent, total_weight ≈ 0 and e_a_k ≈ 0 (zero vector).
      This is safe: Stage 2 cross-attention with a zero query simply attends
      uniformly, contributing little to the output.

    Examples
    --------
    >>> features = torch.randn(2, 96, 512, 512)
    >>> S        = torch.softmax(torch.randn(2, 7, 512, 512), dim=1)
    >>> e_a      = masked_average_pooling(features, S)
    >>> e_a.shape
    torch.Size([2, 7, 96])
    """
    # ── Input validation ──────────────────────────────────────────────────
    if decoder_features.dim() != 4:
        raise ValueError(
            f"decoder_features must be 4-D [B, C, H, W], "
            f"got shape {tuple(decoder_features.shape)}."
        )
    if S.dim() != 4:
        raise ValueError(
            f"S must be 4-D [B, num_classes, H, W], "
            f"got shape {tuple(S.shape)}."
        )

    B_f, C, H_f, W_f = decoder_features.shape
    B_s, K, H_s, W_s = S.shape

    if B_f != B_s:
        raise ValueError(
            f"Batch size mismatch: decoder_features has B={B_f}, S has B={B_s}."
        )
    if H_f != H_s or W_f != W_s:
        raise ValueError(
            f"Spatial size mismatch: decoder_features is {H_f}×{W_f}, "
            f"S is {H_s}×{W_s}. "
            f"Resize S to match decoder_features before calling this function."
        )

    # ── Batched soft pooling — all K classes in parallel ─────────────────
    #
    # Goal:  e_a[b, k, c] = Σ_{h,w} S[b,k,h,w] · features[b,c,h,w]
    #                        ──────────────────────────────────────────
    #                              Σ_{h,w} S[b,k,h,w]  + ε
    #
    # We achieve this with a single einsum followed by a division,
    # avoiding any Python-level loop over K.
    #
    # Shapes involved:
    #   S                : [B, K, H, W]
    #   decoder_features : [B, C, H, W]
    #
    # Numerator:
    #   weighted_sum[b, k, c] = Σ_{h,w} S[b,k,h,w] · features[b,c,h,w]
    #
    #   Expand S to [B, K, 1, H, W] and features to [B, 1, C, H, W],
    #   multiply element-wise → [B, K, C, H, W], then sum over (H, W).
    #
    #   Equivalently (and more memory-efficiently) via einsum:
    #   'bkhw, bchw -> bkc'

    # Numerator: weighted sum of features for each class
    # einsum 'bkhw, bchw -> bkc' :
    #   for each batch b, class k, channel c:
    #     sum over h, w of S[b,k,h,w] * features[b,c,h,w]
    weighted_sum = torch.einsum('bkhw, bchw -> bkc', S, decoder_features)
    # weighted_sum : [B, K, C]

    # Denominator: total probability mass per class per image
    # S.sum(dim=[2,3]) : [B, K]
    # unsqueeze(-1)    : [B, K, 1]   (broadcasts over C)
    total_weight = S.sum(dim=[2, 3]).unsqueeze(-1) + 1e-8  # [B, K, 1]

    # Soft weighted average
    e_a = weighted_sum / total_weight   # [B, K, C]

    return e_a


# ─────────────────────────────────────────────────────────────────────────────
# compute_anatomy_embeddings  — convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def compute_anatomy_embeddings(vm_unet_output: dict) -> torch.Tensor:
    """
    Convenience wrapper: extract e_a from a VMUNet output dict.

    Pulls ``decoder_features`` and ``S`` from the dict and calls
    :func:`masked_average_pooling`.

    Parameters
    ----------
    vm_unet_output : dict
        The dict returned by ``VMUNet.forward()``.  Must contain:
          'decoder_features' : [B, C, H, W]
          'S'                : [B, num_classes, H, W]

    Returns
    -------
    e_a : torch.Tensor  [B, num_classes, C]

    Usage in Stage 2 training
    -------------------------
    ::

        with torch.no_grad():
            stage1_out = frozen_vm_unet(x_ldct)
            e_a = compute_anatomy_embeddings(stage1_out)   # [B, 7, 96]
            S   = stage1_out['S']                          # [B, 7, 512, 512]

    Usage in Stage 1 training (for L_anatomy)
    ------------------------------------------
    ::

        out = vm_unet(x_ndct)
        e_a = compute_anatomy_embeddings(out)              # [B, 7, 96]

    Examples
    --------
    >>> out = vm_unet(torch.randn(2, 1, 512, 512))
    >>> e_a = compute_anatomy_embeddings(out)
    >>> e_a.shape
    torch.Size([2, 7, 96])
    """
    required_keys = {'decoder_features', 'S'}
    missing = required_keys - set(vm_unet_output.keys())
    if missing:
        raise KeyError(
            f"vm_unet_output is missing required keys: {missing}. "
            f"Available keys: {set(vm_unet_output.keys())}"
        )

    return masked_average_pooling(
        vm_unet_output['decoder_features'],
        vm_unet_output['S'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("utils/masking.py — self-test")
    print("=" * 60)

    # ── Test 1: basic shape and correctness ───────────────────────────────
    print("\n── Test 1: shape + correctness (all pixels → class 1) ────────")

    B, C, H, W  = 2, 96, 512, 512
    num_classes = 7

    features = torch.randn(B, C, H, W)

    # All pixels assigned to class 1 with probability 1
    S        = torch.zeros(B, num_classes, H, W)
    S[:, 1, :, :] = 1.0

    e_a = masked_average_pooling(features, S)

    # Shape check
    assert e_a.shape == (B, num_classes, C), (
        f"Expected ({B}, {num_classes}, {C}), got {tuple(e_a.shape)}"
    )
    print(f"  Output shape : {list(e_a.shape)}  ✓")

    # Correctness: when all pixels belong to class 1 with weight 1,
    # the weighted average equals the global average of the feature map.
    expected = features.mean(dim=[2, 3])          # [B, C]
    assert torch.allclose(e_a[:, 1, :], expected, atol=1e-5), (
        f"e_a[:, 1, :] does not match global mean. "
        f"Max diff: {(e_a[:, 1, :] - expected).abs().max():.2e}"
    )
    print(f"  e_a[:, 1, :] == global mean  ✓")

    # All other classes should be near zero (weight ≈ 0 → numerator ≈ 0)
    for k in range(num_classes):
        if k == 1:
            continue
        max_val = e_a[:, k, :].abs().max().item()
        assert max_val < 1e-3, (
            f"Class {k} e_a should be ~0, got max abs value {max_val:.2e}"
        )
    print(f"  e_a[:, k≠1, :] ≈ 0          ✓")

    print("\nmasked_average_pooling: PASSED")

    # ── Test 2: soft weighting ────────────────────────────────────────────
    print("\n── Test 2: soft weighting (50/50 split) ──────────────────────")

    features2   = torch.ones(1, C, H, W)      # all features = 1
    S2          = torch.zeros(1, num_classes, H, W)
    # Left half → class 0, right half → class 1 (each with prob 1)
    S2[0, 0, :, :W//2]  = 1.0
    S2[0, 1, :, W//2:]  = 1.0

    e_a2 = masked_average_pooling(features2, S2)

    # Both class 0 and class 1 weighted average of all-ones features = 1
    assert torch.allclose(e_a2[0, 0, :], torch.ones(C), atol=1e-5), \
        "Class 0 embedding should be all-ones."
    assert torch.allclose(e_a2[0, 1, :], torch.ones(C), atol=1e-5), \
        "Class 1 embedding should be all-ones."
    print(f"  50/50 hard split → both class embeddings = 1.0  ✓")

    # ── Test 3: empty class (no pixels assigned) ──────────────────────────
    print("\n── Test 3: empty class (absent organ) ────────────────────────")

    features3   = torch.randn(1, C, H, W)
    S3          = torch.zeros(1, num_classes, H, W)
    S3[0, 0, :, :] = 1.0    # all pixels → class 0

    e_a3 = masked_average_pooling(features3, S3)

    # All other classes have total_weight ≈ 0 → e_a should be near zero
    for k in range(1, num_classes):
        max_val = e_a3[0, k, :].abs().max().item()
        assert max_val < 1e-3, \
            f"Empty class {k}: expected ~0, got {max_val:.2e}"
    print(f"  Absent classes produce zero embeddings  ✓")

    # ── Test 4: gradient flow ─────────────────────────────────────────────
    print("\n── Test 4: gradient flow ─────────────────────────────────────")

    feat_grd = torch.randn(2, C, H, W, requires_grad=True)
    S_grd    = torch.softmax(torch.randn(2, num_classes, H, W), dim=1)

    e_a_grd  = masked_average_pooling(feat_grd, S_grd)
    loss     = e_a_grd.mean()
    loss.backward()

    assert feat_grd.grad is not None, "Gradient did not reach decoder_features."
    gnorm = feat_grd.grad.norm().item()
    assert gnorm > 0, f"Gradient norm is zero."
    print(f"  decoder_features grad norm : {gnorm:.6f}  ✓")

    # ── Test 5: compute_anatomy_embeddings wrapper ────────────────────────
    print("\n── Test 5: compute_anatomy_embeddings wrapper ────────────────")

    fake_output = {
        'decoder_features': torch.randn(2, C, H, W),
        'S':                torch.softmax(torch.randn(2, num_classes, H, W), dim=1),
        'logits':           torch.randn(2, num_classes, H, W),
        'F':                torch.randn(2, 768, 16, 16),
    }
    e_a_wrap = compute_anatomy_embeddings(fake_output)
    assert e_a_wrap.shape == (2, num_classes, C), \
        f"Wrapper output shape wrong: {e_a_wrap.shape}"
    print(f"  Output shape : {list(e_a_wrap.shape)}  ✓")

    # Verify wrapper gives identical result to direct call
    e_a_direct = masked_average_pooling(
        fake_output['decoder_features'], fake_output['S']
    )
    assert torch.allclose(e_a_wrap, e_a_direct), \
        "Wrapper result differs from direct call."
    print(f"  Wrapper == direct call      ✓")

    # Missing key error
    try:
        compute_anatomy_embeddings({'S': fake_output['S']})
        assert False, "Should have raised KeyError."
    except KeyError as exc:
        print(f"  Missing key raises KeyError ✓  ({exc})")

    # ── Test 6: spatial size mismatch raises ──────────────────────────────
    print("\n── Test 6: spatial mismatch raises ValueError ─────────────────")
    try:
        masked_average_pooling(
            torch.randn(2, C, 512, 512),
            torch.softmax(torch.randn(2, num_classes, 256, 256), dim=1),
        )
        assert False, "Should have raised ValueError."
    except ValueError as exc:
        print(f"  Spatial mismatch raises ValueError  ✓")

    # ── Test 7: real-like S from softmax ──────────────────────────────────
    print("\n── Test 7: real-like softmax S ───────────────────────────────")

    features7 = torch.randn(2, C, H, W)
    logits7   = torch.randn(2, num_classes, H, W)
    S7        = torch.softmax(logits7, dim=1)

    e_a7 = masked_average_pooling(features7, S7)
    assert e_a7.shape == (2, num_classes, C)
    # e_a should be finite for real softmax S
    assert torch.isfinite(e_a7).all(), "e_a contains NaN or Inf."
    print(f"  Shape   : {list(e_a7.shape)}  ✓")
    print(f"  Finite  : {torch.isfinite(e_a7).all().item()}  ✓")

    # e_a values should lie within a reasonable range.
    # For softmax weights over large HxW maps, masked means have much lower
    # variance than raw features, so std can naturally be around 1e-3.
    e_a_std = e_a7.std().item()
    assert 1e-4 < e_a_std < 1.0, f"e_a std={e_a_std:.4f} looks wrong."
    print(f"  e_a std : {e_a_std:.4f}  (reasonable range ✓)")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)