"""
tests/test_stage1_pipeline.py

Integration tests for all Stage 1 components working together.
==============================================================
Tests the complete data-flow through:
  Stage1Model → SegmentationLoss → BYOL → masked_average_pooling
  → load_stage1_frozen

Design constraints
------------------
* No real CT data required — uses DummyCTDataset / random tensors.
* image_size = 64 (not 512) so every test runs in seconds on any GPU.
* Each test is self-contained: a failure in one test does not prevent
  the others from running (but is reported at the end).
* Exits with code 1 if any test failed.
"""

import sys
import copy
import tempfile
import traceback
from pathlib import Path
from typing  import List

import torch
import torch.nn as nn

# ── Make the project root importable regardless of cwd ───────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stage1        import Stage1Model, load_stage1_frozen
from losses.stage1_losses import SegmentationLoss, compute_dice_per_class
from models.byol          import get_ema_tau
from datapy.dataset        import DummyCTDataset
from utils.masking        import masked_average_pooling

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE   = 64    # keep small so tests finish quickly
BATCH_SIZE   = 2
NUM_CLASSES  = 7
EMBED_DIM    = 96    # must match default Stage1Model → e_a dim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_model() -> Stage1Model:
    """Return a freshly initialised Stage1Model on DEVICE."""
    return Stage1Model(
        in_channels = 1,
        num_classes = NUM_CLASSES,
        embed_dim   = EMBED_DIM,
        depths      = [2, 2, 2, 2],
    ).to(DEVICE)


def _make_batch():
    """
    Return (x, mask) — two random tensors that mimic a real data batch.
      x    : [B, 1, H, W]  float32 in [0, 1]   (normalised CT slice)
      mask : [B, H, W]     int64   in {0..6}   (organ labels)
    """
    x    = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    mask = torch.randint(0, NUM_CLASSES,
                         (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE),
                         device=DEVICE)
    return x, mask


def _has_nan_grad(model: nn.Module) -> bool:
    """Return True if any parameter gradient contains NaN."""
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def _section(title: str) -> None:
    bar = '─' * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual tests — each returns (passed: bool, message: str)
# ─────────────────────────────────────────────────────────────────────────────

def test_1_shape_flow():
    """
    Test 1: Shape flow
    ------------------
    Verify that every tensor in the forward-pass output dict has the
    expected shape, dtype, and value range.
    """
    _section("Test 1: Shape flow")

    model = _make_model()
    model.eval()
    x, _ = _make_batch()

    with torch.no_grad():
        out = model(x, return_byol=False)

    # ── Required keys ─────────────────────────────────────────────────────
    required_keys = {'logits', 'S', 'e_a', 'F', 'decoder_features', 'byol_loss'}
    missing = required_keys - set(out.keys())
    assert not missing, f"Missing output keys: {missing}"

    # ── Shapes ────────────────────────────────────────────────────────────
    B, C, H, W = BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE

    assert out['logits'].shape           == (B, C, H, W),   \
        f"logits: {out['logits'].shape}"
    assert out['S'].shape                == (B, C, H, W),   \
        f"S: {out['S'].shape}"
    assert out['e_a'].shape              == (B, C, EMBED_DIM), \
        f"e_a: {out['e_a'].shape}"
    assert out['F'].shape                == (B, 768, H // 32, W // 32), \
        f"F: {out['F'].shape}"
    assert out['decoder_features'].shape == (B, EMBED_DIM, H, W), \
        f"decoder_features: {out['decoder_features'].shape}"
    assert out['byol_loss'] is None, \
        "byol_loss should be None when return_byol=False"

    # ── Dtypes ────────────────────────────────────────────────────────────
    assert out['logits'].dtype == torch.float32
    assert out['S'].dtype      == torch.float32
    assert out['e_a'].dtype    == torch.float32

    # ── Value range of S ──────────────────────────────────────────────────
    assert out['S'].min() >= 0.0,  f"S min = {out['S'].min():.4f} < 0"
    assert out['S'].max() <= 1.0,  f"S max = {out['S'].max():.4f} > 1"

    ones = torch.ones(B, H, W, device=DEVICE)
    s_sum = out['S'].sum(dim=1)
    assert torch.allclose(s_sum, ones, atol=1e-5), \
        f"S does not sum to 1; max dev = {(s_sum - ones).abs().max():.2e}"

    # ── e_a is finite ─────────────────────────────────────────────────────
    assert torch.isfinite(out['e_a']).all(), "e_a contains NaN or Inf"

    print(f"  logits           : {list(out['logits'].shape)}  ✓")
    print(f"  S                : {list(out['S'].shape)}  ✓")
    print(f"  e_a              : {list(out['e_a'].shape)}  ✓")
    print(f"  F                : {list(out['F'].shape)}  ✓")
    print(f"  decoder_features : {list(out['decoder_features'].shape)}  ✓")
    print(f"  S in [0,1]       ✓")
    print(f"  S sums to 1      ✓")
    print(f"  e_a finite       ✓")
    print("Test 1 PASSED: Shape flow")
    return True, "Shape flow"


def test_2_loss_computation():
    """
    Test 2: Loss computation
    ------------------------
    Verify that SegmentationLoss accepts the model's logits and returns
    a finite, positive scalar.
    """
    _section("Test 2: Loss computation")

    model     = _make_model()
    criterion = SegmentationLoss(num_classes=NUM_CLASSES, label_smoothing=0.1)
    x, mask   = _make_batch()

    with torch.no_grad():
        out = model(x, return_byol=False)

    loss = criterion(out['logits'], mask)

    # ── Scalar ────────────────────────────────────────────────────────────
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"

    # ── Finite and positive ───────────────────────────────────────────────
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0,      f"Loss should be > 0, got {loss.item()}"

    # ── Reasonable magnitude ────────────────────────────────────────────────
    # Mean CE with random targets is ~ log(num_classes) for calibrated logits,
    # but VM-UNet/Mamba heads can emit larger logits at init → higher CE.
    assert loss.item() < 80.0, f"Loss is suspiciously large: {loss.item()}"

    print(f"  L_seg = {loss.item():.4f}  (finite, positive, < 80)  ✓")
    print("Test 2 PASSED: Loss computation")
    return True, "Loss computation"


def test_3_byol_with_backward():
    """
    Test 3: BYOL loss + backward pass
    ----------------------------------
    Verify that the symmetric BYOL loss is finite and in [0, 8], that
    the combined loss back-propagates without NaN gradients, and that
    gradients reach the backbone.
    """
    _section("Test 3: BYOL with backward")

    model     = _make_model()
    criterion = SegmentationLoss(num_classes=NUM_CLASSES, label_smoothing=0.1)
    x, mask   = _make_batch()

    # Forward with BYOL active
    out = model(x, return_byol=True)

    # ── byol_loss sanity ──────────────────────────────────────────────────
    assert out['byol_loss'] is not None, "byol_loss is None with return_byol=True"
    assert out['byol_loss'].shape == (), \
        f"byol_loss should be scalar, got {out['byol_loss'].shape}"
    byol_val = out['byol_loss'].item()
    # Symmetric BYOL loss is sum of two directions: range [0, 8]
    assert torch.isfinite(out['byol_loss']), \
        f"byol_loss is not finite: {byol_val}"
    assert 0 <= byol_val <= 8.0, \
        f"byol_loss {byol_val:.4f} outside [0, 8]"

    # ── Combined loss ─────────────────────────────────────────────────────
    l_seg      = criterion(out['logits'], mask)
    total_loss = l_seg + 0.1 * out['byol_loss']

    assert torch.isfinite(total_loss), \
        f"total_loss is not finite: {total_loss.item()}"

    # ── Backward ──────────────────────────────────────────────────────────
    total_loss.backward()

    # ── No NaN gradients ──────────────────────────────────────────────────
    assert not _has_nan_grad(model), \
        "NaN gradient found in at least one parameter after backward."

    # ── Gradients reached the backbone ───────────────────────────────────
    backbone_grads = [
        p.grad
        for p in model.backbone.parameters()
        if p.grad is not None and p.grad.norm().item() > 0
    ]
    assert len(backbone_grads) > 0, \
        "No backbone parameter received a non-zero gradient."

    # ── Target projector must NOT have gradients ──────────────────────────
    for name, p in model.byol.target_projector.named_parameters():
        assert not p.requires_grad, \
            f"Target projector param '{name}' has requires_grad=True."
        assert p.grad is None, \
            f"Target projector param '{name}' received a gradient."

    print(f"  L_seg     = {l_seg.item():.4f}  ✓")
    print(f"  L_byol    = {byol_val:.4f}  (in [0, 8])  ✓")
    print(f"  total     = {total_loss.item():.4f}  ✓")
    print(f"  No NaN gradients  ✓")
    print(f"  Backbone params with grad: {len(backbone_grads)}  ✓")
    print(f"  Target projector untouched  ✓")
    print("Test 3 PASSED: BYOL with backward")
    return True, "BYOL with backward"


def test_4_ema_update():
    """
    Test 4: EMA update
    ------------------
    Verify that update_target_projector(tau) moves the target projector
    weights toward the online projector without making them identical.
    """
    _section("Test 4: EMA update")

    model = _make_model()
    tau   = 0.99

    # ── Snapshot of target weights before update ──────────────────────────
    target_before = {
        name: p.data.clone()
        for name, p in model.byol.target_projector.named_parameters()
    }

    # ── Perturb online projector so the two sets differ ───────────────────
    with torch.no_grad():
        for p in model.byol.online_projector.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # ── EMA update ────────────────────────────────────────────────────────
    model.byol.update_target_projector(tau)

    # ── Verify ────────────────────────────────────────────────────────────
    any_changed = False
    for name, p_target in model.byol.target_projector.named_parameters():
        p_online   = dict(model.byol.online_projector.named_parameters())[name]
        expected   = tau * target_before[name] + (1.0 - tau) * p_online.data

        # Target must match the EMA formula exactly
        assert torch.allclose(p_target.data, expected, atol=1e-6), \
            f"EMA formula violated for param '{name}'"

        # Target must NOT be identical to online (because tau < 1)
        if not torch.equal(p_target.data, p_online.data):
            any_changed = True

        # Step size ‖Δθ‖ = (1−τ)·‖θ_online − θ_target_before‖ (not bounded by 1−τ).
        step = p_target.data - target_before[name]
        expected_step = (1.0 - tau) * (p_online.data - target_before[name])
        assert torch.allclose(step, expected_step, atol=1e-5, rtol=1e-4), \
            f"EMA step mismatch for param '{name}'"

    assert any_changed, \
        "Target projector was not updated (still identical to online)."

    # ── Target projector must still have no requires_grad ─────────────────
    for name, p in model.byol.target_projector.named_parameters():
        assert not p.requires_grad, \
            f"Target param '{name}' has requires_grad=True after EMA update."

    print(f"  EMA formula verified for all target params  ✓")
    print(f"  Target ≠ online after update  ✓")
    print(f"  Target still requires_grad=False  ✓")
    print("Test 4 PASSED: EMA update")
    return True, "EMA update"


def test_5_anatomy_conditioning():
    """
    Test 5: Anatomy conditioning
    ----------------------------
    Verify get_anatomy_conditioning returns valid S and e_a tensors and
    that the per-class embeddings are not identical across organ classes.
    """
    _section("Test 5: Anatomy conditioning")

    model = _make_model()
    model.eval()
    x, _ = _make_batch()

    with torch.no_grad():
        S, e_a = model.get_anatomy_conditioning(x)

    # ── Shapes ────────────────────────────────────────────────────────────
    B, H, W = BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE
    assert S.shape   == (B, NUM_CLASSES, H, W), f"S: {S.shape}"
    assert e_a.shape == (B, NUM_CLASSES, EMBED_DIM), f"e_a: {e_a.shape}"

    # ── S sums to 1 per pixel ─────────────────────────────────────────────
    ones  = torch.ones(B, H, W, device=S.device)
    s_sum = S.sum(dim=1)
    assert torch.allclose(s_sum, ones, atol=1e-5), \
        f"S does not sum to 1; max dev = {(s_sum - ones).abs().max():.2e}"

    # ── S in [0, 1] ───────────────────────────────────────────────────────
    assert S.min() >= 0.0 and S.max() <= 1.0

    # ── e_a has no NaN / Inf ──────────────────────────────────────────────
    assert torch.isfinite(e_a).all(), "e_a contains NaN or Inf"

    # ── Per-class embeddings are not all identical ─────────────────────────
    # Compare class 0 to all other classes; at least one must differ.
    ref        = e_a[:, 0, :]   # [B, EMBED_DIM]
    all_same   = all(
        torch.allclose(e_a[:, k, :], ref, atol=1e-6)
        for k in range(1, NUM_CLASSES)
    )
    assert not all_same, \
        "e_a[:, k, :] is identical for all classes — pooling is degenerate."

    # ── No gradient tracked (called under no_grad inside the method) ──────
    assert not S.requires_grad,   "S should not require grad in inference mode"
    assert not e_a.requires_grad, "e_a should not require grad in inference mode"

    print(f"  S shape    : {list(S.shape)}  ✓")
    print(f"  e_a shape  : {list(e_a.shape)}  ✓")
    print(f"  S sums to 1  ✓")
    print(f"  S in [0,1]   ✓")
    print(f"  e_a finite   ✓")
    print(f"  e_a differs across classes  ✓")
    print(f"  No gradient tracking in inference  ✓")
    print("Test 5 PASSED: Anatomy conditioning")
    return True, "Anatomy conditioning"


def test_6_frozen_model():
    """
    Test 6: Frozen model
    --------------------
    Save a checkpoint, reload with load_stage1_frozen(), and verify:
      * All parameters have requires_grad = False.
      * The model is in eval mode.
      * A forward pass produces outputs identical to the original model.
      * No computation graph is built for the frozen model.
    """
    _section("Test 6: Frozen model")

    model = _make_model()
    model.eval()
    x, _ = _make_batch()

    # ── Reference output from the original model ──────────────────────────
    with torch.no_grad():
        ref_out = model(x, return_byol=False)

    # ── Save checkpoint ───────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}")

        # ── Load frozen model ─────────────────────────────────────────────
        frozen = load_stage1_frozen(
            checkpoint_path = ckpt_path,
            device          = str(DEVICE),
        )

        # ── requires_grad = False for every parameter ─────────────────────
        n_trainable = sum(
            p.numel() for p in frozen.parameters() if p.requires_grad
        )
        assert n_trainable == 0, \
            f"{n_trainable} parameters still have requires_grad=True."

        # ── Eval mode ─────────────────────────────────────────────────────
        assert not frozen.training, \
            "Frozen model should be in eval mode."

        # ── Identical outputs ─────────────────────────────────────────────
        with torch.no_grad():
            frz_out = frozen(x, return_byol=False)

        assert torch.allclose(ref_out['S'],   frz_out['S'],   atol=1e-6), \
            "S differs between original and frozen model."
        assert torch.allclose(ref_out['e_a'], frz_out['e_a'], atol=1e-6), \
            "e_a differs between original and frozen model."

        # ── No computation graph built ────────────────────────────────────
        # If all params have requires_grad=False, .backward() on the output
        # should raise RuntimeError ("does not require grad").
        out_s = frz_out['S']
        try:
            out_s.sum().backward()
            # If we get here, a graph was built — that is wrong.
            assert False, \
                "backward() succeeded on frozen model output — graph leaked."
        except RuntimeError:
            pass   # Expected: "element 0 of tensors does not require grad"

        print(f"  Trainable params after freeze: {n_trainable}  ✓")
        print(f"  Eval mode enforced  ✓")
        print(f"  Loaded S  matches original  ✓")
        print(f"  Loaded e_a matches original  ✓")
        print(f"  No computation graph in frozen forward pass  ✓")

    finally:
        import os
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)

    print("Test 6 PASSED: Frozen model")
    return True, "Frozen model"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    """
    Execute every test in sequence.  Report results and exit with code 1
    if any test failed.
    """
    print("=" * 60)
    print(" Stage 1 Integration Test Suite")
    print(f" device     = {DEVICE}")
    print(f" image_size = {IMAGE_SIZE}")
    print(f" batch_size = {BATCH_SIZE}")
    print("=" * 60)

    tests = [
        test_1_shape_flow,
        test_2_loss_computation,
        test_3_byol_with_backward,
        test_4_ema_update,
        test_5_anatomy_conditioning,
        test_6_frozen_model,
    ]

    results: List[tuple] = []   # (test_name, passed, error_msg)

    for test_fn in tests:
        try:
            passed, name = test_fn()
            results.append((name, True, ''))
        except Exception:
            name = test_fn.__name__
            err  = traceback.format_exc()
            print(f"\n{'!'*60}")
            print(f"  {name} FAILED")
            print(err)
            print(f"{'!'*60}")
            results.append((name, False, err))

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" Results summary")
    print("=" * 60)
    all_passed = True
    for name, passed, _ in results:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("=" * 29)
        print("All Stage 1 integration tests PASSED")
        print("=" * 29)
        sys.exit(0)
    else:
        n_failed = sum(1 for _, p, _ in results if not p)
        print(f"{n_failed} test(s) FAILED — see output above.")
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()