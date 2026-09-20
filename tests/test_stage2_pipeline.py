"""
tests/test_stage2_pipeline.py

Integration tests for the Stage 2 anatomy-conditioned denoiser.
===============================================================
Exercises the full data-flow:
  DummyCTDenoisingDataset → Stage2Model → ResidualDiffusion
  → VSSDDenoiser (SpatialFiLM + cross-attention) → Stage2LossManager

No real CT data required; image_size = 32 so every test runs in seconds.
Exits with code 1 if any test failed.
"""

import sys
import traceback
from pathlib import Path
from typing import List

import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vssd_denoiser import VSSDDenoiser, _build_S_scales
from models.diffusion import ResidualDiffusion
from models.stage2 import Stage2Model
from losses.stage2_losses import Stage2LossManager
from datapy.dataset import DummyCTDataset

IMAGE_SIZE = 32
BATCH_SIZE = 1
NUM_CLASSES = 7
ANATOMY_DIM = 96

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_stage2():
    return Stage2Model(
        stage1_checkpoint=None,
        denoiser_kwargs={'image_size': IMAGE_SIZE},
        diffusion_kwargs={'timesteps': 50, 'sampling_timesteps': 5},
        image_size=IMAGE_SIZE,
    ).to(DEVICE)


def _make_S():
    return torch.rand(BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE,
                      device=DEVICE).softmax(dim=1)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def test_1_denoiser_shape_flow():
    _section("Test 1: VSSD denoiser shape flow")
    model = VSSDDenoiser(image_size=IMAGE_SIZE).to(DEVICE)
    x_ldct = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    x_noisy = torch.randn_like(x_ldct)
    t = torch.randint(0, 1000, (BATCH_SIZE,), device=DEVICE)
    S_scales = _build_S_scales(_make_S(), model.scale_sizes)
    e_a = torch.randn(BATCH_SIZE, NUM_CLASSES, ANATOMY_DIM, device=DEVICE)

    pred_res, pred_noise, kd_logits = model(x_ldct, x_noisy, t, S_scales, e_a)

    assert pred_res.shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert pred_noise.shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert kd_logits.shape == (BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(pred_res).all() and torch.isfinite(pred_noise).all()
    print(f"  pred_res   : {list(pred_res.shape)}  ✓")
    print(f"  pred_noise : {list(pred_noise.shape)}  ✓")
    print(f"  kd_logits  : {list(kd_logits.shape)}  ✓")
    return True, "VSSD denoiser shape flow"


def test_2_training_loss():
    _section("Test 2: Training loss (L_res + L_noise)")
    model = _make_stage2()
    x_ldct = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    x_hdct = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    out = model(x_ldct, x_hdct, mode='train')

    assert torch.isfinite(out['loss_res']) and out['loss_res'].item() >= 0
    assert torch.isfinite(out['loss_noise']) and out['loss_noise'].item() >= 0
    print(f"  L_res={out['loss_res']:.4f}  L_noise={out['loss_noise']:.4f}  ✓")
    return True, "Training loss"


def test_3_ddim_inference():
    _section("Test 3: DDIM inference")
    model = _make_stage2()
    model.stage1.eval()
    x_ldct = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    out = model(x_ldct, mode='inference')

    assert out['x_denoised'].shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(out['x_denoised']).all()
    print(f"  x_denoised : {list(out['x_denoised'].shape)}  ✓")
    return True, "DDIM inference"


def test_4_gradient_isolation():
    _section("Test 4: Gradient isolation (Stage 1 frozen)")
    model = _make_stage2()
    x_ldct = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    x_hdct = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    out = model(x_ldct, x_hdct, mode='train')
    (out['loss_res'] + out['loss_noise']).backward()

    s1_grads = [p for p in model.stage1.parameters() if p.grad is not None]
    den_grads = [p for p in model.denoiser.parameters()
                 if p.grad is not None and p.grad.norm().item() > 0]
    assert len(s1_grads) == 0, "Stage 1 received gradients"
    assert len(den_grads) > 0, "Denoiser received no gradients"
    print(f"  Stage 1 grads : {len(s1_grads)}  ✓")
    print(f"  Denoiser grads: {len(den_grads)}  ✓")
    return True, "Gradient isolation"


def test_5_progressive_losses():
    _section("Test 5: Progressive loss schedule")
    mgr = Stage2LossManager(num_classes=NUM_CLASSES)
    loss_res = torch.tensor(1.0, device=DEVICE)
    kd_logits = torch.randn(BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
    e_a_pred = torch.randn(BATCH_SIZE, NUM_CLASSES, ANATOMY_DIM, device=DEVICE)
    e_a_gt = torch.randn(BATCH_SIZE, NUM_CLASSES, ANATOMY_DIM, device=DEVICE)

    p1 = mgr.compute(100, loss_res, kd_logits, labels, e_a_pred, e_a_gt)
    assert p1['loss_kd'] is None and p1['loss_anatomy'] is None

    p2 = mgr.compute(mgr.PHASE2_END - 1, loss_res, kd_logits, labels)
    assert p2['loss_kd'] is not None and p2['loss_anatomy'] is None

    p3 = mgr.compute(mgr.PHASE2_END + 5, loss_res, kd_logits, labels, e_a_pred, e_a_gt)
    assert p3['loss_kd'] is not None and p3['loss_anatomy'] is not None

    print(f"  phase1: L_res only            ✓")
    print(f"  phase2: +L_kd                 ✓")
    print(f"  phase3: +L_kd +L_anatomy      ✓")
    return True, "Progressive loss schedule"


def test_6_dataset_compatibility():
    _section("Test 6: Dataset compatibility")
    ds = DummyCTDataset(length=4, image_size=IMAGE_SIZE)
    batch = ds[0]
    assert batch['ldct'].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert batch['ndct'].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert batch['mask'].shape == (IMAGE_SIZE, IMAGE_SIZE)
    print(f"  ldct/ndct/mask shapes  ✓")
    return True, "Dataset compatibility"


def run_all_tests() -> None:
    print("=" * 60)
    print(" Stage 2 Integration Test Suite")
    print(f" device     = {DEVICE}")
    print(f" image_size = {IMAGE_SIZE}")
    print("=" * 60)

    tests = [
        test_1_denoiser_shape_flow,
        test_2_training_loss,
        test_3_ddim_inference,
        test_4_gradient_isolation,
        test_5_progressive_losses,
        test_6_dataset_compatibility,
    ]

    results: List[tuple] = []
    for test_fn in tests:
        try:
            passed, name = test_fn()
            results.append((name, True, ''))
        except Exception:
            name = test_fn.__name__
            err = traceback.format_exc()
            print(f"\n{'!' * 60}\n  {name} FAILED\n{err}{'!' * 60}")
            results.append((name, False, err))

    print()
    print("=" * 60)
    print(" Results summary")
    print("=" * 60)
    all_passed = True
    for name, passed, _ in results:
        print(f"  {'PASSED ✓' if passed else 'FAILED ✗'}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All Stage 2 integration tests PASSED")
        sys.exit(0)
    n_failed = sum(1 for _, p, _ in results if not p)
    print(f"{n_failed} test(s) FAILED — see output above.")
    sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
