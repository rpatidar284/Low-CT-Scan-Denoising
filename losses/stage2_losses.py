"""
losses/stage2_losses.py

Stage 2 progressive loss schedule.

Phase 1  (step 0 – 50k):           L = L_res
Phase 2  (step 50k – 150k):        L = L_res + 0.1 * L_kd
Phase 3  (step 150k+):             L = L_res + 0.1 * L_kd + 0.05 * L_anatomy
                                   (L_anatomy applied every 5th step)
"""

import torch
import torch.nn as nn

try:
    from losses.stage1_losses import SegmentationLoss
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from losses.stage1_losses import SegmentationLoss


class Stage2LossManager:
    PHASE1_END = 50_000
    PHASE2_END = 150_000

    def __init__(self, num_classes=7, label_smoothing=0.1, kd_weight=0.1, anatomy_weight=0.05):
        self.kd_weight = kd_weight
        self.anatomy_weight = anatomy_weight
        self.kd_criterion = SegmentationLoss(num_classes=num_classes, label_smoothing=label_smoothing)
        self.anatomy_criterion = nn.L1Loss()

    def _get_phase(self, step):
        if step < self.PHASE1_END: return 1
        elif step < self.PHASE2_END: return 2
        return 3

    def compute(self, step, loss_res, kd_logits, pseudo_labels, e_a_pred=None, e_a_gt=None):
        phase = self._get_phase(step)
        loss_kd = None
        loss_anatomy = None
        total = loss_res

        if phase >= 2:
            loss_kd = self.kd_criterion(kd_logits, pseudo_labels)
            total = total + self.kd_weight * loss_kd

        if phase >= 3 and step % 5 == 0:
            if e_a_pred is not None and e_a_gt is not None:
                loss_anatomy = self.anatomy_criterion(e_a_pred, e_a_gt)
                total = total + self.anatomy_weight * loss_anatomy

        return {'total': total, 'loss_res': loss_res, 'loss_kd': loss_kd, 'loss_anatomy': loss_anatomy}


if __name__ == '__main__':
    print("=" * 60)
    print("losses/stage2_losses.py — self-test")
    print("=" * 60)

    B, C, H, W = 2, 7, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_res = torch.tensor(1.5, device=device)
    kd_logits = torch.randn(B, C, H, W, device=device)
    pseudo_labels = torch.randint(0, C, (B, H, W), device=device)
    e_a_pred = torch.randn(B, C, 96, device=device)
    e_a_gt = torch.randn(B, C, 96, device=device)

    mgr = Stage2LossManager(num_classes=C)

    print("\n── Phase 1 (step 100) ─────────────────────────────────")
    out = mgr.compute(100, loss_res, kd_logits, pseudo_labels, e_a_pred, e_a_gt)
    assert out['loss_kd'] is None and out['loss_anatomy'] is None
    assert torch.allclose(out['total'], loss_res)
    print(f"  total: {out['total']:.4f} ✓ (L_res only)")

    print("\n── Phase 2 (step 60000) ──────────────────────────────")
    out2 = mgr.compute(60000, loss_res, kd_logits, pseudo_labels)
    assert out2['loss_kd'] is not None and out2['loss_anatomy'] is None
    assert out2['total'] > loss_res
    print(f"  total: {out2['total']:.4f}, L_kd: {out2['loss_kd']:.4f} ✓")

    print("\n── Phase 3 (step 150005, every 5th) ─────────────────")
    out3 = mgr.compute(150005, loss_res, kd_logits, pseudo_labels, e_a_pred, e_a_gt)
    assert out3['loss_kd'] is not None and out3['loss_anatomy'] is not None
    print(f"  total: {out3['total']:.4f}, L_kd: {out3['loss_kd']:.4f}, L_anatomy: {out3['loss_anatomy']:.4f} ✓")

    print("\n── Phase 3 (step 150001, not 5th) ────────────────────")
    out4 = mgr.compute(150001, loss_res, kd_logits, pseudo_labels, e_a_pred, e_a_gt)
    assert out4['loss_anatomy'] is None, "L_anatomy should be inactive on non-5th step"
    print(f"  L_anatomy inactive ✓")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
