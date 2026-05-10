"""
models/stage2.py

Stage 2: Anatomy-Conditioned VSSD Denoiser.
=============================================
Complete Stage 2 model that wraps:
  1. Frozen Stage 1 (VM-UNet) — produces S, e_a from LDCT
  2. VSSD Denoiser — predicts residual given anatomy conditioning
  3. Residual Diffusion — noise schedule and loss computation

Two modes:
  Training:
    Given NDCT and LDCT, runs frozen Stage 1 → S, e_a
    Forward diffusion adds noise to residual
    Denoiser predicts residual → loss computation

  Inference:
    Given LDCT only, runs frozen Stage 1 → S, e_a
    Iterative DDIM denoising starting from noise
    Returns x_denoised ≈ NDCT

Reference: Architecture.pdf — Stage 2 (VSSD Denoiser)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.stage1 import load_stage1_frozen
    from models.vssd_denoiser import VSSDDenoiser, _build_S_scales
    from models.diffusion import ResidualDiffusion
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.stage1 import load_stage1_frozen
    from models.vssd_denoiser import VSSDDenoiser, _build_S_scales
    from models.diffusion import ResidualDiffusion


class Stage2Model(nn.Module):
    """
    Complete Stage 2 denoising pipeline.

    Parameters
    ----------
    stage1_checkpoint : str or None
        Path to trained Stage 1 checkpoint. If None, uses random init Stage 1
        (for testing only — real training requires a trained Stage 1).
    denoiser_kwargs : dict
        Forwarded to VSSDDenoiser constructor.
    diffusion_kwargs : dict
        Forwarded to ResidualDiffusion constructor.
    image_size : int
        Input image spatial resolution (default 512).
    freeze_stage1 : bool
        Whether to freeze Stage 1 (always True for real training).
    """

    def __init__(
        self,
        stage1_checkpoint: str = None,
        denoiser_kwargs: dict = None,
        diffusion_kwargs: dict = None,
        image_size: int = 512,
        freeze_stage1: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        # ── Stage 1 (frozen anatomy teacher) ────────────────────────────────
        if stage1_checkpoint is not None:
            self.stage1 = load_stage1_frozen(stage1_checkpoint, device='cpu')
        else:
            # Dummy Stage 1 for testing — will be overwritten later
            from models.stage1 import Stage1Model
            self.stage1 = Stage1Model()
            if freeze_stage1:
                for p in self.stage1.parameters():
                    p.requires_grad_(False)

        # ── VSSD Denoiser ───────────────────────────────────────────────────
        denoiser_kwargs = denoiser_kwargs or {}
        denoiser_kwargs.setdefault('image_size', image_size)
        self.denoiser = VSSDDenoiser(**denoiser_kwargs)

        # Cache S scale sizes
        self.scale_sizes = self.denoiser.scale_sizes

        # ── Residual Diffusion ──────────────────────────────────────────────
        diffusion_kwargs = diffusion_kwargs or {}
        self.diffusion = ResidualDiffusion(**diffusion_kwargs)

    def get_conditioning(self, x_ldct: torch.Tensor):
        """
        Run frozen Stage 1 to get anatomy conditioning.

        Parameters
        ----------
        x_ldct : [B, 1, H, W]

        Returns
        -------
        S_scales : list of [B, 7, H_i, W_i]
        e_a : [B, 7, C_anat]
        """
        with torch.no_grad():
            S, e_a = self.stage1.get_anatomy_conditioning(x_ldct)
        S_scales = _build_S_scales(S, self.scale_sizes)
        return S_scales, e_a

    def forward(
        self,
        x_ldct: torch.Tensor,
        x_ndct: torch.Tensor = None,
        mode: str = 'train',
    ):
        """
        Parameters
        ----------
        x_ldct : [B, 1, H, W]  low-dose CT
        x_ndct : [B, 1, H, W]  normal-dose CT (required for train mode)
        mode : str              'train' or 'inference'

        Returns
        -------
        dict with keys depending on mode:
          train:
            'loss'       — scalar diffusion loss L_res
            'pred_res'   — [B, 1, H, W] predicted residual
            'kd_logits'  — [B, 7, H, W] seg KD head output
          inference:
            'x_denoised' — [B, 1, H, W] denoised output
        """
        B = x_ldct.shape[0]
        device = x_ldct.device

        # ── Get anatomy conditioning from frozen Stage 1 ─────────────────
        S_scales, e_a = self.get_conditioning(x_ldct)

        if mode == 'train':
            assert x_ndct is not None, "x_ndct required for training"

            # True residual
            x_res = x_ldct - x_ndct

            # Sample random timestep
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device).long()

            # Forward diffusion: add noise to residual
            noise = torch.randn_like(x_res)
            x_noisy = self.diffusion.q_sample(x_ldct, x_res, t, noise)

            # Denoiser predicts the residual
            pred_res, kd_logits = self.denoiser(x_ldct, x_noisy, t, S_scales, e_a)

            # L_res
            loss_res = self.diffusion.training_loss(pred_res, x_res)

            return {
                'loss_res': loss_res,
                'pred_res': pred_res,
                'kd_logits': kd_logits,
                'S_scales': S_scales,
                'e_a': e_a,
                't': t,
            }

        elif mode == 'inference':
            # DDIM denoising from noise to clean residual
            B, C, H, W = B, x_ldct.shape[1], self.image_size, self.image_size
            shape = (B, C, H, W)

            x_denoised = self.diffusion.ddim_sample(
                model=self.denoiser,
                x_ldct=x_ldct,
                S_scales=S_scales,
                e_a=e_a,
                shape=shape,
            )

            return {
                'x_denoised': x_denoised,
                'S_scales': S_scales,
                'e_a': e_a,
            }

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'inference'.")

    @torch.no_grad()
    def denoise(self, x_ldct: torch.Tensor) -> torch.Tensor:
        """Shortcut: denoise a single LDCT image."""
        out = self.forward(x_ldct, mode='inference')
        return out['x_denoised']

    def count_parameters(self) -> dict:
        def _n(m):
            return sum(p.numel() for p in m.parameters())

        return {
            'stage1': _n(self.stage1),
            'denoiser': _n(self.denoiser),
            'diffusion': _n(self.diffusion),
            'total': _n(self),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("models/stage2.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, C, H, W = 2, 1, 256, 256

    x_ldct = torch.randn(B, C, H, W, device=device)
    x_ndct = torch.randn(B, C, H, W, device=device)

    # Create Stage 2 with dummy Stage 1
    model = Stage2Model(
        stage1_checkpoint=None,
        denoiser_kwargs={'image_size': H},
        diffusion_kwargs={'timesteps': 100, 'loss_type': 'l2'},
        image_size=H,
    ).to(device)

    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k:<12} : {v/1e6:>7.1f}M")

    # ── Training mode ────────────────────────────────────────────────────
    print("\n── Training mode ──────────────────────────────────────────────")
    out = model(x_ldct, x_ndct, mode='train')

    assert out['loss_res'].shape == (), f"loss_res: {out['loss_res'].shape}"
    assert out['pred_res'].shape == (B, C, H, W), f"pred_res: {out['pred_res'].shape}"
    assert out['kd_logits'].shape == (B, 7, H, W), f"kd_logits: {out['kd_logits'].shape}"
    assert out['loss_res'].item() > 0
    print(f"  loss_res : {out['loss_res'].item():.4f} ✓")
    print(f"  pred_res : {tuple(out['pred_res'].shape)} ✓")
    print(f"  kd_logits: {tuple(out['kd_logits'].shape)} ✓")
    print(f"  S_scales : {len(out['S_scales'])} scales")
    print(f"  e_a      : {tuple(out['e_a'].shape)} ✓")

    # ── Inference mode ───────────────────────────────────────────────────
    print("\n── Inference mode ────────────────────────────────────────────")
    model.eval()
    out_inf = model(x_ldct, mode='inference')
    assert out_inf['x_denoised'].shape == (B, C, H, W)
    assert torch.isfinite(out_inf['x_denoised']).all()
    print(f"  x_denoised: {tuple(out_inf['x_denoised'].shape)} ✓")

    # ── Gradient flow (only Stage 2 params) ──────────────────────────────
    print("\n── Gradient isolation ─────────────────────────────────────────")
    # Stage 1 should be frozen
    stage1_trainable = sum(p.numel() for p in model.stage1.parameters() if p.requires_grad)
    assert stage1_trainable == 0, f"Stage 1 has {stage1_trainable} trainable params!"
    print(f"  Stage 1 trainable params: {stage1_trainable} ✓ (frozen)")

    # Stage 2 should have gradients
    out_train = model(x_ldct, x_ndct, mode='train')
    total_loss = out_train['loss_res']
    total_loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.denoiser.parameters() if p.grad is not None)
    assert gnorm > 0, "No gradients in denoiser!"
    print(f"  Denoiser gradient norm: {gnorm:.4f} ✓")

    # Stage 1 should have no gradients
    s1_grad = [p.grad for p in model.stage1.parameters() if p.grad is not None]
    assert len(s1_grad) == 0, f"Stage 1 has {len(s1_grad)} gradients!"
    print(f"  Stage 1 gradients: {len(s1_grad)} ✓ (none)")

    print(f"\n{'='*60}")
    print("All tests PASSED")
    print("=" * 60)
