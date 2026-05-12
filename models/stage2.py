"""
models/stage2.py

Stage 2: Anatomy-Conditioned VSSD Denoiser.

Wraps frozen Stage 1 (S, e_a from LDCT) + VSSDDenoiser + ResidualDiffusion.

Training: get S/e_a from frozen Stage1, forward diffusion, predict res+noise.
Inference: DDIM reverse from LDCT+noise → x_hdct (denoised).
"""

import torch
import torch.nn as nn

try:
    from models.stage1 import load_stage1_frozen, Stage1Model
    from models.vssd_denoiser import VSSDDenoiser, _build_S_scales
    from models.diffusion import ResidualDiffusion
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.stage1 import load_stage1_frozen, Stage1Model
    from models.vssd_denoiser import VSSDDenoiser, _build_S_scales
    from models.diffusion import ResidualDiffusion


class Stage2Model(nn.Module):
    def __init__(self, stage1_checkpoint=None, denoiser_kwargs=None,
                 diffusion_kwargs=None, image_size=512, freeze_stage1=True):
        super().__init__()
        self.image_size = image_size

        if stage1_checkpoint is not None:
            self.stage1 = load_stage1_frozen(stage1_checkpoint, device='cpu')
        else:
            self.stage1 = Stage1Model()
            if freeze_stage1:
                for p in self.stage1.parameters():
                    p.requires_grad_(False)

        denoiser_kwargs = denoiser_kwargs or {}
        denoiser_kwargs.setdefault('image_size', image_size)
        self.denoiser = VSSDDenoiser(**denoiser_kwargs)
        self.scale_sizes = self.denoiser.scale_sizes

        diffusion_kwargs = diffusion_kwargs or {}
        self.diffusion = ResidualDiffusion(**diffusion_kwargs)

    @torch.no_grad()
    def get_conditioning(self, x):
        S, e_a = self.stage1.get_anatomy_conditioning(x)
        return _build_S_scales(S, self.scale_sizes), e_a

    def forward(self, x_ldct, x_hdct=None, mode='train'):
        B, device = x_ldct.shape[0], x_ldct.device

        if mode == 'train':
            assert x_hdct is not None

            # Normalize to [-1, 1]
            ldct_n = self.diffusion.normalize(x_ldct)
            hdct_n = self.diffusion.normalize(x_hdct)

            # Anatomy from frozen Stage 1 (run on HDCT at train time)
            with torch.no_grad():
                S_scales, e_a = self.get_conditioning(hdct_n)

            x_res = ldct_n - hdct_n
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device).long()
            noise = torch.randn_like(x_res)
            x_noisy = self.diffusion.q_sample(hdct_n, x_res, t, noise)

            pred_res, pred_noise, kd_logits = self.denoiser(ldct_n, x_noisy, t, S_scales, e_a)

            losses = self.diffusion.training_losses(pred_res, pred_noise, x_res, noise)

            return {'loss_res': losses['loss_res'], 'loss_noise': losses['loss_noise'],
                    'pred_res': pred_res, 'pred_noise': pred_noise,
                    'kd_logits': kd_logits, 'S_scales': S_scales, 'e_a': e_a}

        elif mode == 'inference':
            B, C, H, W = B, 1, self.image_size, self.image_size

            ldct_n = self.diffusion.normalize(x_ldct.view(B, C, H, W))

            with torch.no_grad():
                S_scales, e_a = self.get_conditioning(ldct_n)

            shape = (B, C, H, W)
            x_denoised_n = self.diffusion.ddim_sample(self.denoiser, ldct_n, S_scales, e_a, shape)
            x_denoised = self.diffusion.unnormalize(x_denoised_n)

            return {'x_denoised': x_denoised, 'S_scales': S_scales, 'e_a': e_a}

        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def denoise(self, x_ldct):
        out = self.forward(x_ldct, mode='inference')
        return out['x_denoised']


if __name__ == '__main__':
    print("=" * 60)
    print("models/stage2.py — self-test")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, C, H = 2, 1, 64
    x_ldct = torch.rand(B, C, H, H, device=device) * 0.5
    x_hdct = torch.rand(B, C, H, H, device=device) * 0.5

    model = Stage2Model(stage1_checkpoint=None, denoiser_kwargs={'image_size': H},
                        diffusion_kwargs={'timesteps': 50, 'loss_type': 'l2'},
                        image_size=H).to(device)

    print(f"  denoiser: {sum(p.numel() for p in model.denoiser.parameters())/1e6:.1f}M")
    print(f"  stage1 frozen: {all(not p.requires_grad for p in model.stage1.parameters())}")

    out = model(x_ldct, x_hdct, mode='train')
    assert out['loss_res'].item() >= 0 and out['loss_noise'].item() >= 0
    assert not torch.isnan(out['loss_res'])
    assert not torch.isnan(out['loss_noise'])
    print(f"  Training: L_res={out['loss_res']:.4f}, L_noise={out['loss_noise']:.4f} ✓")

    model.eval()
    out_inf = model(x_ldct[:1], mode='inference')
    assert out_inf['x_denoised'].shape == (1, C, H, H)
    assert torch.isfinite(out_inf['x_denoised']).all()
    print(f"  Inference: {tuple(out_inf['x_denoised'].shape)} ✓")

    # Gradient isolation
    out_train = model(x_ldct, x_hdct, mode='train')
    (out_train['loss_res'] + out_train['loss_noise']).backward()
    s1_grads = [p.grad for p in model.stage1.parameters() if p.grad is not None]
    den_grads = [p.grad for p in model.denoiser.parameters() if p.grad is not None]
    assert len(s1_grads) == 0
    assert len(den_grads) > 0
    print(f"  Gradients: Stage1=0, Denoiser={len(den_grads)} ✓")

    print(f"\n{'='*60}")
    print("All tests PASSED")
    print("=" * 60)
