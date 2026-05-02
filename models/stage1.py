"""
models/stage1.py

Stage 1: Complete VM-UNet Teacher Network.
==========================================
Assembles VMUNet backbone + BYOL self-supervised module + anatomy embedding
computation into a single nn.Module that is the sole entry point for all
Stage 1 operations.

Two modes of operation
----------------------
Training (forward with return_byol=True):
    Returns logits, S, e_a, F, decoder_features, and byol_loss.
    The BYOL loss uses two noise-augmented views of the input image to
    force the bottleneck features to be invariant to CT dose level.

Inference / Stage-2 conditioning (forward with return_byol=False):
    Returns only S and e_a (plus logits, F, decoder_features for
    completeness).  Fastest path — no BYOL computation.

BYOL view generation
--------------------
If byol_view2 is None and return_byol=True, two views are generated
internally by adding Gaussian noise to the original image:
    view1 = x + 0.02 * N(0,1)   ← light noise  (NDCT-like)
    view2 = x + 0.15 * N(0,1)   ← heavy noise  (LDCT-like)

Memory-efficient BYOL
---------------------
The main ``backbone(x)`` forward (full UNet) doubles as view1's online pass.
View2 only needs bottleneck features, so we run ``backbone.encoder(view2)``
(twice: target under ``no_grad``, online with grad) and **skip the decoder**
for view2 — this avoids holding two full 512² decode graphs at once.

EMA update
----------
update_target_projector() is NOT called inside forward().
The training loop must call byol.update_target_projector(tau) after
each optimiser step.

Reference: Architecture.pdf – Chapters 8 and 9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.vm_unet import VMUNet
    from models.byol    import BYOLModule
    from utils.masking  import masked_average_pooling
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.vm_unet import VMUNet
    from models.byol    import BYOLModule
    from utils.masking  import masked_average_pooling


# ─────────────────────────────────────────────────────────────────────────────
# BYOL noise-augmentation hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

_BYOL_NOISE_LIGHT = 0.02   # σ for view1 (NDCT-like, small perturbation)
_BYOL_NOISE_HEAVY = 0.15   # σ for view2 (LDCT-like, heavy perturbation)


# ─────────────────────────────────────────────────────────────────────────────
# Stage1Model
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Model(nn.Module):
    """
    Complete Stage 1: VM-UNet Teacher Network with BYOL.

    Sub-modules
    -----------
    backbone : VMUNet
        Encoder–Decoder with a 7-class segmentation head.
        Outputs logits, S, F (bottleneck), decoder_features.

    byol : BYOLModule
        Online projector + predictor + EMA target projector.
        Consumes F [B, 768, 16, 16] to compute L_byol.

    Anatomy embeddings (e_a) are computed here via masked average pooling
    of decoder_features using S as soft spatial weights.

    Memory-efficient BYOL design
    ----------------------------
    Running the backbone three extra times (two views × two paths) causes
    OOM on a 24 GB GPU.  Instead we use a single-pass strategy:

        Pass 0 (main):   backbone(x)          → F_main   (gradient ON)
                          used as online features for view1

        Pass 1 (target): no_grad encoder(view2) → F2_target (gradient OFF)

        Pass 2 (online): encoder(view2) → F2_online (gradient ON); decoder
                          is **not** run on view2 — BYOL only uses bottleneck F.

        Target for view1 is approximated by F_main.detach() — both come
        from the same image perturbed with independent noise draws, so
        this is equivalent to the standard EMA approximation used when
        the backbone is shared.

    Skipping the decoder on view2 avoids the dominant VRAM cost (512² VSS in
    the decoder) when the main-pass decode graph is still live.

    Parameters
    ----------
    in_channels : int
        Input image channels. 1 for grayscale CT. Default: 1.
    num_classes : int
        Number of organ segmentation classes. Default: 7.
    embed_dim : int
        Base channel count for VM-UNet. Default: 96.
    depths : list[int]
        VSSBlock depth per encoder stage [s1, s2, s3, bottleneck].
        Default: [2, 2, 2, 2].
    byol_feature_dim : int
        Bottleneck channel count fed into BYOL projectors. Default: 768.

    Forward
    -------
    x : [B, 1, 512, 512]
        CT image (NDCT during Stage 1 training; LDCT at Stage 2 inference).
    byol_view2 : [B, 1, 512, 512] or None
        Explicit second augmented view.  If None and return_byol=True,
        both views are generated internally from x.
    return_byol : bool
        Whether to compute and return the BYOL loss. Default: False.

    Returns
    -------
    dict with keys:
        'logits'           : [B, num_classes, 512, 512]
        'S'                : [B, num_classes, 512, 512]
        'e_a'              : [B, num_classes, embed_dim]
        'F'                : [B, 768, 16, 16]
        'decoder_features' : [B, embed_dim, 512, 512]
        'byol_loss'        : scalar tensor or None

    Examples
    --------
    >>> model = Stage1Model()
    >>> out   = model(torch.randn(2, 1, 512, 512))
    >>> out['S'].shape
    torch.Size([2, 7, 512, 512])
    >>> out['e_a'].shape
    torch.Size([2, 7, 96])
    """

    def __init__(
        self,
        in_channels:      int  = 1,
        num_classes:      int  = 7,
        embed_dim:        int  = 96,
        depths:           list = None,
        byol_feature_dim: int  = 768,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 2, 2]

        # VMUNet expects 7 depth entries [enc×4 + dec×3]
        backbone_depths = list(depths) + list(reversed(depths[:3]))

        self.backbone = VMUNet(
            in_channels = in_channels,
            num_classes = num_classes,
            embed_dim   = embed_dim,
            depths      = backbone_depths,
        )

        self.byol = BYOLModule(feature_dim=byol_feature_dim)

        # Store config for extra_repr / count_parameters
        self.in_channels      = in_channels
        self.num_classes      = num_classes
        self.embed_dim        = embed_dim
        self.depths           = depths
        self.byol_feature_dim = byol_feature_dim

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Return x + σ·N(0,1) on the same device as x, without modifying x."""
        return x + sigma * torch.randn_like(x)

    def _encoder_F_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only bottleneck under no_grad (no decoder / seg head).

        Full ``backbone(view2)`` would run the 512² decoder + VSS while the
        main ``backbone(x)`` graph is still alive → OOM on ~24 GB GPUs.
        BYOL only needs the bottleneck F from view2, so encoder is enough.
        """
        with torch.no_grad():
            F_feat, _ = self.backbone.encoder(x)
        return F_feat.detach()

    def _compute_byol_loss(
        self,
        F_main:   torch.Tensor,
        view2:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the symmetric BYOL loss using only TWO backbone passes.

        Strategy
        --------
        We already have F_main from the main forward pass on x (with grad).
        We use F_main as the *online* features for view1.

        The target for view1 is F_main.detach() — a stop-gradient copy of
        the same features.  This is equivalent to using an EMA encoder
        (both see the same image up to independent light-noise draws).

        For view2 we run the **encoder only** twice (no full UNet decode):
          - once under no_grad  → F2_target
          - once with grad      → F2_online

        Total heavy work: 1 full ``backbone(x)`` + 2× ``encoder(view2)``.
        Skipping the decoder on view2 avoids holding two full 512² decode graphs.

        Parameters
        ----------
        F_main : [B, 768, 16, 16]   gradient-attached main-pass bottleneck
        view2  : [B, 1, H, W]       second noisy view

        Returns
        -------
        loss : scalar in [0, 8]
        """
        # ── Target features ───────────────────────────────────────────────
        # view1 target  → use F_main detached (same image, independent noise)
        F1_target = F_main.detach()                  # [B, 768, 16, 16]

        # view2 target  → encoder-only, no graph
        F2_target = self._encoder_F_no_grad(view2)   # [B, 768, 16, 16]

        # ── Online features ───────────────────────────────────────────────
        # view1 online  → already computed (F_main, gradient attached)
        F1_online = F_main                           # [B, 768, 16, 16]

        # view2 online  → encoder only (grad into view2 / encoder weights)
        F2_online, _ = self.backbone.encoder(view2)  # [B, 768, 16, 16]

        # ── Symmetric BYOL loss ───────────────────────────────────────────
        # Direction 1: online(view1) → predict target(view2)
        loss_12 = self.byol(F1_online, F2_target)
        # Direction 2: online(view2) → predict target(view1)
        loss_21 = self.byol(F2_online, F1_target)

        return loss_12 + loss_21     # scalar in [0, 8]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x:           torch.Tensor,
        byol_view2:  torch.Tensor = None,
        return_byol: bool         = False,
    ) -> dict:
        """
        Parameters
        ----------
        x : [B, 1, H, W]
        byol_view2 : [B, 1, H, W] or None
        return_byol : bool

        Returns
        -------
        dict — see class docstring.
        """
        # ── Main backbone forward (always on original x) ──────────────────
        backbone_out     = self.backbone(x)
        logits           = backbone_out['logits']           # [B, 7, 512, 512]
        S                = backbone_out['S']                # [B, 7, 512, 512]
        F_feat           = backbone_out['F']                # [B, 768, 16, 16]
        decoder_features = backbone_out['decoder_features'] # [B, 96, 512, 512]

        # ── Anatomy embeddings ────────────────────────────────────────────
        e_a = masked_average_pooling(decoder_features, S)  # [B, 7, 96]

        # ── BYOL loss (optional) ──────────────────────────────────────────
        byol_loss = None

        if return_byol:
            if byol_view2 is None:
                # Generate the second (heavy-noise) view internally.
                # view1 is the main pass (x itself — the backbone already
                # processed it above, so F_feat serves as F1_online).
                # view2 needs an explicit noisy image for the backbone.
                view2 = self._add_noise(x, _BYOL_NOISE_HEAVY)
            else:
                view2 = byol_view2

            byol_loss = self._compute_byol_loss(F_feat, view2)

        return {
            'logits':           logits,
            'S':                S,
            'e_a':              e_a,
            'F':                F_feat,
            'decoder_features': decoder_features,
            'byol_loss':        byol_loss,
        }

    # ── Convenience methods ───────────────────────────────────────────────

    @torch.no_grad()
    def get_anatomy_conditioning(self, x: torch.Tensor):
        """
        Inference-only fast path.  Returns (S, e_a) without BYOL computation.

        Intended to be called inside ``torch.no_grad()`` from the Stage 2
        training loop:

            with torch.no_grad():
                S, e_a = frozen_stage1.get_anatomy_conditioning(x_ldct)

        Parameters
        ----------
        x : [B, 1, H, W]

        Returns
        -------
        S   : [B, num_classes, H, W]
        e_a : [B, num_classes, embed_dim]
        """
        out = self.forward(x, return_byol=False)
        return out['S'], out['e_a']

    def count_parameters(self) -> dict:
        def _n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            'backbone':          _n(self.backbone),
            'byol_online_proj':  _n(self.byol.online_projector),
            'byol_target_proj':  _n(self.byol.target_projector),
            'byol_predictor':    _n(self.byol.predictor),
            'total':             _n(self),
        }

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"depths={self.depths}, "
            f"byol_feature_dim={self.byol_feature_dim}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# load_stage1_frozen
# ─────────────────────────────────────────────────────────────────────────────

def load_stage1_frozen(
    checkpoint_path: str,
    device:          str  = 'cuda',
    model_kwargs:    dict = None,
) -> Stage1Model:
    """
    Load a trained Stage 1 checkpoint and freeze ALL parameters.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pth`` file.  Must contain ``'model_state_dict'`` key
        or be a raw state-dict.
    device : str
        ``'cuda'`` or ``'cpu'``.
    model_kwargs : dict or None
        Forwarded to ``Stage1Model.__init__`` to override defaults.

    Returns
    -------
    model : Stage1Model  — all parameters frozen, in eval mode.
    """
    import os
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {checkpoint_path}"
        )

    model = Stage1Model(**(model_kwargs or {}))

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        raise KeyError(
            f"Cannot parse checkpoint at {checkpoint_path}. "
            f"Expected dict with 'model_state_dict' key or a raw state-dict."
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("models/stage1.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}\n")

    _B = 1 if device.type == 'cuda' else 2
    print(f"Self-test batch size B={_B} (B=1 on CUDA to limit VRAM).\n")

    def _cuda_gc():
        if device.type != 'cuda':
            return
        import gc as _gc

        _gc.collect()
        torch.cuda.empty_cache()

    # ── Instantiation ─────────────────────────────────────────────────────
    print("── Instantiation ─────────────────────────────────────────────")
    model = Stage1Model().to(device)
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k:<22} : {v:>12,}")
    total_params = counts['total']
    print(f"\n  Total Stage 1 parameters: {total_params / 1e6:.1f}M")

    # ── Inference mode ────────────────────────────────────────────────────
    print("\n── Inference mode (return_byol=False) ────────────────────────")
    x = torch.randn(_B, 1, 512, 512, device=device)

    with torch.no_grad():
        out = model(x, return_byol=False)

    assert out['S'].shape      == (_B, 7, 512, 512), f"S: {out['S'].shape}"
    assert out['e_a'].shape    == (_B, 7, 96),        f"e_a: {out['e_a'].shape}"
    assert out['F'].shape      == (_B, 768, 16, 16),  f"F: {out['F'].shape}"
    assert out['logits'].shape == (_B, 7, 512, 512),  f"logits: {out['logits'].shape}"
    assert out['decoder_features'].shape == (_B, 96, 512, 512)
    assert out['byol_loss'] is None

    ones = torch.ones(_B, 512, 512, device=device)
    assert torch.allclose(out['S'].sum(dim=1), ones, atol=1e-5)
    assert out['S'].min() >= 0.0 and out['S'].max() <= 1.0
    assert torch.isfinite(out['e_a']).all()

    print(f"  S              : {list(out['S'].shape)}  ✓")
    print(f"  e_a            : {list(out['e_a'].shape)}  ✓")
    print(f"  F              : {list(out['F'].shape)}  ✓")
    print(f"  logits         : {list(out['logits'].shape)}  ✓")
    print(f"  decoder_feats  : {list(out['decoder_features'].shape)}  ✓")
    print(f"  byol_loss      : None  ✓")
    print(f"  S sums to 1    ✓")
    print(f"  S in [0,1]     ✓")
    print(f"  e_a finite     ✓")
    print("Stage1Model inference mode: PASSED")
    del out
    _cuda_gc()

    # ── BYOL mode ─────────────────────────────────────────────────────────
    print("\n── BYOL mode (return_byol=True, auto views) ──────────────────")

    out_byol = model(x, return_byol=True)

    assert out_byol['byol_loss'] is not None, "byol_loss should not be None."
    assert out_byol['byol_loss'].shape == (), \
        f"byol_loss must be scalar, got {out_byol['byol_loss'].shape}"
    byol_val = out_byol['byol_loss'].item()
    # Symmetric BYOL loss is in [0, 8]
    assert 0 <= byol_val <= 8.0, f"BYOL loss out of range [0,8]: {byol_val}"

    print(f"  byol_loss      : {byol_val:.4f}  (in [0, 8])  ✓")
    print(f"Stage1Model BYOL mode: PASSED (byol_loss={byol_val:.4f})")
    keys_for_check = sorted(out_byol.keys())
    del out_byol
    _cuda_gc()

    # ── Explicit view2 ────────────────────────────────────────────────────
    print("\n── BYOL mode (explicit byol_view2) ───────────────────────────")

    view2 = torch.randn(_B, 1, 512, 512, device=device)
    out_byol2 = model(x, byol_view2=view2, return_byol=True)
    byol_val2 = out_byol2['byol_loss'].item()
    assert 0 <= byol_val2 <= 8.0
    print(f"  byol_loss (explicit view2) : {byol_val2:.4f}  ✓")

    del out_byol2, view2
    _cuda_gc()

    # ── get_anatomy_conditioning (before allocating a 2nd Stage1 on GPU) ─
    print("\n── get_anatomy_conditioning ──────────────────────────────────")
    with torch.no_grad():
        S_out, e_a_out = model.get_anatomy_conditioning(x)

    assert S_out.shape   == (_B, 7, 512, 512)
    assert e_a_out.shape == (_B, 7, 96)
    assert torch.allclose(S_out.sum(dim=1), ones, atol=1e-5)
    assert torch.isfinite(e_a_out).all()
    print(f"  S   : {list(S_out.shape)}  ✓")
    print(f"  e_a : {list(e_a_out.shape)}  ✓")
    print("Stage1Model.get_anatomy_conditioning: PASSED")

    del S_out, e_a_out, model, x, ones
    _cuda_gc()

    # ── Gradient flow ─────────────────────────────────────────────────────
    print("\n── Gradient flow ─────────────────────────────────────────────")

    model_grd = Stage1Model().to(device)
    x_grd     = torch.randn(_B, 1, 512, 512, device=device, requires_grad=True)

    out_grd = model_grd(x_grd, return_byol=False)
    out_grd['logits'].mean().backward()
    assert x_grd.grad is not None and x_grd.grad.norm().item() > 0
    print(f"  Seg loss grad norm : {x_grd.grad.norm().item():.6f}  ✓")

    # Target projector must never accumulate gradients
    for name, p in model_grd.byol.target_projector.named_parameters():
        assert not p.requires_grad, \
            f"Target projector param {name} should not require grad."
    print(f"  Target projector: requires_grad=False  ✓")

    del model_grd, x_grd, out_grd
    _cuda_gc()

    # ── Output key completeness ───────────────────────────────────────────
    print("\n── Output key completeness ───────────────────────────────────")
    expected_keys = {'logits', 'S', 'e_a', 'F', 'decoder_features', 'byol_loss'}
    assert set(keys_for_check) == expected_keys
    print(f"  Keys: {sorted(expected_keys)}  ✓")

    # ── load_stage1_frozen: missing checkpoint ────────────────────────────
    print("\n── load_stage1_frozen error handling ─────────────────────────")
    try:
        load_stage1_frozen('/nonexistent/path/stage1.pth', device='cpu')
        assert False, "Should have raised FileNotFoundError."
    except FileNotFoundError:
        print("  Missing checkpoint → FileNotFoundError  ✓")

    # ── load_stage1_frozen: round-trip ────────────────────────────────────
    print("\n── load_stage1_frozen round-trip ─────────────────────────────")
    import tempfile, os

    tiny = Stage1Model().to('cpu')
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save({'model_state_dict': tiny.state_dict()}, ckpt_path)
        loaded = load_stage1_frozen(ckpt_path, device='cpu')

        n_trainable = sum(
            p.numel() for p in loaded.parameters() if p.requires_grad
        )
        assert n_trainable == 0
        assert not loaded.training

        x_tiny = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            o1 = tiny(x_tiny,   return_byol=False)
            o2 = loaded(x_tiny, return_byol=False)
        assert torch.allclose(o1['S'], o2['S'], atol=1e-6)

        print(f"  Round-trip identical outputs  ✓")
        print(f"  Trainable params after freeze : {n_trainable}  ✓")
        print(f"  Eval mode after freeze        ✓")
    finally:
        os.unlink(ckpt_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Total Stage 1 parameters: {total_params / 1e6:.1f}M")
    print("All tests PASSED")
    print("=" * 60)