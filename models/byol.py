"""
losses/byol.py

BYOL (Bootstrap Your Own Latent) — noise-invariant anatomy features.
=====================================================================
Makes Stage 1 VM-UNet features invariant to CT noise level so that
the same anatomy produces the same representation regardless of dose.

Why BYOL?
---------
Stage 1 is trained on NDCT (clean) images.
Stage 2 runs Stage 1 on LDCT (noisy) images for conditioning.
Without noise invariance, liver at 25% dose ≠ liver at 100% dose → bad conditioning.

BYOL forces: F(noisy_view) ≈ F(clean_view)  for the same anatomy.

Architecture
------------
Online network:  encoder (shared VM-UNet) + online_projector + predictor
Target network:  EMA encoder + target_projector  (NO predictor)

The BYOLModule owns only the projector and predictor heads.
The caller maintains EMA for the full encoder backbone separately.

Loss formula (one direction):
    z_online = online_projector(F_online)           [B, 256]
    q_online = predictor(z_online)                  [B, 256]
    z_target = target_projector(F_target).detach()  [B, 256]
    L = 2 - 2 * (normalize(q_online) · normalize(z_target)).mean()

Both directions are run by the caller and summed.

Reference: Architecture.pdf — Chapter 9 (BYOL)

Also re-exported from ``models/byol.py`` for the canonical import path
``from models.byol import BYOLModule``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
# ProjectorMLP
# ─────────────────────────────────────────────────────────────────────────────

class ProjectorMLP(nn.Module):
    """
    2-layer MLP: bottleneck features → compact projection vector.

    Input:  F [B, in_dim, 16, 16]  (spatial feature map)
    Step 1: Global average pool → [B, in_dim]
    Step 2: Linear(in_dim → hidden_dim) → BatchNorm1d → ReLU
    Step 3: Linear(hidden_dim → out_dim)
    Output: z [B, out_dim]

    Parameters
    ----------
    in_dim : int
        Input channel count (from bottleneck). Default: 768.
    hidden_dim : int
        Expansion width. Default: 4096.
    out_dim : int
        Output projection dimension. Default: 256.

    Notes
    -----
    * BatchNorm1d is applied after the first linear layer, consistent with
      the original BYOL paper and Architecture.pdf Chapter 9.
    * Global average pooling collapses (H, W) → scalar per channel,
      giving a single vector per image regardless of spatial size.
    * The last linear layer has NO BatchNorm or activation — the
      raw projection is normalised in the loss computation.
    """

    def __init__(
        self,
        in_dim:     int = 768,
        hidden_dim: int = 4096,
        out_dim:    int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # ── Layer 1 ──────────────────────────────────────────────────
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # ── Layer 2 ──────────────────────────────────────────────────
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feat : torch.Tensor  [B, in_dim, H, W]  or  [B, in_dim]
            If 4-D, global average pooling is applied first.

        Returns
        -------
        z : torch.Tensor  [B, out_dim]
        """
        if feat.dim() == 4:
            # Global average pooling: [B, C, H, W] → [B, C]
            x = feat.mean(dim=[2, 3])
        else:
            x = feat  # already [B, C]

        return self.net(x)   # [B, out_dim]

    def extra_repr(self) -> str:
        in_dim  = self.net[0].in_features
        hid_dim = self.net[0].out_features
        out_dim = self.net[-1].out_features
        return f"in_dim={in_dim}, hidden_dim={hid_dim}, out_dim={out_dim}"


# ─────────────────────────────────────────────────────────────────────────────
# PredictorMLP
# ─────────────────────────────────────────────────────────────────────────────

class PredictorMLP(nn.Module):
    """
    2-layer MLP used ONLY in the online network.
    Predicts the target network's projection from the online projection.

    Input:  z [B, in_dim]
    Step 1: Linear(in_dim → hidden_dim) → BatchNorm1d → ReLU
    Step 2: Linear(hidden_dim → out_dim)
    Output: q [B, out_dim]

    The last linear layer is zero-initialised at construction so that the
    predictor starts as a near-zero function → stable training start.
    Zero-init is applied externally by BYOLModule.__init__ to keep this
    class reusable.

    Parameters
    ----------
    in_dim : int
        Input dimension (= projector out_dim). Default: 256.
    hidden_dim : int
        Expansion width. Default: 4096.
    out_dim : int
        Output dimension. Default: 256.
    """

    def __init__(
        self,
        in_dim:     int = 256,
        hidden_dim: int = 4096,
        out_dim:    int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # ── Layer 1 ──────────────────────────────────────────────────
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # ── Layer 2 ──────────────────────────────────────────────────
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor  [B, in_dim]

        Returns
        -------
        q : torch.Tensor  [B, out_dim]
        """
        return self.net(z)

    def extra_repr(self) -> str:
        in_dim  = self.net[0].in_features
        hid_dim = self.net[0].out_features
        out_dim = self.net[-1].out_features
        return f"in_dim={in_dim}, hidden_dim={hid_dim}, out_dim={out_dim}"


# ─────────────────────────────────────────────────────────────────────────────
# BYOLModule
# ─────────────────────────────────────────────────────────────────────────────

class BYOLModule(nn.Module):
    """
    BYOL self-supervised module for noise-invariant anatomy features.

    Owns:
      * online_projector  — updated by gradients normally
      * target_projector  — slow EMA copy of online_projector (no grad)
      * predictor         — online-only; zero-initialised last layer

    The VM-UNet encoder backbone is NOT stored here.  The caller passes
    already-computed F_online and F_target feature maps.

    Parameters
    ----------
    feature_dim : int
        Bottleneck channel count (default 768, matching VM-UNet).
    projector_hidden : int
        Hidden width of both projector MLPs. Default: 4096.
    projector_out : int
        Output dimension of both projectors and the predictor. Default: 256.

    Forward
    -------
    F_online : [B, feature_dim, H, W]   gradient-connected features
    F_target : [B, feature_dim, H, W]   detached (from EMA encoder)

    Returns
    -------
    loss : scalar in [0, 4]
        0 = perfect alignment, 4 = opposite directions.

    Loss formula (one direction, symmetric is handled by calling forward
    twice with swapped arguments):
        z_online = online_projector(F_online)
        q_online = predictor(z_online)
        z_target = target_projector(F_target).detach()
        loss = 2 - 2 * mean( normalize(q_online) · normalize(z_target) )

    Collapse prevention
    -------------------
    The asymmetric predictor + EMA target + stop-gradient combination
    prevents mode collapse.  See Architecture.pdf Chapter 9 for details.

    Zero-initialisation
    -------------------
    The last Linear layer of the predictor is zero-initialised:
        weight = 0,  bias = 0
    This makes the predictor output zero at init → identity residual at
    the very first training step → stable start.

    Examples
    --------
    >>> byol = BYOLModule(feature_dim=768)
    >>> F_online = torch.randn(2, 768, 16, 16, requires_grad=True)
    >>> F_target = torch.randn(2, 768, 16, 16)
    >>> loss = byol(F_online, F_target)
    >>> loss.shape
    torch.Size([])
    >>> 0 <= loss.item() <= 4.0
    True
    """

    def __init__(
        self,
        feature_dim:      int = 768,
        projector_hidden: int = 4096,
        projector_out:    int = 256,
    ):
        super().__init__()

        # ── Online projector (receives gradients) ─────────────────────────
        self.online_projector = ProjectorMLP(
            in_dim     = feature_dim,
            hidden_dim = projector_hidden,
            out_dim    = projector_out,
        )

        # ── Target projector (EMA copy, no gradients) ─────────────────────
        self.target_projector = deepcopy(self.online_projector)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

        # ── Predictor (online network only) ───────────────────────────────
        self.predictor = PredictorMLP(
            in_dim     = projector_out,
            hidden_dim = projector_hidden,
            out_dim    = projector_out,
        )

        # ── Zero-initialise the last predictor layer ──────────────────────
        # self.predictor.net[-1] is the last nn.Linear
        nn.init.zeros_(self.predictor.net[-1].weight)
        nn.init.zeros_(self.predictor.net[-1].bias)

    # ── Loss computation ──────────────────────────────────────────────────

    @staticmethod
    def _byol_loss(
        q: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BYOL loss for one direction.

        L = 2 - 2 * mean( normalize(q) · normalize(z) )

        Parameters
        ----------
        q : [B, D]  online predictor output
        z : [B, D]  target projection (already detached)

        Returns
        -------
        scalar in [0, 4]
        """
        q_norm = F.normalize(q, dim=-1)   # unit sphere
        z_norm = F.normalize(z, dim=-1)   # unit sphere
        return 2.0 - 2.0 * (q_norm * z_norm).sum(dim=-1).mean()

    def forward(
        self,
        F_online: torch.Tensor,
        F_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BYOL loss for one direction: online(F_online) → target(F_target).

        Call twice with swapped arguments and sum for the symmetric loss:
            loss = byol(F_a, F_b) + byol(F_b, F_a)

        Parameters
        ----------
        F_online : [B, feature_dim, H, W]
            Gradient-connected bottleneck features from the online encoder.
        F_target : [B, feature_dim, H, W]
            Detached bottleneck features from the EMA target encoder.
            Should be detached by the caller (or will be detached here).

        Returns
        -------
        loss : scalar tensor in [0, 4]
        """
        # ── Online path ───────────────────────────────────────────────────
        # Gradients flow: F_online → online_projector → predictor → loss
        z_online = self.online_projector(F_online)   # [B, projector_out]
        q_online = self.predictor(z_online)          # [B, projector_out]

        # ── Target path (stop-gradient) ───────────────────────────────────
        # No gradients through target branch.
        # We detach here as a safety measure even if caller already detached.
        with torch.no_grad():
            z_target = self.target_projector(F_target.detach())  # [B, projector_out]

        # ── BYOL loss (one direction) ─────────────────────────────────────
        loss = self._byol_loss(q_online, z_target)

        return loss

    # ── EMA update ────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_target_projector(self, tau: float) -> None:
        """
        EMA update: target_proj ← τ * target_proj + (1 - τ) * online_proj.

        Call after every optimiser step during Stage 1 training.

        Parameters
        ----------
        tau : float
            EMA decay coefficient in [0, 1).
            tau = 0.996 → target changes slowly (99.6% old, 0.4% new).
            Larger tau = slower target update = more stable target.
        """
        for p_online, p_target in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            p_target.data = tau * p_target.data + (1.0 - tau) * p_online.data

    def extra_repr(self) -> str:
        feature_dim = self.online_projector.net[0].in_features
        proj_out    = self.online_projector.net[-1].out_features
        return f"feature_dim={feature_dim}, projector_out={proj_out}"


# ─────────────────────────────────────────────────────────────────────────────
# get_ema_tau
# ─────────────────────────────────────────────────────────────────────────────

def get_ema_tau(
    current_step: int,
    total_steps:  int,
    tau_start:    float = 0.996,
    tau_end:      float = 1.0,
) -> float:
    """
    Linearly interpolate EMA decay coefficient from tau_start to tau_end.

    At the beginning of training tau is small (target updates faster, giving
    a more responsive target) and increases toward 1.0 as training proceeds
    (target becomes more stable).

    Formula
    -------
    tau = tau_start + (tau_end - tau_start) * (current_step / total_steps)

    Parameters
    ----------
    current_step : int
        Current training step (0-indexed).
    total_steps : int
        Total number of training steps.
    tau_start : float
        Initial EMA decay. Default: 0.996.
    tau_end : float
        Final EMA decay. Default: 1.0.

    Returns
    -------
    tau : float  in [tau_start, tau_end]

    Examples
    --------
    >>> get_ema_tau(0, 100000)
    0.996
    >>> get_ema_tau(100000, 100000)
    1.0
    >>> get_ema_tau(50000, 100000)
    0.998
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}.")
    # Clamp progress to [0, 1] to handle step == total_steps edge case
    progress = min(current_step / total_steps, 1.0)
    return tau_start + (tau_end - tau_start) * progress


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("losses/byol.py — self-test")
    print("=" * 60)

    # ── ProjectorMLP ──────────────────────────────────────────────────────
    print("\n── ProjectorMLP ──────────────────────────────────────────────")

    proj = ProjectorMLP(in_dim=768, hidden_dim=4096, out_dim=256)

    # 4-D input (bottleneck feature map)
    F_map = torch.randn(2, 768, 16, 16)
    z = proj(F_map)
    assert z.shape == (2, 256), f"Expected (2, 256), got {z.shape}"
    print(f"  4-D input [2, 768, 16, 16] → z {tuple(z.shape)} ✓")

    # 2-D input (already pooled)
    F_vec = torch.randn(2, 768)
    z2 = proj(F_vec)
    assert z2.shape == (2, 256), f"Expected (2, 256), got {z2.shape}"
    print(f"  2-D input [2, 768]          → z {tuple(z2.shape)} ✓")

    # Gradient flows
    F_grd = torch.randn(2, 768, 16, 16, requires_grad=True)
    proj(F_grd).mean().backward()
    assert F_grd.grad is not None and F_grd.grad.norm() > 0
    print(f"  Gradient flow ✓")

    # ── PredictorMLP ──────────────────────────────────────────────────────
    print("\n── PredictorMLP ──────────────────────────────────────────────")

    pred = PredictorMLP(in_dim=256, hidden_dim=4096, out_dim=256)

    # Zero-init the last layer (as BYOLModule does)
    nn.init.zeros_(pred.net[-1].weight)
    nn.init.zeros_(pred.net[-1].bias)

    z_in = torch.randn(2, 256)
    q    = pred(z_in)
    assert q.shape == (2, 256), f"Expected (2, 256), got {q.shape}"
    print(f"  [2, 256] → q {tuple(q.shape)} ✓")

    # Zero-init means output is zero at init
    assert q.abs().max().item() == 0.0, "Zero-init: output should be exactly 0"
    print(f"  Zero-initialised output = 0.0 ✓")

    # Gradient flows through predictor
    z_grd = torch.randn(2, 256, requires_grad=True)
    # Reset init to non-zero so gradient is non-trivial
    nn.init.kaiming_uniform_(pred.net[-1].weight)
    pred(z_grd).mean().backward()
    assert z_grd.grad is not None and z_grd.grad.norm() > 0
    print(f"  Gradient flow ✓")

    # ── BYOLModule — basic forward + backward ─────────────────────────────
    print("\n── BYOLModule — forward + backward ──────────────────────────")

    byol = BYOLModule(feature_dim=768)

    # Predictor is zero-initialised → q_online ≈ 0 at first forward.
    # F.normalize(0) has no useful gradient to the backbone here, so backward
    # from q would not reach F_online. Slightly perturb the predictor for this
    # test only (real training reaches non-zero q after the first optimizer step).
    with torch.no_grad():
        byol.predictor.net[-1].weight.add_(torch.randn_like(byol.predictor.net[-1].weight) * 1e-3)

    F_online = torch.randn(2, 768, 16, 16, requires_grad=True)
    F_target = torch.randn(2, 768, 16, 16)

    loss = byol(F_online, F_target)

    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert 0 <= loss.item() <= 4.0, f"BYOL loss out of range: {loss.item()}"
    print(f"  Output shape : {tuple(loss.shape)} (scalar) ✓")
    print(f"  Loss value   : {loss.item():.4f}  (in [0, 4]) ✓")

    loss.backward()
    assert F_online.grad is not None, "Gradient did not reach F_online."
    gnorm = F_online.grad.norm().item()
    assert gnorm > 0, f"Gradient norm is zero."
    print(f"  Gradient norm (F_online) : {gnorm:.6f} ✓")
    print(f"BYOLModule forward+backward: PASSED (loss={loss.item():.4f})")

    # Target projector should have no grad
    for name, p in byol.target_projector.named_parameters():
        assert not p.requires_grad, f"Target param {name} should not require grad"
    print(f"  Target projector: requires_grad=False ✓")

    # Zero-init of predictor last layer
    # (already applied in __init__)
    # We verify by checking that the weight was re-initialised to zero at construction
    byol2 = BYOLModule(feature_dim=768)
    last_w = byol2.predictor.net[-1].weight
    last_b = byol2.predictor.net[-1].bias
    assert last_w.abs().max().item() == 0.0, "Predictor last weight should be zero."
    assert last_b.abs().max().item() == 0.0, "Predictor last bias should be zero."
    print(f"  Predictor last layer zero-init ✓")

    # ── Symmetric BYOL loss ───────────────────────────────────────────────
    print("\n── Symmetric loss (both directions) ─────────────────────────")

    F_a = torch.randn(2, 768, 16, 16, requires_grad=True)
    F_b = torch.randn(2, 768, 16, 16, requires_grad=True)

    byol3 = BYOLModule(feature_dim=768)
    with torch.no_grad():
        byol3.predictor.net[-1].weight.add_(torch.randn_like(byol3.predictor.net[-1].weight) * 1e-3)

    loss_ab = byol3(F_a, F_b)
    loss_ba = byol3(F_b, F_a)
    sym_loss = loss_ab + loss_ba

    assert sym_loss.shape == ()
    assert 0 <= sym_loss.item() <= 8.0
    print(f"  Symmetric loss (sum of both directions): {sym_loss.item():.4f} ✓")

    sym_loss.backward()
    assert F_a.grad is not None
    assert F_b.grad is not None
    print(f"  Gradients flow to both F_a and F_b ✓")

    # ── EMA update ────────────────────────────────────────────────────────
    print("\n── EMA update ────────────────────────────────────────────────")

    tau = get_ema_tau(500, 100000)
    assert abs(tau - (0.996 + (1.0 - 0.996) * 500 / 100000)) < 1e-9

    # Snapshot target params before update
    byol_ema = BYOLModule(feature_dim=768)
    target_before = {
        n: p.data.clone()
        for n, p in byol_ema.target_projector.named_parameters()
    }

    # Perturb online params slightly
    with torch.no_grad():
        for p in byol_ema.online_projector.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    byol_ema.update_target_projector(tau)

    # Check EMA update was applied
    for n, p_target in byol_ema.target_projector.named_parameters():
        p_online = dict(byol_ema.online_projector.named_parameters())[n]
        expected = tau * target_before[n] + (1.0 - tau) * p_online.data
        assert torch.allclose(p_target.data, expected, atol=1e-6), \
            f"EMA mismatch for param {n}"
    print(f"  EMA update applied correctly (tau={tau:.6f}) ✓")
    print(f"EMA update: PASSED (tau={tau:.6f})")

    # ── get_ema_tau ───────────────────────────────────────────────────────
    print("\n── get_ema_tau ───────────────────────────────────────────────")

    assert get_ema_tau(0, 100000) == 0.996,       "tau at step 0 should be tau_start"
    assert get_ema_tau(100000, 100000) == 1.0,    "tau at end should be tau_end"
    assert abs(get_ema_tau(50000, 100000) - 0.998) < 1e-9, "tau at midpoint"

    # Clamping: step > total_steps → tau = tau_end
    assert get_ema_tau(200000, 100000) == 1.0
    print(f"  tau at step    0 : {get_ema_tau(0, 100000):.6f}  (=tau_start) ✓")
    print(f"  tau at step 50000: {get_ema_tau(50000, 100000):.6f}  (midpoint) ✓")
    print(f"  tau at step 100000: {get_ema_tau(100000, 100000):.6f} (=tau_end) ✓")
    print(f"  tau clamped past total_steps : {get_ema_tau(200000, 100000):.6f} ✓")

    # ── Loss range checks ─────────────────────────────────────────────────
    print("\n── Loss range / edge cases ───────────────────────────────────")

    byol_edge = BYOLModule(feature_dim=768)
    # BatchNorm1d in projector/predictor rejects B=1 in training mode.
    byol_edge.eval()

    # Identical F_online and F_target should give low loss (near 0 after training)
    # At init with zero predictor, q ≈ 0 → loss ≈ 2
    F_same = torch.randn(2, 768, 16, 16)
    loss_same = byol_edge(F_same, F_same.clone())
    assert 0 <= loss_same.item() <= 4.0
    print(f"  Identical views loss : {loss_same.item():.4f}  (in [0, 4]) ✓")

    # Batch size = 1
    F_b1 = torch.randn(1, 768, 16, 16, requires_grad=True)
    loss_b1 = byol_edge(F_b1, torch.randn(1, 768, 16, 16))
    assert loss_b1.shape == ()
    loss_b1.backward()
    assert F_b1.grad is not None
    print(f"  B=1 works : loss={loss_b1.item():.4f} ✓")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)