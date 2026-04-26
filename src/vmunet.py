"""Stage-1 VMUNet adapter backed by `third_party/VM-UNet`.

Mamba is loaded directly from third_party/mamba — no pip install needed.
Gradient checkpointing is enabled to reduce VRAM usage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── resolve repo root regardless of cwd ──────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
VMUNET_ROOT = REPO_ROOT / "third_party" / "VM-UNet"
MAMBA_ROOT  = REPO_ROOT / "third_party" / "mamba"

for _p in (str(MAMBA_ROOT), str(VMUNET_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── patch vmamba.py to use local selective_scan before importing ──────────────
def _patch_vmamba() -> None:
    """
    Monkey-patch the selective_scan backend in vmamba.py so it always uses
    the fast path from third_party/mamba instead of the slow Python fallback.
    """
    vmamba_path = VMUNET_ROOT / "models" / "vmunet" / "vmamba.py"
    if not vmamba_path.exists():
        return
    src = vmamba_path.read_text()
    # Already patched in a previous run
    if "_LOCAL_MAMBA_PATCHED" in src:
        return
    new_block = (
        "\n# --- local-mamba patch injected by vmunet.py ---\n"
        "_LOCAL_MAMBA_PATCHED = True\n"
        "import sys as _sys\n"
        f"_sys.path.insert(0, r'{MAMBA_ROOT}')\n"
        "try:\n"
        "    from mamba_ssm.ops.selective_scan_interface import (\n"
        "        selective_scan_fn, selective_scan_ref)\n"
        "    print('[VM-UNet] selective scan backend: mamba_ssm (fast)')\n"
        "except Exception as _e:\n"
        "    print(f'[VM-UNet] mamba_ssm unavailable ({_e}), using pytorch_fallback')\n"
        "# --- end patch ---\n"
    )
    # Insert after the existing (failed) try/except import block
    marker = "except:\n    pass\n"
    if marker in src:
        src = src.replace(marker, marker + new_block, 1)
        vmamba_path.write_text(src)

_patch_vmamba()

try:
    from models.vmunet.vmunet import VMUNet as ThirdPartyVMUNet  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Failed to import third_party/VM-UNet. "
        "Ensure third_party/VM-UNet exists and einops/timm are installed."
    ) from exc


class VMUNet(nn.Module):
    """Thin wrapper that exposes logits, anatomy embeddings, and bottleneck features.

    Changes vs original:
    - use_checkpoint=True  → gradient checkpointing cuts VRAM ~40%
    - compute_embeddings flag skips pooling during BYOL-only steps
    - All forward ops grouped to minimise Python overhead
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = ThirdPartyVMUNet(
            input_channels=1,
            num_classes=num_classes,
            use_checkpoint=True,   # gradient checkpointing — big VRAM saving
        )

    # ------------------------------------------------------------------
    def masked_average_pool(
        self, feat: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """Vectorised soft-mask pooling — replaces slow Python loop."""
        # feat:  [B, C, H, W]
        # probs: [B, K, H, W]
        # out:   [B, K, C]
        weight = probs.unsqueeze(2)                    # [B, K, 1, H, W]
        feat_  = feat.unsqueeze(1)                     # [B, 1, C, H, W]
        num    = (weight * feat_).sum(dim=(-2, -1))    # [B, K, C]
        denom  = weight.sum(dim=(-2, -1)).clamp(min=1e-6)
        return num / denom

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        compute_embeddings: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1|3, H, W] input image
            compute_embeddings: if False skip e_a (saves memory during BYOL steps)

        Returns:
            logits      [B, C, H, W]
            e_a         [B, C, embed_dim]  or empty tensor
            F_bottle    [B, 768, H/32, W/32]
        """
        if x.size(1) == 1:
            x_in = x
        elif x.size(1) == 3:
            x_in = x.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.size(1)}.")

        F_bottleneck, skip_list = self.model.vmunet.forward_features(x_in)
        dec    = self.model.vmunet.forward_features_up(F_bottleneck, skip_list)
        logits = self.model.vmunet.forward_final(dec)

        if compute_embeddings:
            probs    = torch.softmax(logits, dim=1)
            dec_chw  = dec.permute(0, 3, 1, 2).contiguous()
            feat_up  = F.interpolate(
                dec_chw, scale_factor=4.0, mode="bilinear", align_corners=False
            )
            e_a = self.masked_average_pool(feat_up, probs)
        else:
            e_a = torch.empty(0, device=logits.device, dtype=logits.dtype)

        return logits, e_a, F_bottleneck.permute(0, 3, 1, 2).contiguous()