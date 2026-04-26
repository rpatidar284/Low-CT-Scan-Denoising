"""Stage-2 anatomy conditioning stubs for future M4 implementation."""

import torch.nn as nn


class SpatialFiLM(nn.Module):
    """Placeholder FiLM generator conditioned on segmentation maps."""

    # TODO M4: takes S [B,7,H,W], outputs gamma [B,C,H,W] and beta [B,C,H,W]
    def __init__(self, num_classes: int = 7, hidden: int = 32, out_channels: int | None = None) -> None:
        super().__init__()
        _ = (num_classes, hidden, out_channels)
        raise NotImplementedError("Implement in M4")


class AnatomyMamba_block(nn.Module):
    """Placeholder anatomy-aware Mamba block for Stage-2."""

    # TODO M4: replaces Mamba_block in DADiff.py
    # Adds SpatialFiLM conditioning on S and cross-attention on e_a
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        _ = (args, kwargs)
        raise NotImplementedError("Implement in M4")

