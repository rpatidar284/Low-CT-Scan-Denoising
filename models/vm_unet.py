"""
models/vm_unet.py

VM-UNet: Visual State Space UNet for CT organ segmentation.
===========================================================
Stage 1 of the anatomy-aware CT denoising pipeline.

Contents
--------
  VMUNetEncoder   — 4-scale encoder with skip connections and bottleneck.
  VMUNetDecoder   — 3-scale decoder that reconstructs full-resolution features.
  VMUNet          — Complete encoder + decoder + segmentation head.

Data-format convention
----------------------
  PatchEmbed, PatchMerging, Conv2d  : BCHW  (channel-first)
  VSSBlock, VSSD                    : BHWC  (channel-last)

  Permutes are inserted at every BCHW↔BHWC boundary.  All skip connections,
  the bottleneck, and all public outputs are in BCHW so downstream code
  (bilinear upsample, Conv2d, masked average pooling) works without extra
  permutes.

Reference: Architecture.pdf – Chapter 8 (VM-UNet / Stage 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Works when running as a package/module:
    #   python -m models.vm_unet
    from models.vmamba_blocks import PatchEmbed, PatchMerging, VSSBlock
except ModuleNotFoundError:
    # Fallback for direct script execution:
    #   python models/vm_unet.py
    from vmamba_blocks import PatchEmbed, PatchMerging, VSSBlock


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: build a Sequential of VSSBlocks with per-block drop-path rates
# ─────────────────────────────────────────────────────────────────────────────

def _make_vss_layer_with_rates(
    d_model:  int,
    depth:    int,
    d_state:  int,
    dp_rates: list,
) -> nn.Sequential:
    assert len(dp_rates) == depth
    return nn.Sequential(
        *[VSSBlock(d_model=d_model, d_state=d_state, drop_path=dp_rates[i])
          for i in range(depth)]
    )


# ─────────────────────────────────────────────────────────────────────────────
# VMUNetEncoder
# ─────────────────────────────────────────────────────────────────────────────

class VMUNetEncoder(nn.Module):
    """
    VM-UNet Encoder — 4-scale hierarchical feature extractor.

    Input:  [B, 1, 512, 512]   BCHW
    Output:
      bottleneck : [B, 768, 16, 16]   BCHW  ("F" for BYOL)
      skips      : [skip1, skip2, skip3]  all BCHW
                    skip1 [B,  96, 128, 128]
                    skip2 [B, 192,  64,  64]
                    skip3 [B, 384,  32,  32]
    """

    def __init__(
        self,
        in_channels:    int   = 1,
        embed_dim:      int   = 96,
        depths:         list  = None,
        patch_size:     int   = 4,
        d_state:        int   = 16,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        assert len(depths) == 4

        self.in_channels    = in_channels
        self.embed_dim      = embed_dim
        self.depths         = depths
        self.patch_size     = patch_size
        self.d_state        = d_state
        self.drop_path_rate = drop_path_rate

        C = embed_dim
        self.dims = [C, C * 2, C * 4, C * 8]

        # Linear drop-path schedule across all blocks
        total    = sum(depths)
        dp_rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        # Build encoder stages
        idx = 0
        self.layer1 = _make_vss_layer_with_rates(
            self.dims[0], depths[0], d_state, dp_rates[idx: idx + depths[0]])
        idx += depths[0]
        self.merge1  = PatchMerging(self.dims[0])

        self.layer2 = _make_vss_layer_with_rates(
            self.dims[1], depths[1], d_state, dp_rates[idx: idx + depths[1]])
        idx += depths[1]
        self.merge2  = PatchMerging(self.dims[1])

        self.layer3 = _make_vss_layer_with_rates(
            self.dims[2], depths[2], d_state, dp_rates[idx: idx + depths[2]])
        idx += depths[2]
        self.merge3  = PatchMerging(self.dims[2])

        self.bottleneck_layer = _make_vss_layer_with_rates(
            self.dims[3], depths[3], d_state, dp_rates[idx: idx + depths[3]])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x     = self.patch_embed(x)                  # [B, 96, 128, 128]  BCHW

        x     = x.permute(0, 2, 3, 1)               # → BHWC
        x     = self.layer1(x)
        x     = x.permute(0, 3, 1, 2)               # → BCHW
        skip1 = x                                    # [B, 96, 128, 128]
        x     = self.merge1(x)                       # [B, 192, 64, 64]

        x     = x.permute(0, 2, 3, 1)
        x     = self.layer2(x)
        x     = x.permute(0, 3, 1, 2)
        skip2 = x                                    # [B, 192, 64, 64]
        x     = self.merge2(x)                       # [B, 384, 32, 32]

        x     = x.permute(0, 2, 3, 1)
        x     = self.layer3(x)
        x     = x.permute(0, 3, 1, 2)
        skip3 = x                                    # [B, 384, 32, 32]
        x     = self.merge3(x)                       # [B, 768, 16, 16]

        x     = x.permute(0, 2, 3, 1)
        x     = self.bottleneck_layer(x)
        x     = x.permute(0, 3, 1, 2)               # [B, 768, 16, 16]  BCHW

        return x, [skip1, skip2, skip3]


# ─────────────────────────────────────────────────────────────────────────────
# DecoderBlock — one upsampling + skip-connection + VSS stage
# ─────────────────────────────────────────────────────────────────────────────

class _DecoderBlock(nn.Module):
    """
    One decoder stage:
      1. Bilinear 2× upsample
      2. 1×1 Conv: halve channels  (c_in → c_in//2)
      3. Concatenate skip along channel dim  → doubled channels
      4. VSSBlocks
      5. 1×1 Conv: reduce back to c_in//2

    Parameters
    ----------
    c_in   : channels coming from below (e.g. 768)
    depth  : number of VSSBlocks
    d_state: Mamba state size
    dp_rates: per-block drop-path rates (len == depth)

    Shapes (example for c_in=768, skip has 384 channels)
    ------
    Input from below : [B, 768, 16, 16]
    After upsample   : [B, 768, 32, 32]
    After proj_in    : [B, 384, 32, 32]   (Conv1×1  768→384)
    After cat(skip)  : [B, 768, 32, 32]   (384+384)
    After VSSBlocks  : [B, 768, 32, 32]
    After proj_out   : [B, 384, 32, 32]   (Conv1×1  768→384)
    """

    def __init__(
        self,
        c_in:     int,
        depth:    int,
        d_state:  int,
        dp_rates: list,
    ):
        super().__init__()

        c_out = c_in // 2          # e.g. 768 → 384

        # 1×1 conv to halve channels after upsample (before concatenation)
        self.proj_in = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)

        # VSS blocks operate on concatenated tensor: c_out (up) + c_out (skip)
        c_cat = c_out * 2          # e.g. 384 + 384 = 768
        self.vss_blocks = _make_vss_layer_with_rates(c_cat, depth, d_state, dp_rates)

        # 1×1 conv to reduce back to c_out after VSSBlocks
        self.proj_out = nn.Conv2d(c_cat, c_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : BCHW — features from the deeper stage
        skip : BCHW — skip connection from the matching encoder scale

        Returns
        -------
        BCHW tensor with c_in//2 channels
        """
        # Step 1 + 2: upsample then reduce channels
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.proj_in(x)                      # [B, c_out, 2H, 2W]

        # Step 3: concatenate with skip
        x = torch.cat([x, skip], dim=1)          # [B, c_cat, 2H, 2W]

        # Step 4: VSSBlocks (BHWC ↔ BCHW boundary)
        x = x.permute(0, 2, 3, 1)               # BCHW → BHWC
        x = self.vss_blocks(x)                   # [B, 2H, 2W, c_cat]
        x = x.permute(0, 3, 1, 2)               # BHWC → BCHW

        # Step 5: reduce channels
        x = self.proj_out(x)                     # [B, c_out, 2H, 2W]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# VMUNetDecoder
# ─────────────────────────────────────────────────────────────────────────────

class VMUNetDecoder(nn.Module):
    """
    VM-UNet Decoder with skip connections.

    Progressively recovers spatial resolution from the bottleneck using
    bilinear upsampling, skip-connection concatenation, and VSS blocks.

    Input
    -----
    bottleneck : [B, 768, 16, 16]   BCHW
    skips      : [skip1, skip2, skip3]   all BCHW
                  skip1 [B,  96, 128, 128]
                  skip2 [B, 192,  64,  64]
                  skip3 [B, 384,  32,  32]

    Output
    ------
    features         : [B, 96, 512, 512]   BCHW  ← full-resolution feature map
    decoder_features : [B, 96, 512, 512]   BCHW  ← same tensor, alias
                       (used for masked average pooling → e_a)

    Processing pipeline
    -------------------
    Scale 3 up:
      bottleneck [B,768,16,16]
        → upsample 2× → [B,768,32,32]
        → proj_in Conv1×1(768→384) → [B,384,32,32]
        → cat(skip3) → [B,768,32,32]
        → 2×VSSBlock(768) → [B,768,32,32]
        → proj_out Conv1×1(768→384) → [B,384,32,32]

    Scale 2 up:
      [B,384,32,32]
        → upsample 2× → [B,384,64,64]
        → proj_in Conv1×1(384→192) → [B,192,64,64]
        → cat(skip2) → [B,384,64,64]
        → 2×VSSBlock(384) → [B,384,64,64]
        → proj_out Conv1×1(384→192) → [B,192,64,64]

    Scale 1 up:
      [B,192,64,64]
        → upsample 2× → [B,192,128,128]
        → proj_in Conv1×1(192→96) → [B,96,128,128]
        → cat(skip1) → [B,192,128,128]
        → 2×VSSBlock(192) → [B,192,128,128]
        → proj_out Conv1×1(192→96) → [B,96,128,128]

    Final upsample:
      [B,96,128,128]
        → bilinear 4× → [B,96,512,512]   ← decoder_features / features

    Parameters
    ----------
    embed_dim : int        Channel count at Scale 1 (default 96).
    depths    : list[int]  VSSBlocks per decoder stage [s3, s2, s1] (default [2,2,2]).
    d_state   : int        Mamba state dimension (default 16).
    drop_path_rate : float Maximum stochastic-depth rate (default 0.1).
    """

    def __init__(
        self,
        embed_dim:      int   = 96,
        depths:         list  = None,
        d_state:        int   = 16,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2]
        assert len(depths) == 3, "Decoder expects 3 depth values [s3, s2, s1]."

        self.embed_dim      = embed_dim
        self.depths         = depths
        self.d_state        = d_state
        self.drop_path_rate = drop_path_rate

        C = embed_dim
        # Channels entering each decoder block from below
        # Scale-3 block receives bottleneck (8C); outputs 4C
        # Scale-2 block receives 4C; outputs 2C
        # Scale-1 block receives 2C; outputs C
        c_ins = [C * 8, C * 4, C * 2]   # [768, 384, 192]

        # Linear drop-path schedule across all decoder blocks
        total    = sum(depths)
        dp_rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]

        idx = 0
        self.up3 = _DecoderBlock(
            c_in=c_ins[0], depth=depths[0], d_state=d_state,
            dp_rates=dp_rates[idx: idx + depths[0]],
        )
        idx += depths[0]

        self.up2 = _DecoderBlock(
            c_in=c_ins[1], depth=depths[1], d_state=d_state,
            dp_rates=dp_rates[idx: idx + depths[1]],
        )
        idx += depths[1]

        self.up1 = _DecoderBlock(
            c_in=c_ins[2], depth=depths[2], d_state=d_state,
            dp_rates=dp_rates[idx: idx + depths[2]],
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, bottleneck: torch.Tensor, skips: list):
        """
        Parameters
        ----------
        bottleneck : [B, 768, 16, 16]   BCHW
        skips      : [skip1, skip2, skip3]
                     skip1 [B,  96, 128, 128]
                     skip2 [B, 192,  64,  64]
                     skip3 [B, 384,  32,  32]

        Returns
        -------
        features         : [B, 96, 512, 512]   BCHW
        decoder_features : [B, 96, 512, 512]   BCHW  (same tensor)
        """
        skip1, skip2, skip3 = skips

        x = self.up3(bottleneck, skip3)   # [B, 384, 32,  32]
        x = self.up2(x,          skip2)   # [B, 192, 64,  64]
        x = self.up1(x,          skip1)   # [B,  96, 128, 128]

        # Final 4× upsample to restore full image resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        # [B, 96, 512, 512]

        # decoder_features is the pre-head representation used for
        # masked average pooling → e_a (anatomy embeddings)
        decoder_features = x

        return x, decoder_features


# ─────────────────────────────────────────────────────────────────────────────
# VMUNet — complete Stage 1 model
# ─────────────────────────────────────────────────────────────────────────────

class VMUNet(nn.Module):
    """
    Complete VM-UNet for Stage 1 CT organ segmentation.

    Combines VMUNetEncoder + VMUNetDecoder + segmentation head.

    After training, this model is **frozen** forever.  Stage 2 calls it
    with ``torch.no_grad()`` to obtain S and e_a as anatomy conditioning.

    Segmentation head
    -----------------
    Conv2d(96, num_classes, kernel_size=1)  → logits [B, 7, 512, 512]
    Softmax(dim=1)                           → S      [B, 7, 512, 512]

    Returns  (dict)
    ---------------
    'logits'          : [B, 7, 512, 512]   raw class scores    (for CE loss)
    'S'               : [B, 7, 512, 512]   organ probabilities (0-1, sums to 1)
    'F'               : [B, 768, 16, 16]   bottleneck features (for BYOL)
    'decoder_features': [B, 96, 512, 512]  pre-head features   (for e_a pooling)

    Parameters
    ----------
    in_channels : int   Input image channels (1 for grayscale CT).
    num_classes : int   Number of organ classes (7 per architecture spec).
    embed_dim   : int   Base channel count (96).
    depths      : list  [enc_s1, enc_s2, enc_s3, bottleneck,
                         dec_s3, dec_s2, dec_s1]  (7 entries).
                  Default [2,2,2,2, 2,2,2].
    patch_size  : int   PatchEmbed stride (4).
    d_state     : int   Mamba hidden-state size (16).
    drop_path_rate: float  Max stochastic-depth rate (0.1).

    Examples
    --------
    >>> model = VMUNet()
    >>> out   = model(torch.randn(2, 1, 512, 512))
    >>> out['S'].shape
    torch.Size([2, 7, 512, 512])
    >>> out['S'].sum(dim=1).allclose(torch.ones(2, 512, 512))
    True
    """

    def __init__(
        self,
        in_channels:    int   = 1,
        num_classes:    int   = 7,
        embed_dim:      int   = 96,
        depths:         list  = None,
        patch_size:     int   = 4,
        d_state:        int   = 16,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2, 2, 2, 2]  # 4 encoder + 3 decoder stages
        assert len(depths) == 7, (
            f"depths must have 7 entries "
            f"[enc_s1, enc_s2, enc_s3, bottleneck, dec_s3, dec_s2, dec_s1], "
            f"got {len(depths)}."
        )

        self.in_channels    = in_channels
        self.num_classes    = num_classes
        self.embed_dim      = embed_dim
        self.depths         = depths
        self.patch_size     = patch_size
        self.d_state        = d_state
        self.drop_path_rate = drop_path_rate

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = VMUNetEncoder(
            in_channels    = in_channels,
            embed_dim      = embed_dim,
            depths         = depths[:4],          # [s1, s2, s3, bottleneck]
            patch_size     = patch_size,
            d_state        = d_state,
            drop_path_rate = drop_path_rate,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = VMUNetDecoder(
            embed_dim      = embed_dim,
            depths         = depths[4:],          # [dec_s3, dec_s2, dec_s1]
            d_state        = d_state,
            drop_path_rate = drop_path_rate,
        )

        # ── Segmentation head ─────────────────────────────────────────────
        # Conv2d(96, 7, kernel=1): independent per-pixel class score.
        # kernel_size=1 is equivalent to a channel-wise Linear projection
        # but stays in BCHW without a permute.
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, 1, H, W]   BCHW raw CT image (HU values)

        Returns
        -------
        dict with keys:
          'logits'           [B, num_classes, H, W]
          'S'                [B, num_classes, H, W]
          'F'                [B, 768, H/32, W/32]
          'decoder_features' [B, embed_dim, H, W]
        """
        # ── Encode ────────────────────────────────────────────────────────
        # F (bottleneck) is the "F" in the architecture spec used for BYOL.
        F_feat, skips = self.encoder(x)          # F_feat [B,768,16,16]

        # ── Decode ────────────────────────────────────────────────────────
        features, decoder_features = self.decoder(F_feat, skips)
        # features == decoder_features == [B, 96, 512, 512]

        # ── Segmentation head ─────────────────────────────────────────────
        logits = self.seg_head(features)          # [B, 7, 512, 512]
        S      = torch.softmax(logits, dim=1)     # [B, 7, 512, 512]  sums to 1

        return {
            'logits':           logits,           # raw scores for CE loss
            'S':                S,                # soft organ probabilities
            'F':                F_feat,           # bottleneck for BYOL
            'decoder_features': decoder_features, # for masked avg pooling → e_a
        }

    # ------------------------------------------------------------------ #
    def freeze(self):
        """
        Freeze all parameters so Stage 2 cannot modify them.

        Call this after Stage 1 training is complete, before loading
        the checkpoint for Stage 2 training.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> dict:
        """Return parameter counts for each sub-module."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            'encoder':    _count(self.encoder),
            'decoder':    _count(self.decoder),
            'seg_head':   _count(self.seg_head),
            'total':      _count(self),
        }

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"depths={self.depths}, "
            f"patch_size={self.patch_size}, "
            f"d_state={self.d_state}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("models/vm_unet.py — self-test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}\n")

    # ── 1. Encoder (carried over from previous test) ──────────────────────
    print("── VMUNetEncoder ─────────────────────────────────────────────")
    enc = VMUNetEncoder().to(device)
    x   = torch.randn(2, 1, 512, 512, device=device)
    with torch.no_grad():
        bn, skips = enc(x)
    assert tuple(bn.shape)       == (2, 768, 16, 16)
    assert tuple(skips[0].shape) == (2,  96, 128, 128)
    assert tuple(skips[1].shape) == (2, 192,  64,  64)
    assert tuple(skips[2].shape) == (2, 384,  32,  32)
    print(f"  bottleneck : {list(bn.shape)}  ✓")
    for i, s in enumerate(skips, 1):
        print(f"  skip{i}      : {list(s.shape)}  ✓")
    print("VMUNetEncoder: PASSED\n")

    # ── 2. Decoder ────────────────────────────────────────────────────────
    print("── VMUNetDecoder ─────────────────────────────────────────────")
    dec = VMUNetDecoder().to(device)
    with torch.no_grad():
        features, dec_feat = dec(bn, skips)
    assert tuple(features.shape)  == (2, 96, 512, 512), \
        f"features: {features.shape}"
    assert tuple(dec_feat.shape)  == (2, 96, 512, 512), \
        f"decoder_features: {dec_feat.shape}"
    # Verify they are the same tensor (alias, not a copy)
    assert features is dec_feat, "features and decoder_features should be the same object."
    print(f"  features         : {list(features.shape)}  ✓")
    print(f"  decoder_features : {list(dec_feat.shape)}  ✓  (same tensor)")
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"  Params : {dec_params:,}")
    print("VMUNetDecoder: PASSED\n")

    # ── 3. Full VMUNet ────────────────────────────────────────────────────
    print("── VMUNet (full model) ───────────────────────────────────────")
    model = VMUNet().to(device)
    param_counts = model.count_parameters()
    print(f"  Encoder    params : {param_counts['encoder']:>12,}")
    print(f"  Decoder    params : {param_counts['decoder']:>12,}")
    print(f"  Seg head   params : {param_counts['seg_head']:>12,}")
    print(f"  Total      params : {param_counts['total']:>12,}")
    print()

    x_in = torch.randn(2, 1, 512, 512, device=device)
    with torch.no_grad():
        out = model(x_in)

    # Shape assertions (from the task specification)
    assert out['S'].shape      == (2, 7, 512, 512), \
        f"S wrong: {out['S'].shape}"
    assert out['logits'].shape == (2, 7, 512, 512), \
        f"logits wrong: {out['logits'].shape}"
    assert out['F'].shape      == (2, 768, 16, 16), \
        f"F wrong: {out['F'].shape}"
    assert out['decoder_features'].shape == (2, 96, 512, 512), \
        f"decoder_features wrong: {out['decoder_features'].shape}"

    print(f"  logits           : {list(out['logits'].shape)}  ✓")
    print(f"  S                : {list(out['S'].shape)}  ✓")
    print(f"  F                : {list(out['F'].shape)}  ✓")
    print(f"  decoder_features : {list(out['decoder_features'].shape)}  ✓")

    # Softmax sanity: S must sum to 1 over the class dimension at every pixel
    s_sum = out['S'].sum(dim=1)                     # [2, 512, 512]
    ones  = torch.ones(2, 512, 512, device=device)
    assert torch.allclose(s_sum, ones, atol=1e-5), \
        f"S does not sum to 1; max deviation = {(s_sum - ones).abs().max():.2e}"
    print(f"  S sums to 1 (atol=1e-5)  ✓")

    # S must be in [0, 1]
    assert out['S'].min() >= 0.0 and out['S'].max() <= 1.0, \
        "S contains values outside [0, 1]."
    print(f"  S in [0, 1]              ✓")

    # Gradient flow
    x_grd  = torch.randn(1, 1, 512, 512, device=device, requires_grad=True)
    model_g = VMUNet().to(device)
    loss    = model_g(x_grd)['logits'].mean()
    loss.backward()
    assert x_grd.grad is not None and x_grd.grad.norm().item() > 0
    print(f"  Gradient norm            : {x_grd.grad.norm().item():.6f}  ✓")

    # Freeze test
    model.freeze()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == 0, f"freeze() left {n_trainable} trainable parameters."
    assert not model.training, "freeze() should put model in eval mode."
    print(f"  freeze() → 0 trainable   ✓  (eval mode)")

    # Output dict keys
    expected_keys = {'logits', 'S', 'F', 'decoder_features'}
    assert set(out.keys()) == expected_keys, \
        f"Output keys mismatch: {set(out.keys())} vs {expected_keys}"
    print(f"  Output keys              ✓  {sorted(out.keys())}")

    print()
    print("=" * 60)
    print("VMUNet: PASSED")
    print("=" * 60)