# ============================================================
# models/vmamba_blocks.py — self-test
# ============================================================
# Running on: cuda

# Raw CT input : [2, 1, 512, 512]

# ── LayerNorm2d (channels_first) ──────────────────────────────
#   [2, 1, 512, 512] → [2, 1, 512, 512]  ✓

# ── LayerNorm2d (channels_last) ───────────────────────────────
#   [2, 512, 512, 1] → [2, 512, 512, 1]  ✓

# ── PatchEmbed ────────────────────────────────────────────────
#   [2, 1, 512, 512] → [2, 96, 128, 128]  ✓
#   Params: 1,728

# ── PatchMerging ──────────────────────────────────────────────
#   [2, 96, 128, 128] → [2, 192, 64, 64]  ✓
#   Params: 74,496

# PatchEmbed and PatchMerging: PASSED

# ── SS2D (causal, 4-directional — Stage 1 reference) ─────────
#   [2, 8, 8, 32] → [2, 8, 8, 32]  ✓
#   Params: 3,712

# SS2D: PASSED

# ── VSSD (bidirectional, 2-axis — Stage 2) ────────────────────
#   Input  : [2, 16, 16, 96]  (BHWC)
#   Output : [2, 16, 16, 96]  ✓
#   Params : 23,232
#   Max |output − input| : 4.4550  (non-trivial ✓)
#   Input grad norm      : 0.001943  ✓
#   Bidirectional symmetry ratio : 1.0206  (expected ≈ 1.0 ✓)

#   Shape contract for full feature maps:
#     [B, 128, 128, 96]  →  [B, 128, 128, 96]  is guaranteed
#     by the reshape-only operations wrapping the recurrence.

# VSSD: PASSED

# ============================================================
# All tests PASSED
# ============================================================
# ============================================================

# Patient ID      | HDCT       | LDCT      
# ----------------------------------------
# C002            | 280        | 280       
# C004            | 361        | 361       
# C012            | 351        | 351       
# C016            | 319        | 319       
# C021            | 378        | 378       
# C030            | 303        | 303       
# C050            | 394        | 394       
# C052            | 342        | 342       
# C067            | 365        | 365       
# C081            | 356        | 356       
# C095            | 322        | 322       
# C107            | 391        | 391       
# C121            | 392        | 392       
# C124            | 383        | 383       
# C128            | 345        | 345       
# C130            | 379        | 379       
# C135            | 355        | 355       
# C158            | 363        | 363       
# C160            | 315        | 315       
# C162            | 343        | 343       
# C179            | 334        | 334       
# C193            | 365        | 365       
# C203            | 302        | 302       
# C218            | 357        | 357       
# C219            | 344        | 344       
# C224            | 332        | 332       
# C227            | 303        | 303       
# C232            | 353        | 353       
# C234            | 380        | 380       
# C241            | 366        | 366       
# C246            | 363        | 363       
# C252            | 363        | 363       
# C261            | 329        | 329       
# C267            | 333        | 333       
# C268            | 370        | 370       
# C280            | 369        | 369       
# C295            | 303        | 303       
# C296            | 329        | 329       
# L004            | 99         | 99        
# L006            | 215        | 215       
# L019            | 169        | 169       
# L033            | 162        | 162       
# L056            | 93         | 93        
# L058            | 210        | 210       
# L064            | 209        | 209       
# L071            | 154        | 154       
# L072            | 142        | 142       
# L077            | 161        | 161       
# L081            | 132        | 132       
# L110            | 133        | 133       
# L123            | 151        | 151       
# L125            | 205        | 205       
# L131            | 91         | 91        
# L134            | 217        | 217       
# L145            | 160        | 160       
# L148            | 210        | 210       
# L160            | 115        | 115       
# L170            | 137        | 137       
# L178            | 202        | 202       
# L179            | 143        | 143       
# L186            | 167        | 167       
# L193            | 169        | 169       
# L209            | 98         | 98        
# L210            | 149        | 149       
# L212            | 204        | 204       
# L220            | 156        | 156       
# L221            | 99         | 99        
# L237            | 140        | 140       
# L241            | 155        | 155       
# L266            | 175        | 175       
# masks           | 0          | 0         
# ----------------------------------------
# TOTAL           | 18254      | 18254
# ========================================================================================================================

#============================================================
# models/vm_unet.py — VMUNetEncoder self-test
# ============================================================
# Device : cuda

# Total parameters     : 9,573,312
# Trainable parameters : 9,573,312

# ── Forward pass ──────────────────────────────────────────────
#   Input           : [2, 1, 512, 512]
#   bottleneck (F)  : [2, 768, 16, 16]
#   skip1           : [2, 96, 128, 128]
#   skip2           : [2, 192, 64, 64]
#   skip3           : [2, 384, 32, 32]

# ── Shape assertions ──────────────────────────────────────────
#   bottleneck (2,768,16,16)   ✓
#   skip1      (2,96,128,128)  ✓
#   skip2      (2,192,64,64)   ✓
#   skip3      (2,384,32,32)   ✓

# ── Format check (4D tensors, expected BCHW) ─────────────────
#   bottleneck    [B=2, C=768, H=16, W=16]  ✓
#   skip1         [B=2, C=96, H=128, W=128]  ✓
#   skip2         [B=2, C=192, H=64, W=64]  ✓
#   skip3         [B=2, C=384, H=32, W=32]  ✓

# ── Gradient flow ─────────────────────────────────────────────
#   Input grad norm : 0.004656  ✓

# ── Channel progression ───────────────────────────────────────
#   PatchEmbed   [B,1,512,512] → [B, 96,128,128]
#   Scale1 VSS   [B,96,128,128]               (skip1)
#   PatchMerge1  [B,96,128,128] → [B,192,64,64]
#   Scale2 VSS   [B,192,64,64]                (skip2)
#   PatchMerge2  [B,192,64,64]  → [B,384,32,32]
#   Scale3 VSS   [B,384,32,32]                (skip3)
#   PatchMerge3  [B,384,32,32]  → [B,768,16,16]
#   Bottleneck   [B,768,16,16]                (F / BYOL input)

# ── Drop-path schedule (8 blocks, linear 0→0.1) ───────────────
#   block  0  drop_path = 0.0000
#   block  1  drop_path = 0.0143
#   block  2  drop_path = 0.0286
#   block  3  drop_path = 0.0429
#   block  4  drop_path = 0.0571
#   block  5  drop_path = 0.0714
#   block  6  drop_path = 0.0857
#   block  7  drop_path = 0.1000

# ============================================================
# VMUNetEncoder: PASSED
# ============================================================# ============================================================

# ============================================================
# models/vm_unet.py — self-test
# ============================================================
# Device : cuda

# ── VMUNetEncoder ─────────────────────────────────────────────
#   bottleneck : [2, 768, 16, 16]  ✓
#   skip1      : [2, 96, 128, 128]  ✓
#   skip2      : [2, 192, 64, 64]  ✓
#   skip3      : [2, 384, 32, 32]  ✓
# VMUNetEncoder: PASSED

# ── VMUNetDecoder ─────────────────────────────────────────────
#   features         : [2, 96, 512, 512]  ✓
#   decoder_features : [2, 96, 512, 512]  ✓  (same tensor)
#   Params : 8,687,616
# VMUNetDecoder: PASSED

# ── VMUNet (full model) ───────────────────────────────────────
#   Encoder    params :    9,573,312
#   Decoder    params :    8,687,616
#   Seg head   params :          679
#   Total      params :   18,261,607

#   logits           : [2, 7, 512, 512]  ✓
#   S                : [2, 7, 512, 512]  ✓
#   F                : [2, 768, 16, 16]  ✓
#   decoder_features : [2, 96, 512, 512]  ✓
#   S sums to 1 (atol=1e-5)  ✓
#   S in [0, 1]              ✓
#   Gradient norm            : 0.725947  ✓
#   freeze() → 0 trainable   ✓  (eval mode)
#   Output keys              ✓  ['F', 'S', 'decoder_features', 'logits']

# ============================================================
# VMUNet: PASSED
# ============================================================# ============================================================
# ============================================================
# utils/masking.py — self-test
# ============================================================

# ── Test 1: shape + correctness (all pixels → class 1) ────────
#   Output shape : [2, 7, 96]  ✓
#   e_a[:, 1, :] == global mean  ✓
#   e_a[:, k≠1, :] ≈ 0          ✓

# masked_average_pooling: PASSED

# ── Test 2: soft weighting (50/50 split) ──────────────────────
#   50/50 hard split → both class embeddings = 1.0  ✓

# ── Test 3: empty class (absent organ) ────────────────────────
#   Absent classes produce zero embeddings  ✓

# ── Test 4: gradient flow ─────────────────────────────────────
#   decoder_features grad norm : 0.000146  ✓

# ── Test 5: compute_anatomy_embeddings wrapper ────────────────
#   Output shape : [2, 7, 96]  ✓
#   Wrapper == direct call      ✓
#   Missing key raises KeyError ✓  ("vm_unet_output is missing required keys: {'decoder_features'}. Available keys: {'S'}")

# ── Test 6: spatial mismatch raises ValueError ─────────────────
#   Spatial mismatch raises ValueError  ✓

# ── Test 7: real-like softmax S ───────────────────────────────
#   Shape   : [2, 7, 96]  ✓
#   Finite  : True  ✓
#   e_a std : 0.0026  (reasonable range ✓)

# ============================================================
# All tests PASSED
# ========================================================================================================================