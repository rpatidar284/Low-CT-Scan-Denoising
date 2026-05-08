# Complete Architecture: Anatomy-Aware Low-Dose CT Denoising
## STARTING FROM ABSOLUTE ZERO
---
# CHAPTER 0: WHAT IS A CT SCAN AND WHAT ARE WE TRYING TO DO?
## What is CT?
A CT (Computed Tomography) scanner rotates X-ray beams around a patient and
measures how much X-ray is absorbed by different tissues. Dense tissue (bone)
absorbs more X-rays. Soft tissue absorbs medium. Air absorbs very little.
The scanner reconstructs a 3D image where each tiny cube (called a **voxel**) has
an intensity value measured in **Hounsfield Units (HU)**:
- Air = -1000 HU
- Fat = -100 HU
- Water = 0 HU
- Soft tissue (liver, muscle) = +20 to +80 HU
- Bone = +400 to +1000 HU
When you look at a single cross-sectional slice through the body, you get a 2D
grayscale image where brighter = denser tissue.
## What is Low-Dose CT?
More X-ray dose = clearer image BUT higher cancer risk for the patient.
Hospitals reduce dose to 25% of normal. This saves the patient from radiation but
makes the image extremely noisy — random speckles and granular artifacts appear
everywhere, obscuring fine details.
```
Normal dose CT (NDCT): Low dose CT (LDCT):
████████████████ ▓▒░▓▒▓░▒▓▒░▓▒▓░
████ liver ████ → ▓▒ liver ▒▓░▒▓▒
████████████████ ░▒▓░▒▓▒░▓░▒▓░▒
(clean, sharp) (noisy, grainy)
```
## What Does a Denoiser Do?
Takes the noisy LDCT image as input, outputs a clean image that looks like NDCT.
This is an image-to-image translation task.
## Why Do Existing Methods Fail?
Standard denoisers use pixel-level loss functions:
**MSE (Mean Squared Error):**
```
MSE = (1/N) * sum[(predicted_pixel - true_pixel)²]
```
This treats every pixel identically. A pixel in empty air background gets the same
importance as a pixel on the liver-kidney boundary. To minimize this loss, the
network learns to "play it safe" at uncertain regions by outputting the average —
which creates blurred boundaries. This is called **over-smoothing**.
**The clinical consequence:** A blurred liver boundary might cause a radiologist to
miss a tumor growing at the edge. Good PSNR score, but clinically dangerous.
## Our Core Idea
Tell the denoiser WHERE each organ is and WHAT each organ looks like. Apply
organ-specific denoising strategies. Liver gets liver-appropriate denoising. Lung
gets lung-appropriate denoising. Boundaries get boundary-preserving treatment.
---
# CHAPTER 1: TENSORS AND DIMENSIONS — THE LANGUAGE OF DEEP LEARNING
Before anything else, you need to understand how data is represented.
## What is a Tensor?
A tensor is just a multi-dimensional array of numbers. Think of it as nested lists.
- **Scalar**: a single number. `5.3`
- **Vector (1D tensor)**: a list of numbers. `[1.2, 3.4, 5.6]`
- **Matrix (2D tensor)**: a grid of numbers. Shape [rows, columns]
- **3D tensor**: a cube of numbers. Shape [depth, rows, columns]
- **4D tensor**: Shape [A, B, C, D]
## What Does [B, C, H, W] Mean?
Your CT image is 512×512 pixels, grayscale (one intensity value per pixel).
In Python/PyTorch, a single image is stored as a 3D tensor:
```
[channels, height, width] = [1, 512, 512]
```
**Channels (C):** How many "layers" of information per pixel.
- Grayscale CT: 1 channel (just intensity)
- RGB photo: 3 channels (red, green, blue)
- After neural network processing: could be 96 channels (96 different features per
pixel)
**Height (H):** Number of pixel rows = 512
**Width (W):** Number of pixel columns = 512
**Batch (B):** In training, you don't process one image at a time — you process
multiple images simultaneously for efficiency. If B=8, you're processing 8 images
at once. All stacked into a 4D tensor.
```
Single image: [1, 512, 512] (1 channel, 512 height, 512 width)
Batch of 8: [8, 1, 512, 512] (8 images, each with 1 channel, 512×512)
```
## Physical Meaning of Each Dimension
```
tensor[b, c, h, w]
b = which image in the batch (0 to B-1)
c = which channel/feature (0 to C-1)
h = which row/pixel from top (0 to H-1)
w = which column/pixel from left (0 to W-1)
```
So `tensor[3, 0, 256, 128]` = the intensity value of pixel at row 256, column 128,
in the 4th image of the batch, channel 0.
## What Happens to Channels Through a Network?
The network transforms the representation:
```
Input: [B, 1, 512, 512] → 1 feature per pixel (raw HU intensity)
After layer 1: [B, 96, 128, 128] → 96 features per pixel (learned representations)
After layer 2: [B, 192, 64, 64] → 192 features per pixel at lower resolution
...
Output: [B, 7, 512, 512] → 7 features per pixel (probability per organ class)
```
The number 96, 192, 384, 768 are just design choices — how many features the
network uses internally.
## What is [B, H, W, C] vs [B, C, H, W]?
PyTorch convention: **channel-first** = [B, C, H, W]
Some operations use **channel-last** = [B, H, W, C]
The data is the same, just arranged differently in memory. You convert with:
```python
x_BCHW = x_BHWC.permute(0, 3, 1, 2) # move C from last to second position
x_BHWC = x_BCHW.permute(0, 2, 3, 1) # move C from second to last position
```
Mamba operations expect BHWC (channel-last). The rest of PyTorch uses BCHW
(channel-first). You'll see these `.permute()` calls frequently.
---
# CHAPTER 2: THE THREE-PART SYSTEM OVERVIEW
The entire system has three parts that run in sequence:
```
═══════════════════════════════════════════════════════════════
PART 0: TotalSegmentator (runs ONCE before training, offline)
Input: 3D CT volume (full-dose NDCT)
Output: 3D organ mask (which voxel belongs to which organ)
Saved: as .npy files on disk
═══════════════════════════════════════════════════════════════
↓ (mask files used during training)
═══════════════════════════════════════════════════════════════
STAGE 1: VM-UNet Teacher (trained first, then frozen forever)
Input: CT image [B, 1, 512, 512]
Output: S [B, 7, 512, 512] ← organ probability maps
e_a [B, 7, C] ← organ feature embeddings
Purpose: understand anatomy
═══════════════════════════════════════════════════════════════
↓ (S and e_a passed to Stage 2)
═══════════════════════════════════════════════════════════════
STAGE 2: VSSD Denoiser (the final model, trained second)
Input: LDCT image + S + e_a
Output: Denoised CT image [B, 1, 512, 512]
Purpose: denoise while preserving anatomy
═══════════════════════════════════════════════════════════════
```
**At inference time (actual deployment):**
- Stage 1 runs first on the LDCT → produces S and e_a
- Stage 2 takes LDCT + S + e_a → produces denoised image
- TotalSegmentator is NOT used at inference
**Why separate stages?** Training them together creates conflicting gradients. The
segmentation loss wants sharp boundaries. The denoising loss wants smooth pixel
predictions. These fight each other and both learn poorly. Separate training lets
each do its job optimally.
---
# CHAPTER 3: TOTALSEGMENTATOR — GENERATING ORGAN LABELS
## What is TotalSegmentator?
A pre-trained neural network (based on nnU-Net) published by Wasserthal et al.
2023. Given any CT volume, it automatically labels 104 anatomical structures — one
integer label per voxel.
You don't train TotalSegmentator. You just download it and run it. It's a tool,
like a pre-installed software.
## Why Do You Need It?
To train Stage 1 (the segmentation network), you need labels — which pixel belongs
to which organ. Getting a radiologist to manually label thousands of CT slices
costs millions of dollars and months of time.
TotalSegmentator does it automatically in ~20 seconds per volume with 85-95%
accuracy. These automatically generated labels are called **pseudo-labels**
("pseudo" = fake/approximate, but good enough for training).
## The 2D/3D Problem
Your dataset stores CT images as **2D slices** (individual .npy files). Think of it
like slices of bread — each slice is a 2D image.
TotalSegmentator needs the **full 3D volume** (the whole loaf of bread) because
organ identity requires 3D context. A 2D slice might show a circular blob — is it a
kidney cross-section? A lymph node? A cyst? Looking at multiple consecutive slices
reveals the true shape across slices, making identification reliable.
## The Pipeline
```
Step 1: Collect all 2D .npy slices for one patient
[slice_001.npy, slice_002.npy, ..., slice_200.npy]
Each slice: [512, 512] array of HU values
Step 2: Stack into 3D volume
volume = np.stack(all_slices) → shape [200, 512, 512]
Step 3: Convert to NIfTI format (.nii.gz)
NIfTI = medical imaging format that stores 3D volumes with metadata
(voxel spacing, orientation, etc.)
sitk.WriteImage(sitk.GetImageFromArray(volume), "patient.nii.gz")
Step 4: Run TotalSegmentator
totalsegmentator(input="patient.nii.gz", output="patient_mask.nii.gz")
→ output is 3D mask [200, 512, 512] with integers 0-103
Step 5: Remap 104 labels → 7 labels
(see table below)
Step 6: Split back to 2D slices
for i, slice in enumerate(mask_3d):
np.save(f"mask_{i:04d}.npy", slice)
```
## Why 7 Classes Instead of 104?
Different organs have different noise characteristics in CT. For denoising, you
need to distinguish tissues that behave differently:
| Class ID | What's Included | Why This Group |
|----------|----------------|----------------|
| 0 | Background, air | Air has completely different noise (quantum noise dominates
at -1000 HU) |
| 1 | Liver, Spleen | Similar HU range (40-60 HU), similar soft tissue noise |
| 2 | Kidney left, Kidney right | Distinct bright cortex + dark medulla pattern |
| 3 | Aorta, Inferior Vena Cava | Blood vessels, motion artifacts from heartbeat |
| 4 | Lung, Lung vessels | Air-filled, very dark (-700 HU), totally different noise
statistics |
| 5 | Vertebrae, Ribs | Bone (HU > 400), completely different noise pattern |
| 6 | Soft tissue, Muscle, rest | Everything else |
104 classes = too hard to segment accurately, too much memory, unnecessary for
denoising. 7 classes = captures all meaningfully different tissue types.
## Why Only Run TotalSegmentator on NDCT?
Your dataset has paired images: same patient, same body position, one NDCT scan and
one LDCT scan. They're perfectly aligned (registered).
The anatomy is identical in both — only the noise differs. So the NDCT mask is
valid for the LDCT image too. Running TotalSegmentator on noisy LDCT would give
less accurate results (noisy input → noisy segmentation).
```python
# In your dataset loader:
# For NDCT image: load mask directly
mask = np.load(f"masks/ndct_{patient}_{slice}.npy")
# For LDCT image: use the SAME mask (same anatomy, just different noise)
mask = np.load(f"masks/ndct_{patient}_{slice}.npy") # same file!
```
## Label Smoothing (ε = 0.1)
TotalSegmentator makes mistakes 5-15% of the time, especially at organ boundaries.
If you train your network to be 100% confident about these sometimes-wrong labels,
it learns overconfidence.
Instead of hard labels:
```
Hard label for "liver" pixel: [0, 1, 0, 0, 0, 0, 0]
```
Use smooth labels:
```
Smooth label: [0.014, 0.914, 0.014, 0.014, 0.014, 0.014, 0.014]
Formula: smooth[k] = (1 - ε) if k==true_class else ε/(num_classes-1)
= (1 - 0.1) = 0.9 for true class
= 0.1/6 ≈ 0.014 for other classes
```
This is called **label smoothing** — it tells the network "be mostly confident
about liver, but acknowledge there's slight uncertainty."
---
# CHAPTER 4: NEURAL NETWORK FUNDAMENTALS
Before explaining the architectures, let's establish what neural networks do.
## What is a Layer?
A layer takes a tensor in, applies a mathematical transformation, outputs a new
tensor. Neural networks are sequences of layers.
## What is a Convolution (Conv2d)?
The most basic image processing operation. A small filter (kernel) slides across
the image:
```
Input pixel neighborhood: 3×3 filter (learned weights):
[1, 2, 1] [a, b, c]
[0, 3, 2] * [d, e, f] = sum of element-wise products
[1, 1, 0] [g, h, i]
Output = 1*a + 2*b + 1*c + 0*d + 3*e + 2*f + 1*g + 1*h + 0*i
```
The filter slides across every position in the image. Different filters detect
different patterns (edges, textures, curves). The filters are learned by the
network during training.
**Conv2d(in_channels, out_channels, kernel_size):**
- in_channels: how many channels in the input
- out_channels: how many channels in the output (= how many different filters)
- kernel_size: size of each filter (3 = 3×3, 4 = 4×4)
## What is Stride?
How many pixels the filter jumps between positions.
- stride=1: filter moves 1 pixel at a time → output same size as input
- stride=2: filter jumps 2 pixels → output is half the size
- stride=4: filter jumps 4 pixels → output is quarter the size
## What is Downsampling?
Reducing spatial resolution (H and W become smaller). Like shrinking an image.
Why do this? At smaller resolutions, each "pixel" covers a larger area of the
original image, giving the network a wider view to understand global structure
(where the liver is relative to the spine, etc.)
## What is Upsampling?
Increasing spatial resolution back (H and W become larger). Like enlarging an
image.
**Bilinear upsampling:** Smooth interpolation — new pixels get values based on
weighted average of nearby original pixels. No artifacts.
## What is LayerNorm?
Normalization stabilizes training. LayerNorm normalizes the feature values so they
have mean=0 and standard deviation=1:
```
LayerNorm(x) = (x - mean(x)) / (std(x) + ε) * γ + β
```
Where γ and β are learned parameters (the network can learn to rescale after
normalization).
Why normalize? During training, feature values can grow very large or very small,
making gradients explode or vanish. Normalization keeps everything in a reasonable
range.
## What is a Residual Connection?
```
output = layer(input) + input
```
The input is added directly to the output, bypassing the layer. This means the
layer only needs to learn the **difference** (residual) from the input, not a
completely new representation.
Benefits:
1. Gradient flows directly through the addition → no vanishing gradient
2. If the layer isn't helpful, it can learn to output zeros → identity function
3. Makes deep networks much easier to train
## What is a Linear Layer (Fully Connected)?
Applies a matrix multiplication:
```
output = input @ weight_matrix + bias
```
Every input value connects to every output value. Used for changing dimensions or
mixing information across the channel dimension.
---
# CHAPTER 5: WHAT IS MAMBA (STATE SPACE MODELS)?
## The Problem with Convolutions for Images
A 3×3 convolution can only see 3×3 = 9 pixels at a time. To see across an entire
512×512 image, you'd need many stacked layers, which is slow and computationally
expensive.
## The Problem with Transformers (Self-Attention)
Self-attention lets every pixel look at every other pixel:
```
attention_cost = O(N²) where N = number of pixels
For 512×512: N = 262,144
N² = 68 billion operations
```
This is computationally impossible for full-resolution images.
## What is a State Space Model (SSM)?
An SSM processes a sequence element by element while maintaining a **hidden state**
— a compressed summary of everything seen so far.
Think of it like reading a book:
- You read word by word (sequential)
- You maintain a mental summary of the plot so far (hidden state)
- Your understanding of the current word depends on both the word itself AND your
accumulated mental model
Mathematical formulation:
```
h_t = A * h_(t-1) + B * x_t ← state update equation
y_t = C * h_t ← output equation
Where:
x_t = current input (the pixel you're currently looking at)
h_t = current hidden state (summary of everything seen so far)
h_(t-1) = previous hidden state
A = state matrix (how much to remember from past)
B = input matrix (how much current input affects state)
C = output matrix (how to read the state to produce output)
y_t = output for current position
```
This processes the whole sequence in O(N) time — each element processed once.
## What Makes Mamba Special?
In standard SSMs, A, B, C are **fixed matrices** — they never change regardless of
the input content.
In **Mamba (Selective SSM)**, A, B, C are **functions of the input** — they change
at each position based on what the input actually is:
```
B_t = Linear(x_t) ← how much current pixel affects the state depends on pixel
value
C_t = Linear(x_t) ← how to read the state depends on pixel value
```
This "selectivity" means:
- At a boring background pixel: Mamba learns to have small B values → input barely
updates the state → state passes through mostly unchanged → background doesn't
dominate the representation
- At an important organ boundary: Mamba learns to have large B values → input
strongly updates the state → boundary information propagates through subsequent
positions
## How VMamba Adapts Mamba for 2D Images
Mamba processes sequences (1D). Images are 2D. VMamba solves this by unrolling the
2D image into four different 1D sequences and scanning each:
```
Scan 1: Left→Right, Top→Bottom (normal reading order)
Row 0: pixel(0,0), pixel(0,1), ..., pixel(0,511)
Row 1: pixel(1,0), pixel(1,1), ..., pixel(1,511)
...
Scan 2: Right→Left, Bottom→Top (reversed)
Scan 3: Top→Bottom, Left→Right (column order)
Scan 4: Bottom→Top, Right→Left (reversed column)
```
Results from all 4 scans are aggregated (added together). This way, each pixel can
receive context from all other pixels through at least one scan direction.
This operator is called **SS2D (2D Selective Scan)**.
## VSSD vs SS2D — The Crucial Difference
**SS2D (VMamba1, used in original FoundDiff):**
Each scan is **causal** — when processing position i, only positions 0 to i are
visible.
```
Scanning left→right: pixel at column 300 can see columns 0-299 but NOT 301-511
```
This is fine for language (don't look at future words) but wrong for images (every
pixel should see all other pixels).
**VSSD (VMamba2/ICCV 2025):**
**Non-causal** — every position sees ALL other positions in all 4 directions
simultaneously.
```
Every pixel sees every other pixel → truly global context
+ Uses Mamba2's SSD kernel → 3-4× faster than Mamba1
```
For denoising a CT image, you want every pixel to use information from the entire
image. VSSD is strictly better than SS2D for this task.
---
# CHAPTER 6: THE VSS BLOCK — VMAMBA'S BUILDING BLOCK
Every scale in the VM-UNet encoder and decoder uses VSS blocks (Visual State Space
blocks). Here is every single operation inside one VSS block, step by step:
## Input
```
x: tensor of shape [B, H, W, C]
B = batch size (e.g., 8)
H = height at this scale (e.g., 128)
W = width at this scale (e.g., 128)
C = channels at this scale (e.g., 96)
```
Note: VSS blocks work in BHWC format (channel-last), not BCHW.
## Step 1: LayerNorm
```python
x_norm = LayerNorm(x)
# Shape stays: [B, H, W, C]
# Normalizes the C-dimensional feature vector at each spatial position
# → zero mean, unit variance per position
# → stabilizes training
```
## Step 2: Linear Projection to 2× Channels
```python
x_expanded = Linear(C → 2C)(x_norm)
# Shape: [B, H, W, 2C]
x_main, x_gate = split(x_expanded, 2, dim=-1)
# x_main: [B, H, W, C] ← will go through the Mamba scan
# x_gate: [B, H, W, C] ← will control/filter the Mamba output
```
**Why split into two branches?** This is a **gating mechanism**. The main branch
processes information. The gate branch learns to filter/suppress that information.
Together they give the network finer control over what information to keep. (More
on gating below.)
## Step 3: Depthwise Convolution 3×3 on x_main
```python
x_main = DepthwiseConv2d(C, C, kernel=3, padding=1)(x_main)
# Shape stays: [B, H, W, C]
```
**Depthwise convolution:** Each of the C channels gets its own independent 3×3
filter. Channel 0 is processed by filter 0, channel 1 by filter 1, etc. This is
much cheaper than regular convolution (which mixes all channels).
**Why do this before the Mamba scan?** The Mamba scan processes the image
sequentially and understands global context. But it lacks local spatial detail
(nearby pixels). The depthwise conv injects local neighborhood information before
the global scan. Together: local detail + global context.
## Step 4: SS2D (or VSSD) Scan
```python
x_main = SS2D(x_main) # for Stage 1 (VMamba1)
# OR
x_main = VSSD(x_main) # for Stage 2 (VMamba2, non-causal)
# Shape stays: [B, H, W, C]
```
This is where the magic happens. The selective state-space scan runs across the
image in 4 directions. Each pixel's representation is now informed by all other
pixels in the image.
After this step, `x_main[b, h, w, :]` is a C-dimensional vector that contains:
- Local information about pixel (h,w) itself
- Global context from all other pixels via the state propagation
## Step 5: Gating (The Crucial Step)
```python
x_gated = x_main * SiLU(x_gate)
# Shape: [B, H, W, C]
# SiLU(x) = x * sigmoid(x)
# sigmoid(x) = 1 / (1 + exp(-x)) → output always in [0, 1]
# SiLU(x) → output in (-0.28, +∞), smooth, non-monotonic
```
**What is gating physically?**
Imagine the Mamba scan produces 96 features at each pixel. Some of these features
are useful and informative. Some are noise or irrelevant information. The gate
learns to decide which ones to keep.
```
Feature 0 from Mamba: 3.2 (maybe this detects "organ boundary present")
Gate value for feature 0: 0.95 (high → keep this feature)
Result: 3.2 * 0.95 = 3.04 ← mostly kept
Feature 47 from Mamba: 2.1 (maybe this is spurious noise from the scan)
Gate value for feature 47: 0.02 (low → suppress this feature)
Result: 2.1 * 0.02 = 0.042 ← almost zeroed out
```
The gate values are learned during training. The network learns which features from
the Mamba scan are reliable and worth using.
**Why SiLU instead of sigmoid?** Sigmoid outputs 0-1 (pure multiplicative gate).
SiLU allows negative values and has a smoother gradient, which trains better
empirically.
## Step 6: Project Back to Original Channels
```python
x_out = Linear(C → C)(x_gated)
# Shape: [B, H, W, C]
# Mixes the gated features across channels
# Allows the network to create new representations by combining features
```
## Step 7: Residual Connection
```python
output = x + x_out
# x = original input (before LayerNorm)
# x_out = processed output
# Shape: [B, H, W, C]
```
The original input is added back. The network only needs to learn the **change**
(residual) from input to output. At initialization, the network produces near-zero
outputs, so output ≈ input. Training gradually learns what changes to make.
## Why Use 2 VSS Blocks Per Scale?
At each encoder/decoder scale, two VSS blocks are applied sequentially:
```
input → [VSS Block 1] → intermediate → [VSS Block 2] → output
```
First block: raw feature extraction — "what is at each position?"
Second block: contextual refinement — "how does each position relate to others
given the first block's features?"
One block could theoretically do both, but two blocks with separate parameters
learn better. More blocks = better features but more computation. 2 is the standard
balance.
---
# CHAPTER 7: UNET — THE SKELETON OF BOTH STAGES
## What is UNet?
UNet is a neural network architecture designed for image-to-image tasks. It has a
specific shape (like the letter U) with three parts:
### The Encoder Path (Going Down)
Progressively reduces spatial resolution while increasing channels:
```
[B, 1, 512, 512] → full resolution, raw image
↓ (downsample)
[B, 96, 128, 128] → 4× smaller, 96 features per pixel
↓ (downsample)
[B, 192, 64, 64] → 8× smaller, 192 features per pixel
↓ (downsample)
[B, 384, 32, 32] → 16× smaller, 384 features per pixel
↓ (downsample)
[B, 768, 16, 16] → 32× smaller, 768 features per pixel (BOTTLENECK)
```
At each scale, VSS blocks process the features. Downsampling happens between
scales.
**Why decrease resolution?** At 16×16 resolution, each "pixel" represents a 32×32
area of the original image. The network can now understand global organ layout
(where is the liver? where is the spine?). You can't see the whole liver at full
512×512 resolution because the liver spans hundreds of pixels — the receptive field
is too small.
**Why increase channels?** We lose spatial information when downsampling. We
compensate by having more feature channels — more ways to describe each remaining
position.
### The Bottleneck
The most compressed representation [B, 768, 16, 16]. Only 16×16 spatial positions
but each has a 768-dimensional feature vector. This is where the network has the
best global understanding of the entire image.
### The Decoder Path (Going Up)
Progressively restores spatial resolution:
```
[B, 768, 16, 16] (bottleneck)
↑ (upsample)
[B, 384, 32, 32]
↑ (upsample)
[B, 192, 64, 64]
↑ (upsample)
[B, 96, 128, 128]
↑ (upsample 4×)
[B, 96, 512, 512] → full resolution, 96 features
↓
[B, 7, 512, 512] → 7 organ probability values per pixel (for Stage 1)
OR
[B, 1, 512, 512] → predicted residual (for Stage 2)
```
### Skip Connections — The Key Innovation
At each scale, the encoder features are directly concatenated to the decoder
features:
```
Encoder scale 1: [B, 96, 128, 128] ──────────────→ concatenated to Decoder scale 1
Encoder scale 2: [B, 192, 64, 64] ───────────────→ concatenated to Decoder scale 2
Encoder scale 3: [B, 384, 32, 32] ───────────────→ concatenated to Decoder scale 3
Bottleneck [B, 768, 16, 16]
```
**Why skip connections?** The decoder must reconstruct the full-resolution output.
But the bottleneck only has 16×16 spatial positions — it has lost most fine-grained
spatial detail. Skip connections directly reintroduce the fine details from the
encoder. Without them, the decoder would have to reconstruct boundaries and
textures from scratch — very hard.
**What concatenation does:**
```
Upsampled from below: [B, 384, 32, 32]
Skip from encoder: [B, 384, 32, 32]
After concat: [B, 768, 32, 32] ← doubled channels
After VSS blocks: [B, 384, 32, 32] ← back to expected channels
```
The VSS blocks then merge the upsampled (global context) features with the skip
(fine local detail) features.
---
# CHAPTER 8: STAGE 1 — VM-UNET TEACHER NETWORK
## Purpose
Stage 1 is a dedicated anatomy understanding network. It learns to look at a CT
image and:
1. Identify which pixels belong to which organ (segmentation)
2. Create compact representations of what each organ looks like (embeddings)
After training, Stage 1 is **frozen** — its weights never change again. It becomes
a fixed tool that Stage 2 uses for anatomy information.
## Patch Embedding — The Input Layer
```
Input: [B, 1, 512, 512] (grayscale CT image)
Operation: Conv2d(in_channels=1, out_channels=96, kernel_size=4, stride=4)
Output: [B, 96, 128, 128]
```
**What this does:**
- Divides the 512×512 image into non-overlapping 4×4 patches
- Each 4×4 patch (16 pixels) is converted into a single 96-dimensional vector
- 512/4 = 128, so we get a 128×128 grid of 96-dim vectors
**Why 4×4 patches?**
- Individual pixels are too fine-grained for Mamba processing — you'd have 262,144
positions to process
- 4×4 patches give 16,384 positions — much more manageable
- Adjacent pixels within a patch are similar anyway; no information loss
- This is identical to how Vision Transformers (ViT) work
## PatchMerging — The Downsampling Layer
Between each encoder scale, spatial resolution is halved and channels are doubled:
```
Input: [B, C, H, W] e.g., [B, 96, 128, 128]
Step 1: Gather 4 neighboring patches
top-left = input[:, :, 0::2, 0::2] → [B, C, H/2, W/2]
top-right = input[:, :, 0::2, 1::2] → [B, C, H/2, W/2]
bottom-left = input[:, :, 1::2, 0::2] → [B, C, H/2, W/2]
bottom-right = input[:, :, 1::2, 1::2] → [B, C, H/2, W/2]
Step 2: Concatenate along channel dimension
merged = cat([TL, TR, BL, BR], dim=1) → [B, 4C, H/2, W/2]
Step 3: Linear projection to reduce channels
output = Linear(4C → 2C) → [B, 2C, H/2, W/2]
```
**Why this design?**
- Takes ALL 4 neighboring patches → no information is thrown away
- Spatial size halves (128→64)
- Channel doubling (96→192) compensates — total information is roughly conserved
- Linear projection mixes information from the 4 patches into a richer single
representation
## Full VM-UNet Architecture for Stage 1
```
INPUT: [B, 1, 512, 512]
↓
┌─────────────────────────────────────────────────────┐
│ Patch Embedding: Conv2d(1, 96, 4, stride=4) │
│ Output: [B, 96, 128, 128] │
└─────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│ ENCODER SCALE 1 │
│ Input: [B, 96, 128, 128] │
│ VSS Block 1: [B, 96, 128, 128] → [B, 96, 128, 128]│
│ VSS Block 2: [B, 96, 128, 128] → [B, 96, 128, 128]│
│ Output: [B, 96, 128, 128] ←── SKIP CONNECTION 1 │
└─────────────────────────────────────────────────────┘
↓ PatchMerging
┌─────────────────────────────────────────────────────┐
│ ENCODER SCALE 2 │
│ Input: [B, 192, 64, 64] │
│ VSS Block 1: [B, 192, 64, 64] → [B, 192, 64, 64] │
│ VSS Block 2: [B, 192, 64, 64] → [B, 192, 64, 64] │
│ Output: [B, 192, 64, 64] ←── SKIP CONNECTION 2 │
└─────────────────────────────────────────────────────┘
↓ PatchMerging
┌─────────────────────────────────────────────────────┐
│ ENCODER SCALE 3 │
│ Input: [B, 384, 32, 32] │
│ VSS Block 1: [B, 384, 32, 32] → [B, 384, 32, 32] │
│ VSS Block 2: [B, 384, 32, 32] → [B, 384, 32, 32] │
│ Output: [B, 384, 32, 32] ←── SKIP CONNECTION 3 │
└─────────────────────────────────────────────────────┘
↓ PatchMerging
┌─────────────────────────────────────────────────────┐
│ BOTTLENECK │
│ Input: [B, 768, 16, 16] │
│ VSS Block 1: [B, 768, 16, 16] → [B, 768, 16, 16] │
│ VSS Block 2: [B, 768, 16, 16] → [B, 768, 16, 16] │
│ Output: [B, 768, 16, 16] ← THIS IS "F" (for BYOL)│
└─────────────────────────────────────────────────────┘
↓ Bilinear 2× Upsample + Linear(768→384)
┌─────────────────────────────────────────────────────┐
│ DECODER SCALE 3 │
│ Concat with SKIP 3: │
│ [B, 384, 32, 32] + [B, 384, 32, 32] = [B, 768, 32, 32]│
│ VSS Block 1: [B, 768, 32, 32] → [B, 384, 32, 32] │
│ VSS Block 2: [B, 384, 32, 32] → [B, 384, 32, 32] │
└─────────────────────────────────────────────────────┘
↓ Bilinear 2× Upsample + Linear(384→192)
┌─────────────────────────────────────────────────────┐
│ DECODER SCALE 2 │
│ Concat with SKIP 2: │
│ [B, 192, 64, 64] + [B, 192, 64, 64] = [B, 384, 64, 64]│
│ VSS Block 1: [B, 384, 64, 64] → [B, 192, 64, 64] │
│ VSS Block 2: [B, 192, 64, 64] → [B, 192, 64, 64] │
└─────────────────────────────────────────────────────┘
↓ Bilinear 2× Upsample + Linear(192→96)
┌─────────────────────────────────────────────────────┐
│ DECODER SCALE 1 │
│ Concat with SKIP 1: │
│ [B, 96, 128, 128] + [B, 96, 128, 128] = [B, 192, 128, 128]│
│ VSS Block 1: [B, 192, 128, 128] → [B, 96, 128, 128]│
│ VSS Block 2: [B, 96, 128, 128] → [B, 96, 128, 128]│
└─────────────────────────────────────────────────────┘
↓ Bilinear 4× Upsample (back to full resolution)
┌─────────────────────────────────────────────────────┐
│ [B, 96, 512, 512] │
│ Segmentation Head: Conv2d(96, 7, kernel=1) │
│ Output: [B, 7, 512, 512] ← LOGITS │
│ Softmax(dim=1) │
│ Output: [B, 7, 512, 512] ← THIS IS "S" │
└─────────────────────────────────────────────────────┘
```
**"Bilinear 2× + Linear" explained:**
1. **Bilinear 2×:** Spatially upsample from H×W to 2H×2W using smooth interpolation
(no artifacts)
2. **Linear:** Apply Linear layer (or 1×1 Conv) to change channel count (e.g.,
768→384)
These are two separate operations applied one after another.
## The Three Outputs of Stage 1
### Output 1: S — Soft Segmentation Map [B, 7, 512, 512]
After the segmentation head produces raw scores (logits), softmax converts them to
probabilities:
```python
logits = seg_head(decoder_output) # [B, 7, 512, 512]
S = softmax(logits, dim=1) # [B, 7, 512, 512]
# dim=1 means softmax is applied across the 7 classes for each pixel
# After softmax: S[:, :, h, w] sums to 1.0 for every pixel (h,w)
```
**Physical meaning of S:**
```
S[b, 0, h, w] = probability pixel (h,w) in image b is background
S[b, 1, h, w] = probability pixel (h,w) in image b is liver/spleen
S[b, 2, h, w] = probability pixel (h,w) in image b is kidney
S[b, 3, h, w] = probability pixel (h,w) in image b is vessel
S[b, 4, h, w] = probability pixel (h,w) in image b is lung
S[b, 5, h, w] = probability pixel (h,w) in image b is bone
S[b, 6, h, w] = probability pixel (h,w) in image b is soft tissue
```
For a pixel clearly inside the liver:
```
S[b, :, h, w] ≈ [0.01, 0.93, 0.02, 0.01, 0.01, 0.01, 0.01]
```
For a boundary pixel between liver and background:
```
S[b, :, h, w] ≈ [0.30, 0.55, 0.05, 0.02, 0.03, 0.02, 0.03]
```
This soft uncertainty at boundaries is informative — it tells Stage 2 "this pixel
is ambiguous, blend the conditioning."
### Output 2: e_a — Anatomy Embeddings [B, 7, C]
After computing S, we compute organ-level feature vectors using **masked average
pooling:**
```python
decoder_features = decoder_output # [B, C, 512, 512]
# C = 96 (the features just before the seg head)
e_a = []
for k in range(7): # for each organ class
# Get the probability map for class k
weight = S[:, k, :, :] # [B, 512, 512]
# Use it as a soft attention weight to pool decoder features
weight_expanded = weight.unsqueeze(1) # [B, 1, 512, 512]
# Weight the features: at liver pixels, use liver features
weighted_features = weight_expanded * decoder_features # [B, C, 512, 512]
# Sum over spatial dimensions
summed = weighted_features.sum(dim=[2, 3]) # [B, C]
# Normalize by total weight
total_weight = weight.sum(dim=[1, 2]).unsqueeze(1) # [B, 1]
# Weighted average = "what does the average liver pixel look like?"
e_a_k = summed / (total_weight + 1e-8) # [B, C]
e_a.append(e_a_k)
e_a = torch.stack(e_a, dim=1) # [B, 7, C]
```
**Physical meaning of e_a:**
```
e_a[b, 0, :] = average feature vector of background pixels in image b (C-dim)
e_a[b, 1, :] = average feature vector of liver/spleen pixels in image b
e_a[b, 2, :] = average feature vector of kidney pixels in image b
...
```
This is a patient-specific anatomy description. Each patient's liver looks slightly
different (size, shape, texture due to conditions). e_a captures this. It's what
makes the conditioning patient-adaptive.
### Output 3: F — Bottleneck Features [B, 768, 16, 16]
The raw feature map at the bottleneck. Used only for BYOL training. NOT passed to
Stage 2.
## Stage 1 Loss Functions
### L_seg — Primary Loss (Cross Entropy)
```
L_seg = CrossEntropy(S, pseudo_labels, label_smoothing=0.1)
Cross Entropy = -sum_k [ true_label_k * log(predicted_prob_k) ]
For a liver pixel (true class = 1):
With smooth labels: true_label = [0.014, 0.914, 0.014, 0.014, 0.014, 0.014,
0.014]
L_seg = -(0.014*log(S[0]) + 0.914*log(S[1]) + 0.014*log(S[2]) + ...)
If S[1] is high (correctly predicts liver) → log(S[1]) is small negative → low
loss
If S[1] is low (wrong prediction) → log(S[1]) is large negative → high loss
```
This directly teaches the network: "for this pixel, S[1] should be high."
**Weight = 1.0** → This is the main objective.
### L_byol — BYOL Self-Supervised Loss (explained in next chapter)
**Weight = 0.1, starts at epoch 5**
### Total Stage 1 Loss
```
L_stage1 = 1.0 * L_seg + 0.1 * L_byol
```
---
# CHAPTER 9: BYOL — MAKING FEATURES NOISE-INVARIANT
## The Problem BYOL Solves
Stage 1 is trained on NDCT (clean) images. But during Stage 2 training, Stage 1 is
run on LDCT (noisy) images to generate S and e_a as conditioning signals.
If Stage 1 only learned from clean images, its features for a noisy LDCT image of
the liver might be different from its features for the clean NDCT image of the same
liver. This inconsistency would give Stage 2 bad conditioning.
**We need:** Stage 1's features to be invariant to noise level. The liver at 25%
dose should produce the same e_a as the liver at 100% dose.
## What is BYOL?
BYOL = Bootstrap Your Own Latent. A self-supervised learning method that forces the
network to produce the same representation for two differently corrupted versions
of the same image.
"Self-supervised" means the training signal comes from the data itself, not from
human labels.
## BYOL Architecture: Two Networks
```
VIEW 1 (NDCT + small noise) VIEW 2 (LDCT or NDCT + heavy noise)
↓ ↓
┌─────────────────────┐ ┌─────────────────────┐
│ ONLINE NETWORK │ │ TARGET NETWORK │
│ (actively trained)│ │ (EMA copy, frozen) │
│ │ │ │
│ VM-UNet Encoder │ │ VM-UNet Encoder │
│ ↓ │ │ ↓ │
│ Projector MLP │ │ Projector MLP │
│ ↓ │ │ ↓ │
│ z_online [B,256] │ │ z_target [B,256] │
│ ↓ │ │ │
│ Predictor MLP │ │ (NO predictor!) │
│ ↓ │ │ │
│ q_online [B,256] │ │ │
└─────────────────────┘ └─────────────────────┘
│ │
└───────────────┬───────────────┘
↓
L_byol = 2 - 2*cosine_sim(q_online, z_target.detach())
```
## Every Component Explained
### Projector MLP
Converts the bottleneck features F [B, 768, 16, 16] to a compact vector:
```python
# Step 1: Global Average Pooling — average across all spatial positions
z = F.mean(dim=[2,3]) # [B, 768]
# Step 2: Two-layer MLP
z = Linear(768, 4096)(z) # expand
z = BatchNorm(z) # normalize
z = ReLU(z) # nonlinearity
z = Linear(4096, 256)(z) # compress
# Output z: [B, 256]
```
### Predictor MLP (online network only)
```python
q = Linear(256, 4096)(z) # expand
q = BatchNorm(q)
q = ReLU(q)
q = Linear(4096, 256)(q) # compress
# Output q: [B, 256]
```
### EMA Target Network
```python
# After EVERY training step:
tau = 0.996 # starts here, linearly increases to 1.0 over training
for param_online, param_target in zip(online.parameters(), target.parameters()):
param_target.data = tau * param_target.data + (1 - tau) * param_online.data
# Target changes SLOWLY — 99.6% old values, 0.4% new values
# No gradient flows through this
```
### Cosine Similarity
```
cosine_sim(a, b) = dot_product(a, b) / (||a|| * ||b||)
= (a · b) / (|a| * |b|)
Range: -1 to +1
+1 = vectors point in same direction (identical representations)
0 = perpendicular (unrelated)
-1 = opposite directions
```
### BYOL Loss
```
L_byol = 2 - 2 * cosine_sim(q_online_from_view1, z_target_from_view2.detach())
+ 2 - 2 * cosine_sim(q_online_from_view2, z_target_from_view1.detach())
```
**When loss = 0:** q and z are identical (cosine_sim = 1) → the network produces
the same representation for both noise levels → goal achieved.
**When loss = 4:** cosine_sim = -1 → representations are opposite → worst case.
**.detach()** means "stop gradient here" — no gradient flows back through z_target.
Gradients only update the online network.
## Why the Asymmetry (Predictor Only in Online)?
Without the predictor and EMA asymmetry, the trivial solution is:
```
"Map everything to the same constant vector"
q = z = [0, 0, 0, ..., 0]
cosine_sim([0,...,0], [0,...,0]) = undefined but ≈ 1
L_byol = 0 ← trivially minimized!
```
This is called **collapse** — both views map to the same thing, but nothing
meaningful is learned.
The predictor + EMA + stop_gradient combination prevents collapse:
- The predictor must transform z_online to predict z_target
- z_target is a slowly-moving, stable target (due to EMA)
- The online network must learn informative representations to successfully predict
the target
- If both networks learned to collapse, the predictor couldn't predict the target
because both would be constant — but the EMA target is updated from the online,
creating a feedback loop that prevents collapse
---
# CHAPTER 10: STAGE 2 — THE VSSD ANATOMY-CONDITIONED DENOISER
## What Stage 2 Needs to Do
Take a noisy LDCT image and predict the clean NDCT image, using anatomy information
from Stage 1 to apply organ-specific denoising.
## Residual Diffusion — How Stage 2 Frames the Problem
### What is the Residual?
```
true_residual = x_ndct - x_ldct
Example:
NDCT pixel value: 150 HU
LDCT pixel value: 130 HU (noise made it wrong)
Residual: 150 - 130 = 20 HU (what needs to be added back)
```
Instead of learning to predict the full clean image (a hard task), Stage 2 learns
to predict this small correction. Most pixels have small residuals (close to 0).
The network only needs to learn "what correction to apply."
### What is the Diffusion Process?
Diffusion models work by:
1. **Forward process (training):** Add increasing amounts of Gaussian noise to the
target residual
2. **Reverse process (inference):** Learn to iteratively remove noise from pure
noise → residual
**Forward process:**
```
At timestep t (from 0 to T):
noisy_residual_t = sqrt(α_t) * true_residual + sqrt(1 - α_t) * ε
where:
ε ~ N(0, 1) (pure Gaussian noise)
α_t = noise schedule (decreases from 1 to 0 as t increases)
At t=0: noisy_residual ≈ true_residual (almost no noise added)
At t=T: noisy_residual ≈ ε (almost pure noise, residual information destroyed)
```
**During training:** Sample random t, add t-level noise to the residual, train the
network to predict the noise ε.
**During inference:** Start from pure noise, run the reverse process for T steps:
```
Step T: pure noise → slightly less noisy
Step T-1: slightly less noisy → more signal
...
Step 0: almost pure residual → final prediction
Then: denoised = x_ldct + predicted_residual
```
### The Timestep Embedding
The network needs to know what timestep t it's operating at (how noisy is its
input?).
```python
# Convert scalar t to a high-dimensional vector using sinusoidal encoding:
t_emb = sinusoidal_embedding(t) # [B, 64]
# Then expand with MLP:
t_emb = Linear(64, 256)(t_emb)
t_emb = SiLU(t_emb)
t_emb = Linear(256, 256)(t_emb)
# Final: [B, 256]
```
**Why sinusoidal?** Encodes the scalar timestep as a pattern of sine/cosine waves
at different frequencies, similar to positional encoding in Transformers. This
gives a unique, smooth representation for each timestep value.
## Since You Use Fixed Dose — No DA-CLIP
The original FoundDiff used DA-CLIP to encode dose level information. Since you're
working with a single fixed dose (25%), you don't need this. The dose embedding
`e_d` is simply removed. The timestep embedding `t_emb` handles all conditioning
related to the diffusion process.
## The AnatomyMamba_block — Core Innovation of Stage 2
This is the modified Mamba block that receives both anatomy conditioning signals.
Let's go through every single operation:
### Inputs to One Block
```
x: [B, C, H_feat, W_feat] ← current feature map (e.g., [B, 128, 256, 256])
S_feat: [B, 7, H_feat, W_feat] ← segmentation map at this scale (e.g., [B, 7,
256, 256])
e_a: [B, 7, C_anatomy] ← organ embeddings (e.g., [B, 7, 96])
t_emb: [B, 256] ← timestep embedding
```
### Step 1: adaLN-Zero (Adaptive LayerNorm)
**Standard LayerNorm:** normalizes features, applies learned γ and β (same for all
inputs).
**adaLN-Zero:** γ and β are computed from the timestep embedding — they change
based on what timestep t the diffusion is at.
```python
# Compute time-dependent scale and shift:
t_conditioning = Linear(256, 2*C)(t_emb) # [B, 2*C]
gamma_t, beta_t = t_conditioning.chunk(2, dim=1) # each [B, C]
# Apply to feature map:
# Reshape for broadcasting: [B, C] → [B, C, 1, 1]
gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
beta_t = beta_t.unsqueeze(-1).unsqueeze(-1)
x_norm = LayerNorm(x) # [B, C, H, W]
x = (1 + gamma_t) * x_norm + beta_t
```
**Why "Zero" initialization?** The Linear layer that produces gamma_t and beta_t is
initialized to output zeros. So at the very start of training:
- gamma_t = 0 → (1 + 0) * x_norm + 0 = x_norm → identity transformation
- The block starts doing nothing, then gradually learns what to do
- This prevents training instability at initialization
**Physical meaning:** At different timesteps, the network should behave
differently. At t=1000 (nearly pure noise input), the network needs aggressive
denoising. At t=1 (nearly clean input), subtle refinement. adaLN-Zero tells every
layer what "level of denoising" is currently needed.
### Step 2: Spatial FiLM — Anatomy Spatial Conditioning
**FiLM = Feature-wise Linear Modulation**
Standard FiLM: one (γ, β) per channel, same at all spatial positions.
Spatial FiLM: different (γ, β) at each spatial position, determined by organ
probabilities.
```python
class SpatialFiLM(nn.Module):
def __init__(self, num_classes=7, hidden_size=C):
self.film_net = nn.Sequential(
# Input: S_feat [B, 7, H, W] — 7 organ probs per pixel
nn.Conv2d(7, 32, kernel_size=3, padding=1), # [B, 32, H, W]
nn.SiLU(),
nn.Conv2d(32, 2*C, kernel_size=1) # [B, 2C, H, W]
# First C channels = gamma, last C channels = beta
)
def forward(self, x, S_feat):
# S_feat: [B, 7, H, W]
film_params = self.film_net(S_feat) # [B, 2C, H, W]
gamma, beta = film_params.chunk(2, dim=1) # each [B, C, H, W]
x_norm = LayerNorm(x) # [B, C, H, W]
return (1 + gamma) * x_norm + beta # [B, C, H, W]
```
**What this accomplishes:**
For a pixel at position (h, w) that is 90% liver (S[1,h,w] = 0.90):
- The conv network sees S[:,h,w] = [0.01, 0.90, 0.02, ...]
- It outputs gamma and beta values specific to "high liver probability"
- The feature map at (h,w) gets liver-specific normalization
For a pixel at a boundary (50% liver, 40% background):
- S[:,h,w] = [0.40, 0.50, ...]
- The network outputs a blend of liver and background conditioning
- Smooth transition, no artifacts
**Physical interpretation:** Each organ has different noise statistics.
- Liver: moderate noise, relatively uniform
- Lung: heavy noise in air regions, very dark
- Bone: low relative noise (high signal)
The FiLM network learns to apply different normalization strategies appropriate for
each organ's noise characteristics. A liver pixel gets "liver-mode" normalization.
A lung pixel gets "lung-mode" normalization.
**S at different scales:**
The UNet processes features at multiple resolutions. S starts at [B, 7, 512, 512].
Before passing to each scale's blocks, S is downsampled to match:
```python
S_256 = F.interpolate(S, (256, 256), mode='bilinear') # for scale 1
S_128 = F.interpolate(S, (128, 128), mode='bilinear') # for scale 2
S_64 = F.interpolate(S, (64, 64), mode='bilinear') # for scale 3
S_32 = F.interpolate(S, (32, 32), mode='bilinear') # for scale 4
```
Bilinear interpolation: when going from 512×512 to 256×256, each new pixel is a
weighted average of 4 nearby original pixels. The probability values are preserved
proportionally.
### Step 3: VSSD Non-Causal Scan
```python
x_bhwc = x.permute(0, 2, 3, 1) # [B, C, H, W] → [B, H, W, C]
x_bhwc = vssd_block(x_bhwc) # non-causal 4-directional scan
x = x_bhwc.permute(0, 3, 1, 2) # [B, H, W, C] → [B, C, H, W]
```
After Spatial FiLM applied organ-specific normalization, the VSSD scan propagates
information globally across the entire image. Each pixel now incorporates
information from all other pixels, understanding the full image context.
### Step 4: Cross-Attention with e_a — Semantic Organ Query
```python
class CrossAttentionWithAnatomy(nn.Module):
def __init__(self, hidden_size, anatomy_dim, num_heads):
self.attn = nn.MultiheadAttention(
embed_dim=hidden_size, # query dimension
kdim=anatomy_dim, # key dimension (from e_a)
vdim=anatomy_dim, # value dimension (from e_a)
num_heads=num_heads,
batch_first=True
)
def forward(self, x, e_a):
# x: [B, C, H, W] ← current feature map
# e_a: [B, 7, C_anat] ← 7 organ embedding vectors
B, C, H, W = x.shape
# Flatten spatial dims to make sequence:
x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
# x_seq: [B, H*W, C] ← each pixel position is one element in the sequence
# Cross-attention:
# Queries Q = x_seq → "each pixel asks: which organ am I related to?"
# Keys K = e_a → "these are the 7 organ descriptions"
# Values V = e_a → "these are the organ feature vectors to retrieve"
attended, attn_weights = self.attn(
query=x_seq, # [B, H*W, C] ← each pixel is a query
key=e_a, # [B, 7, C_anat] ← 7 organ keys
value=e_a # [B, 7, C_anat] ← 7 organ values
)
# attended: [B, H*W, C] ← each pixel retrieved organ information
# attn_weights:[B, H*W, 7] ← how much each pixel attended to each organ
# Reshape back:
attended = attended.reshape(B, H, W, C).permute(0, 3, 1, 2)
# [B, C, H, W]
return x + attended # residual connection
```
**What is attention? (Deep explanation)**
Attention computes how relevant each "key" is to each "query":
```
For a single query q and set of keys [k1, k2, ..., k7]:
attention_score_i = dot(q, k_i) / sqrt(d_k) ← how similar is q to key i?
attention_weights = softmax([score_1, ..., score_7]) ← normalize to sum=1
output = sum_i [weight_i * value_i] ← weighted sum of values
```
**Physical meaning for our case:**
Each pixel position in the feature map is a "query." It asks: "which of the 7 organ
embeddings (keys) do I look most like?"
A pixel position that has processed visual features resembling liver will have a
high dot product with e_a[:,1,:] (the liver embedding) → high attention weight on
liver → retrieves mostly the liver embedding as its value.
This gives each pixel access to patient-specific organ appearance information:
- e_a[:,1,:] = "what does THIS patient's liver look like in terms of learned
features"
- A liver pixel attending to e_a[:,1,:] retrieves this patient-specific liver
description
- This information is injected into the pixel's representation
- The denoiser now knows not just "this is a liver region" but "this is what liver
looks like in this specific patient"
**Why is this different from Spatial FiLM?**
| Spatial FiLM | Cross-Attention |
|---|---|
| Uses S (probability map) | Uses e_a (feature embeddings) |
| Answers: "WHERE are the organs?" | Answers: "WHAT does each organ look like?" |
| Per-pixel spatial prior | Per-organ semantic descriptor |
| Pixel at (h,w) gets conditioning based on S[h,w] | Pixel queries e_a to find most
similar organ |
| Spatial/location-based | Content/similarity-based |
Together they give complete anatomy conditioning: spatial location + semantic
appearance.
**Multi-head attention:**
Instead of one attention computation, multi-head attention runs several in
parallel:
```
num_heads = C // 32 (e.g., if C=128, then 4 heads)
Head 1: queries, keys, values each of dim 32 → looks for one aspect
Head 2: queries, keys, values each of dim 32 → looks for another aspect
Head 3: ...
Head 4: ...
Outputs concatenated → same total dimension as input
```
Different heads learn to attend to different aspects of organ appearance (texture
vs. shape vs. boundary characteristics).
### Step 5: Existing Attention Block
The original FoundDiff had a spatial self-attention block in the mid-block. Keep
this as-is:
```python
x = existing_attn_block(x) # self-attention within the feature map
```
### Step 6: Residual Connection
```python
output = input + x
```
The input before all modifications is added back. The entire block only learns
corrections.
## Stage 2 Full Architecture
Since you're using fixed dose (no DA-CLIP), the architecture simplifies:
```
INPUTS:
x_ldct: [B, 1, 512, 512] ← low-dose CT (the noisy input)
x_noisy: [B, 1, 512, 512] ← noisy residual at timestep t
t: scalar ← diffusion timestep
S_scales: list of [B, 7, H, W] at multiple resolutions
e_a: [B, 7, C_anatomy] ← from frozen Stage 1
NETWORK:
init_conv: Conv2d(2, 64, 7, padding=3)
Input: cat([x_noisy, x_ldct], dim=1) = [B, 2, 512, 512]
Output: [B, 64, 512, 512]
Note: concatenate noisy residual AND original LDCT as 2-channel input
Why? Network can see both what it's denoising AND the original noisy image
time_mlp: sinusoidal_embedding → Linear(64, 256) → SiLU → Linear(256, 256)
Input: t (scalar)
Output: [B, 256]
DOWNSAMPLING PATH (with AnatomyMamba_block):
Scale 1 (dim=64):
Input: [B, 64, 512, 512]
AnatomyMamba_block × N (using S_512 and e_a)
Downsample: → [B, 128, 256, 256]
Saved as skip connection
Scale 2 (dim=128):
Input: [B, 128, 256, 256]
AnatomyMamba_block × N (using S_256 and e_a)
Downsample: → [B, 256, 128, 128]
Saved as skip connection
Scale 3 (dim=256):
Input: [B, 256, 128, 128]
AnatomyMamba_block × N (using S_128 and e_a)
Downsample: → [B, 512, 64, 64]
Saved as skip connection
Scale 4 (dim=512):
Input: [B, 512, 64, 64]
AnatomyMamba_block × N (using S_64 and e_a)
Downsample: → [B, 512, 32, 32]
Saved as skip connection
BOTTLENECK (mid block):
Input: [B, 512, 32, 32]
AnatomyMamba_block (using S_32 and e_a)
Output: [B, 512, 32, 32]
← Seg KD head attached here (for L_kd loss):
Conv2d(512, 7, 1) → upsampled → [B, 7, 512, 512]
UPSAMPLING PATH:
Scale 4 up:
Concat with skip: [B, 1024, 32, 32]
AnatomyMamba_block × N (using S_32 and e_a)
Upsample: → [B, 256, 64, 64]
Scale 3 up:
Concat with skip: [B, 512, 64, 64]
AnatomyMamba_block × N (using S_64 and e_a)
Upsample: → [B, 128, 128, 128]
Scale 2 up:
Concat with skip: [B, 256, 128, 128]
AnatomyMamba_block × N (using S_128 and e_a)
Upsample: → [B, 64, 256, 256]
Scale 1 up:
Concat with skip: [B, 128, 256, 256]
AnatomyMamba_block × N (using S_256 and e_a)
Upsample: → [B, 64, 512, 512]
final_conv: Conv2d(64, 1, 1)
Output: [B, 1, 512, 512] ← predicted noise/residual
```
## How Stage 1 Outputs Flow Into Stage 2 During Training
```python
# ======================================================
# At the START of every single training step:
# ======================================================
# 1. Get anatomy conditioning from frozen Stage 1
with torch.no_grad(): # CRITICAL: absolutely no gradients through Stage 1
stage1_out = frozen_stage1(x_ldct)
S = stage1_out.S # [B, 7, 512, 512]
e_a = stage1_out.e_a # [B, 7, 96]
# 2. Precompute S at all UNet scales
S_256 = F.interpolate(S, (256, 256), mode='bilinear', align_corners=False)
S_128 = F.interpolate(S, (128, 128), mode='bilinear', align_corners=False)
S_64 = F.interpolate(S, (64, 64), mode='bilinear', align_corners=False)
S_32 = F.interpolate(S, (32, 32), mode='bilinear', align_corners=False)
# 3. Prepare diffusion inputs
true_residual = x_ndct - x_ldct # [B, 1, 512, 512]
t = torch.randint(0, T, (B,)) # random timestep
noise = torch.randn_like(true_residual) # Gaussian noise
noisy_residual = diffusion.q_sample(true_residual, t, noise) # add noise
# 4. Run Stage 2 UNet
predicted_noise = stage2_unet(
x = noisy_residual, # [B, 1, 512, 512]
t = t, # timestep
x_ldct = x_ldct, # concatenated in init_conv
S_scales = [S_256, S_128, S_64, S_32],
e_a = e_a
)
```
---
# CHAPTER 11: STAGE 2 LOSS FUNCTIONS
## L_res — Primary Denoising Loss
```python
L_res = MSE(predicted_noise, noise)
# OR equivalently:
L_res = MSE(predicted_residual, true_residual)
# (both formulations are used in practice, they're mathematically equivalent
# for simple noise schedules)
Weight: 1.0
Active: from step 0
```
This is the core objective: correctly predict the residual between LDCT and NDCT.
## L_kd — Knowledge Distillation Loss
```python
# The Seg KD Head:
# A small decoder attached to Stage 2's bottleneck
bottleneck = stage2_unet.bottleneck_output # [B, 512, 32, 32]
seg_pred = ConvTranspose2d(512, 256, 4, stride=2, padding=1)(bottleneck) # [B,
256, 64, 64]
seg_pred = ReLU(seg_pred)
seg_pred = ConvTranspose2d(256, 7, 4, stride=2, padding=1)(seg_pred) # [B, 7,
128, 128]
seg_pred = F.interpolate(seg_pred, (512, 512)) # [B, 7,
512, 512]
L_kd = CrossEntropy(seg_pred, pseudo_labels, label_smoothing=0.1)
Weight: 0.1
Active: after 50k steps
```
**Why does Stage 2 need segmentation loss?**
Stage 2 uses cross-attention with e_a. For this to work, Stage 2's feature map must
be "anatomy-aware" — features must carry some organ identity information, so that
when they're used as queries against e_a, they attend to the correct organ
embedding.
If Stage 2's features are pure noise-texture features with no anatomical meaning,
cross-attention becomes random — liver pixels accidentally attend to bone
embeddings, etc.
L_kd forces Stage 2's bottleneck to be interpretable in terms of organ classes,
making cross-attention meaningful.
## L_anatomy — Feature-Space Anatomy Matching Loss
This is the most novel loss. After Stage 2 predicts the denoised image, run Stage 1
on it and verify the anatomy features match:
```python
# This is applied every 5th step (expensive computation)
if step % 5 == 0 and step > 150000:
# Get denoised image estimate
x_hat = x_ldct + predicted_residual.detach() # [B, 1, 512, 512]
# .detach() prevents gradient from flowing back through x_hat into residual
prediction
# Actually we DO want gradient for L_anatomy, so remove .detach() from this
line
with torch.no_grad():
# Run frozen Stage 1 on the denoised output
e_a_pred = frozen_stage1(x_hat).e_a # [B, 7, C]
# Run frozen Stage 1 on the ground truth clean image
e_a_gt = frozen_stage1(x_ndct).e_a # [B, 7, C]
# Note: e_a_gt could be precomputed and cached for efficiency
L_anatomy = F.l1_loss(e_a_pred, e_a_gt)
# = mean(|e_a_pred - e_a_gt|)
Weight: 0.05
Active: after 150k steps
Applied: every 5th step to save computation
```
**What this loss does physically:**
If the denoiser blurs the liver boundary:
- Stage 1 running on x_hat sees a blurry liver → produces an e_a where the liver
region is less well-defined
- Stage 1 running on x_ndct sees a sharp liver → produces an e_a with well-defined
liver features
- L_anatomy = difference between these two e_a → non-zero → penalizes the blurring
This creates a direct feedback signal: "your denoised image has different anatomy
features from the clean reference — you've distorted the anatomy."
**Why L1 not L2?**
L2 = squared difference = heavily penalizes large differences. Some organ classes
are tiny (vessels) — their e_a features might differ a lot even for small
segmentation differences. L1 is more robust to these occasional large feature
differences for small/rare organs.
## Total Stage 2 Loss at Each Training Phase
```
Phase 1 (0 to 50k steps):
L = L_res
Phase 2 (50k to 150k steps):
L = L_res + 0.1 * L_kd
Phase 3 (150k+ steps):
L = L_res + 0.1 * L_kd + 0.05 * L_anatomy (L_anatomy every 5th step)
```
Why progressive? Adding all losses at once creates conflicting gradients before the
network has learned anything useful. Starting with just L_res lets the denoiser
establish basic denoising ability. Then L_kd makes features anatomy-aware. Then
L_anatomy fine-tunes anatomy preservation.
---
# CHAPTER 12: COMPLETE DATA FLOW — ONE TRAINING STEP
Let's trace exactly what happens during one complete training step for Stage 2:
```
════════════════════════════════════════════════════════════════
INPUT DATA (one batch):
x_ldct: [8, 1, 512, 512] ← 8 low-dose CT slices
x_ndct: [8, 1, 512, 512] ← 8 paired clean CT slices
pseudo_labels: [8, 512, 512] ← organ class per pixel (integers 0-6)
════════════════════════════════════════════════════════════════
STEP A: Get anatomy conditioning (no gradients)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with torch.no_grad():
S, e_a = frozen_vm_unet(x_ldct)
# S: [8, 7, 512, 512] ← organ probabilities for all 8 images
# e_a: [8, 7, 96] ← organ embeddings for all 8 images
S_256 = interpolate(S, 256) # [8, 7, 256, 256]
S_128 = interpolate(S, 128) # [8, 7, 128, 128]
S_64 = interpolate(S, 64) # [8, 7, 64, 64]
S_32 = interpolate(S, 32) # [8, 7, 32, 32]
════════════════════════════════════════════════════════════════
STEP B: Prepare diffusion inputs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
true_residual = x_ndct - x_ldct # [8, 1, 512, 512]
t = randint(0, 1000, size=(8,)) # random timestep per image
e.g., t = [423, 17, 891, 234, 657, 98, 445, 772]
noise = randn_like(true_residual) # [8, 1, 512, 512] pure Gaussian noise
# Add noise to residual according to timestep:
# Higher t = more noise added
noisy_residual = q_sample(true_residual, t, noise) # [8, 1, 512, 512]
════════════════════════════════════════════════════════════════
STEP C: Time embedding
━━━━━━━━━━━━━━━━━━━━━━
t_emb = sinusoidal_embedding(t) # [8, 64]
t_emb = time_mlp(t_emb) # [8, 256]
════════════════════════════════════════════════════════════════
STEP D: Stage 2 UNet forward pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concatenate inputs:
unet_input = cat([noisy_residual, x_ldct], dim=1) # [8, 2, 512, 512]
# init_conv:
features = init_conv(unet_input) # [8, 64, 512, 512]
# ENCODER:
# Scale 1:
features = AnatomyMamba_block(features, S_256, e_a, t_emb) # [8, 64, 512, 512]
└── inside: adaLN-Zero(t_emb) → SpatialFiLM(S_256) → VSSD → CrossAttn(e_a) →
residual
skip_1 = features # save for decoder
features = downsample(features) # [8, 128, 256, 256]
# Scale 2:
features = AnatomyMamba_block(features, S_128, e_a, t_emb) # [8, 128, 256, 256]
skip_2 = features
features = downsample(features) # [8, 256, 128, 128]
# Scale 3:
features = AnatomyMamba_block(features, S_64, e_a, t_emb) # [8, 256, 128, 128]
skip_3 = features
features = downsample(features) # [8, 512, 64, 64]
# Scale 4:
features = AnatomyMamba_block(features, S_32, e_a, t_emb) # [8, 512, 64, 64]
skip_4 = features
features = downsample(features) # [8, 512, 32, 32]
# BOTTLENECK:
features = AnatomyMamba_block(features, S_32, e_a, t_emb) # [8, 512, 32, 32]
bottleneck_features = features # save for L_kd
# DECODER:
# Scale 4 up:
features = cat([features, skip_4], dim=1) # [8, 1024, 32, 32]
features = AnatomyMamba_block(features, S_32, e_a, t_emb) # [8, 512, 32, 32]
features = upsample(features) # [8, 256, 64, 64]
# Scale 3 up:
features = cat([features, skip_3], dim=1) # [8, 512, 64, 64]
features = AnatomyMamba_block(features, S_64, e_a, t_emb) # [8, 256, 64, 64]
features = upsample(features) # [8, 128, 128, 128]
# Scale 2 up:
features = cat([features, skip_2], dim=1) # [8, 256, 128, 128]
features = AnatomyMamba_block(features, S_128, e_a, t_emb) # [8, 128, 128, 128]
features = upsample(features) # [8, 64, 256, 256]
# Scale 1 up:
features = cat([features, skip_1], dim=1) # [8, 128, 256, 256]
features = AnatomyMamba_block(features, S_256, e_a, t_emb) # [8, 64, 256, 256]
features = upsample(features) # [8, 64, 512, 512]
# final_conv:
predicted_noise = final_conv(features) # [8, 1, 512, 512]
════════════════════════════════════════════════════════════════
STEP E: Compute losses
━━━━━━━━━━━━━━━━━━━━━━
L_res = MSE(predicted_noise, noise) # always active
if step > 50000:
seg_kd_output = kd_head(bottleneck_features) # [8, 7, 512, 512]
L_kd = CrossEntropy(seg_kd_output, pseudo_labels, label_smoothing=0.1)
if step > 150000 and step % 5 == 0:
x_hat = x_ldct + predicted_noise # approximate denoised image
with torch.no_grad():
e_a_pred = frozen_stage1(x_hat).e_a # [8, 7, 96]
e_a_gt = frozen_stage1(x_ndct).e_a # [8, 7, 96]
L_anatomy = L1(e_a_pred, e_a_gt)
total_loss = L_res + 0.1*L_kd + 0.05*L_anatomy
════════════════════════════════════════════════════════════════
STEP F: Backpropagation and update
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
optimizer.zero_grad()
total_loss.backward()
# Gradients flow through Stage 2 ONLY (Stage 1 has no_grad)
optimizer.step()
# Only Stage 2 weights are updated
# Stage 1 weights: NEVER touched after Stage 1 training
```
---
# CHAPTER 13: WHY EVERY DESIGN CHOICE EXISTS
## Why VM-UNet Instead of Regular UNet for Stage 1?
Regular UNet uses convolutions (local, fixed receptive field). VM-UNet uses Mamba
blocks (global receptive field). Organ segmentation requires understanding global
context:
- Is this dark circular blob a kidney or a cyst? → Look at surrounding structures
(spine position, other organs)
- Where exactly is the liver border? → Need context from the entire right side of
the abdomen
VMamba's 4-directional scanning gives every pixel a global view of the entire image
at O(N) cost.
## Why Freeze Stage 1 During Stage 2 Training?
If Stage 1 were updated during Stage 2 training, the anatomy conditioning signals
(S and e_a) would change every step. Stage 2 would be trying to learn from a moving
target. It would never converge properly because the conditioning signal it's
learning to use keeps changing.
Frozen Stage 1 = stable, fixed anatomy signals = Stage 2 can learn to use them
reliably.
## Why Soft Segmentation (S) Instead of Hard Binary Masks?
Hard threshold: `liver_mask = (S[:,1,:,:] > 0.5)`
Problems:
1. **Not differentiable:** Can't backpropagate through a threshold → can't train
2. **Boundary artifacts:** Pixel A is labeled "liver" (full liver processing),
pixel B next to it is labeled "background" (background processing) → visible seam
in output
3. **TotalSegmentator errors amplified:** A boundary pixel wrongly labeled "liver"
gets full liver processing instead of uncertain blended processing
4. **Information loss:** S[liver]=0.51 is completely different from S[liver]=0.99,
but both become 1.0 after thresholding
Soft S preserves all of this information and is fully differentiable.
## Why Bilinear for Upsampling Not Transposed Convolution?
Transposed convolution (deconvolution) creates "checkerboard artifacts" —
alternating bright/dark pixel patterns caused by the stride creating uneven overlap
patterns. Bilinear upsampling is smooth (weighted average of neighbors) and
artifact-free. A linear projection layer afterwards can adjust channels.
## Why AdamW Not Adam?
Standard Adam applies weight decay as a gradient update, which interacts
incorrectly with Adam's adaptive learning rate scaling. AdamW applies weight decay
directly to the weights (separate from the gradient update), which is
mathematically correct and empirically works much better for
Transformer/Mamba-based networks.
## Why Progressive Loss Introduction?
Training is a curriculum:
1. **Steps 0-50k:** Learn basic denoising — establish what "remove noise" means
2. **Steps 50k-150k:** Add anatomy awareness via L_kd — make internal features
anatomy-relevant
3. **Steps 150k+:** Add L_anatomy — fine-tune so denoised images actually preserve
anatomy in output space
Starting with all losses simultaneously: the network has no good starting point.
L_anatomy gradients point in a direction that conflicts with L_res gradients before
either objective is stable. Training diverges or learns poor solutions.
---
# CHAPTER 14: THE SOFT SEGMENTATION APPROACH — PHYSICAL INTUITION
Imagine you're at a liver-background boundary pixel (row 256, column 312). In
reality, this pixel might be 60% liver tissue and 40% fat/background.
**What S looks like here:**
```
S[b, 0, 256, 312] = 0.35 (background probability)
S[b, 1, 256, 312] = 0.52 (liver probability — slightly higher)
S[b, 2, 256, 312] = 0.05 (kidney)
S[b, 3, 256, 312] = 0.03 (vessel)
S[b, 4, 256, 312] = 0.02 (lung)
S[b, 5, 256, 312] = 0.01 (bone)
S[b, 6, 256, 312] = 0.02 (soft tissue)
```
**What Spatial FiLM does at this pixel:**
The FiLM network sees this distribution and outputs gamma and beta values that are
a blend:
```
gamma[pixel] ≈ 0.35 * (background-gamma) + 0.52 * (liver-gamma) + small
contributions from others
beta[pixel] ≈ 0.35 * (background-beta) + 0.52 * (liver-beta) + small
contributions from others
```
The normalization at this pixel is a weighted mix of liver-mode and background-mode
normalization, weighted by how much each class is present. This is smooth,
continuous, and naturally handles the uncertainty.
**What Cross-attention does at this pixel:**
The pixel's feature vector [dim-C] is used as a query against e_a (7 organ
embeddings). Since this pixel has mixed liver/background features, it might attend:
```
attention_weights ≈ [0.30, 0.48, 0.08, 0.04, 0.03, 0.02, 0.05]
bg liver kidney ...
```
The retrieved value is a weighted combination of e_a[:,0,:] and e_a[:,1,:] — a
blend of background and liver patient-specific appearance. This is the correct
behavior for a boundary pixel.
---
# CHAPTER 15: EVALUATION METRICS
## Standard Pixel Metrics
**PSNR (Peak Signal-to-Noise Ratio):**
```
PSNR = 10 * log10(MAX² / MSE)
Higher is better. Measured in dB.
Typical good LDCT denoising: 35-42 dB
```
**SSIM (Structural Similarity Index):**
```
SSIM = f(luminance, contrast, structure)
Range: 0 to 1. Higher is better.
Typical good LDCT denoising: 0.85-0.95
```
## New Anatomy-Specific Metrics
**Anatomy Dice Score:**
```python
# Run TotalSegmentator on denoised images → predicted_masks
# Run TotalSegmentator on NDCT reference → reference_masks
for organ_k in range(7):
pred_binary = (predicted_masks == k) # binary mask for organ k
ref_binary = (reference_masks == k) # binary mask for organ k
intersection = (pred_binary & ref_binary).sum()
dice_k = 2 * intersection / (pred_binary.sum() + ref_binary.sum())
mean_dice = sum(dice_k for k in range(7)) / 7
```
High Dice = organ boundaries are preserved well in the denoised image.
**Boundary Preservation Score:**
```python
# Apply Sobel edge detector:
edges_denoised = sobel(denoised_image) # magnitude of gradients
edges_ndct = sobel(ndct_image) # magnitude of gradients
# Threshold to binary:
edge_map_pred = (edges_denoised > threshold)
edge_map_gt = (edges_ndct > threshold)
# F1 with 2-pixel tolerance:
# True positive = denoised edge pixel within 2 pixels of a reference edge pixel
F1 = 2*TP / (2*TP + FP + FN)
```
**Anatomy-Weighted SSIM:**
```python
organ_weight_map = torch.where(
S.max(dim=1).values > 0.5, # if any organ has >50% probability
torch.tensor(3.0), # organ pixels: 3× weight
torch.tensor(1.0) # background pixels: 1× weight
)
ssim_map = compute_ssim_map(denoised, ndct) # per-pixel SSIM values
weighted_ssim = (organ_weight_map * ssim_map).sum() / organ_weight_map.sum()
```
This penalizes organ pixel errors 3× more than background errors.
---
# CHAPTER 16: IMPLEMENTATION ORDER AND CHECKLIST
## Milestone 0: Pseudo-Label Generation
```
□ Install TotalSegmentator (pip install TotalSegmentator)
□ Write script to stack 2D slices → 3D volume per patient
□ Convert to NIfTI (SimpleITK)
□ Run TotalSegmentator on all NDCT patients
□ Remap 104 labels → 7 labels
□ Split 3D masks back to 2D .npy files
□ Verify: load one mask, visualize it overlaid on CT slice
```
## Milestone 1: VM-UNet Implementation
```
□ Clone VM-UNet repo (github.com/JCruan519/VM-UNet)
□ Change input channels: 3 → 1
□ Change output classes: 2 → 7
□ Add masked average pooling layer for e_a computation
□ Expose bottleneck features F
□ Test: input [2, 1, 512, 512] → output S [2, 7, 512, 512], e_a [2, 7, 96]
```
## Milestone 2: Stage 1 Training (L_seg only)
```
□ Write dataset loader with mask loading
□ Write training loop with CrossEntropy + label_smoothing=0.1
□ Log Dice per organ every 1000 steps
□ Target: Dice > 0.75 for liver, kidney, lung by step 50k
□ Save checkpoint
```
## Milestone 3: Add BYOL to Stage 1
```
□ Implement EMA target network copy
□ Implement Projector MLP and Predictor MLP
□ Implement noise augmentation (two views)
□ Add L_byol with weight 0.0 first 5 epochs, then 0.1
□ Monitor: BYOL loss should go from ~2.0 toward ~0.05
```
## Milestone 4: AnatomyMamba_block
```
□ Implement SpatialFiLM module
□ Implement CrossAttentionWithAnatomy module
□ Modify Mamba_block forward() to accept S_feat and e_a
□ Add adaLN-Zero for timestep conditioning
□ Replace SS2D with VSSD_Block
□ Test with dummy tensors: verify shapes
```
## Milestone 5: Stage 2 Training (L_res only)
```
□ Load frozen Stage 1 checkpoint
□ Implement data flow (Stage 1 runs with no_grad each step)
□ Implement S interpolation to all scales
□ Train with L_res only
□ Verify PSNR ≥ baseline FoundDiff results
```
## Milestones 6 and 7: Add L_kd and L_anatomy
```
□ Implement Seg KD head
□ Add L_kd at step 50k with weight 0.1
□ Monitor anatomy Dice on denoised outputs
□ Add L_anatomy at step 150k with weight 0.05
□ Apply every 5th step to control computation
□ Monitor: PSNR and Dice should both improve
```
---
# COMPLETE GLOSSARY OF ALL TERMS
| Term | Plain English Meaning |
|------|----------------------|
| **Tensor** | Multi-dimensional array of numbers (the basic data structure) |
| **[B, C, H, W]** | Batch size, Channels, Height, Width — standard 4D image tensor
|
| **BCHW** | Channel-first format (PyTorch default) |
| **BHWC** | Channel-last format (Mamba requires this) |
| **CT / LDCT / NDCT** | CT scan / Low-dose CT (noisy) / Normal-dose CT (clean) |
| **HU** | Hounsfield Units — CT intensity scale |
| **Voxel** | 3D pixel in CT (volume element) |
| **Pseudo-labels** | Auto-generated organ annotations (not manually labeled) |
| **Softmax** | Function that converts raw scores to probabilities summing to 1 |
| **Cross Entropy** | Loss function for classification: -sum(true_label *
log(prediction)) 
| **Label smoothing** | Adding small probability to all classes to prevent
overconfidence |
| **MSE** | Mean Squared Error: average of (predicted - true)² over all pixels |
| **L1 loss** | Mean Absolute Error: average of |predicted - true| |
| **Convolution** | Filter sliding over image; detects local patterns |
| **Stride** | How many pixels a conv filter jumps per step |
| **Kernel size** | Size of the convolution filter (3 = 3×3) |
| **Depthwise conv** | Conv where each channel has its own independent filter |
| **Downsampling** | Reducing spatial resolution (512→256→128...) |
| **Upsampling** | Increasing spatial resolution back (16→32→...→512) |
| **Bilinear interpolation** | Smooth upsampling via weighted average of neighbors
|
| **LayerNorm** | Normalization to zero mean, unit variance per position |
| **Residual connection** | output = layer(input) + input; helps gradient flow |
| **UNet** | Encoder-Bottleneck-Decoder with skip connections |
| **Encoder** | Part that reduces resolution; extracts abstract features |
| **Decoder** | Part that increases resolution; reconstructs spatial details |
| **Bottleneck** | Most compressed representation in UNet |
| **Skip connection** | Direct connection from encoder to corresponding decoder
scale |
| **PatchMerging** | Gather 4 neighbor patches, concat, project → halve H,W, double
C |
| **Patch embedding** | Convert raw image patches to feature vectors |
| **SSM** | State Space Model: sequence model maintaining hidden state |
| **Hidden state** | Compressed summary of all elements seen so far in a sequence |
| **Selective SSM (Mamba)** | SSM where A,B,C matrices depend on input content |
| **SS2D** | 2D Selective Scan: 4-directional causal Mamba scan (VMamba1) |
| **VSSD** | Non-causal 2D scan (VMamba2); every position sees all others |
| **Causal** | Position i can only see positions 0 to i (not future) |
| **Non-causal** | Every position sees all other positions simultaneously |
| **VSS Block** | Full VMamba building block:
LayerNorm→project→depthwise→SS2D→gate→residual |
| **Gating** | Learned per-feature on/off switch: output = features * SiLU(gate) |
| **SiLU** | Activation function: x * sigmoid(x); smooth, non-monotonic |
| **Masked average pooling** | Weighted average of features using organ probability
as weight |
| **S** | Soft segmentation map [B, 7, H, W] — organ probabilities per pixel |
| **e_a** | Anatomy embeddings [B, 7, C] — per-organ feature vectors |
| **F** | Bottleneck feature map from Stage 1 — used only for BYOL |
| **FiLM** | Feature-wise Linear Modulation: x_out = gamma * LayerNorm(x) + beta |
| **Spatial FiLM** | FiLM where gamma, beta are different at each spatial position
|
| **Gamma, Beta** | Scale and shift parameters in FiLM conditioning |
| **adaLN-Zero** | Adaptive LayerNorm where gamma, beta come from timestep
embedding |
| **Cross-attention** | Attention where Query comes from one source, Key/Value from
another |
| **Query, Key, Value** | Components of attention: Q asks a question, K matches it,
V provides answer |
| **Attention weight** | How much each query attends to each key (after softmax) |
| **Multi-head attention** | Run attention in parallel in multiple subspaces |
| **Diffusion model** | Learns to reverse a noise-adding process |
| **Timestep t** | How much noise has been added; model must know this to denoise
correctly |
| **Residual (in denoising)** | x_ndct - x_ldct; the correction needed to clean the
image |
| **Sinusoidal embedding** | Encode scalar t as pattern of sin/cos waves at
different frequencies |
| **BYOL** | Bootstrap Your Own Latent — self-supervised noise invariance training
|
| **Online network** | The actively trained network in BYOL |
| **Target network** | The slowly-moving EMA copy in BYOL (no direct gradient
updates) |
| **EMA** | Exponential Moving Average: target = τ*target + (1-τ)*online |
| **Projector MLP** | Small 2-layer network that maps bottleneck to compact vector
|
| **Predictor MLP** | Small network (online only) that predicts target's
representation |
| **Collapse** | Catastrophic failure where all inputs map to same representation |
| **Cosine similarity** | dot(a,b) / (|a| * |b|); measures angle between vectors |
| **L_seg** | Segmentation cross-entropy loss in Stage 1 |
| **L_byol** | BYOL noise-invariance loss in Stage 1 |
| **L_res** | Residual/denoising MSE loss in Stage 2 |
| **L_kd** | Knowledge distillation segmentation loss in Stage 2 |
| **L_anatomy** | Anatomy feature matching L1 loss in Stage 2 |
| **Seg KD head** | Small decoder on Stage 2 bottleneck that predicts organ classes
|
| **Frozen** | Weights set to not update (requires_grad=False) |
| **torch.no_grad()** | Context manager: don't build computation graph, no
gradients |
| **backpropagation** | Algorithm to compute gradients through the computation
graph |
| **AdamW** | Adam optimizer with decoupled weight decay regularization |
| **Dice score** | Overlap metric for segmentation: 2|A∩B|/(|A|+|B|) |
| **PSNR** | Peak Signal-to-Noise Ratio; higher = better image quality |
| **SSIM** | Structural Similarity Index; perceptual image quality metric |
| **NIfTI** | Standard 3D medical image format (.nii.gz) |
| **TotalSegmentator** | Pre-trained network that segments 104 organs in CT |
| **nnU-Net** | Automatically configured medical segmentation framework |
| **VM-UNet** | UNet where conv blocks are replaced with VMamba VSS blocks |
| **FoundDiff** | The baseline diffusion denoiser this architecture extends |