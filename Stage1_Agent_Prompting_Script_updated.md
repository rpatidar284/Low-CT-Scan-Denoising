# Complete Agent Prompting Script
## Anatomy-Aware Low-Dose CT Denoising

> **How to use this script:**
> - Send each numbered prompt to your coding agent **one at a time**
> - After each step, run the provided **verification test** to confirm correctness
> - Only move to the next step when the current one passes
> - Each prompt is self-contained and tells the agent exactly what to build

---

## EXECUTION ORDER OVERVIEW

```
PART 0  — TotalSegmentator  (runs ONCE offline, before any training)
  STEP P0-A  Explore raw data format
  STEP P0-B  Write the segmentation pipeline script
  STEP P0-C  Run it and verify masks
  STEP P0-D  Visualise one mask (sanity check)

STAGE 1 — VM-UNet Teacher Network  (train first, then freeze)
  STEP  0   Project folder structure
  STEP  1   Config file
  STEP  2   PatchEmbed + PatchMerging
  STEP  3   SS2D scan (causal, pure-PyTorch)
  STEP  4   VSS Block
  STEP  5   VM-UNet Encoder
  STEP  6   VM-UNet Decoder + Segmentation Head
  STEP  7   Masked Average Pooling (e_a)
  STEP  8   Loss functions (L_seg, DiceLoss)
  STEP  9   BYOL module
  STEP 10   Stage1Model (combined)
  STEP 11A  Data exploration script
  STEP 11B  CTSliceDataset loader
  STEP 12   Training loop
  STEP 13   Integration tests
  STEP 14   Final checklist
```

---

# PART 0: TOTALSEGMENTATOR
## Generate Organ Pseudo-Labels (Run ONCE Before Training)

### What is this and why does it exist?

Training Stage 1 requires knowing which pixel belongs to which organ — these are
called **organ labels** or **masks**. Getting a radiologist to manually label
thousands of CT slices costs millions of dollars and months of time.

TotalSegmentator solves this automatically. It is a pre-trained neural network
(based on nnU-Net, published by Wasserthal et al. 2023) that takes any CT volume
and assigns one of 104 anatomical labels to every voxel. We run it once on all our
HDCT (clean) volumes and save the results as `.npy` files. Training then loads
these files as ground truth.

These auto-generated labels are called **pseudo-labels** ("pseudo" = approximate,
not manually verified, but accurate enough — 85-95% accuracy).

### Why run on HDCT only (not LDCT)?

Your dataset has paired scans: same patient, same body position, HDCT and LDCT.
The anatomy is physically identical in both — only the noise level differs.
TotalSegmentator gives more accurate results on clean HDCT images (noisy input →
noisier segmentation). So we run it once on HDCT and reuse the same mask for LDCT.

### Why 7 classes instead of TotalSegmentator's 104?

104 classes is overkill for denoising. What matters is distinguishing tissues with
meaningfully different noise characteristics:

| Class | Organs | Why distinct |
|-------|--------|--------------|
| 0 | Background, air | Quantum noise dominates at -1000 HU |
| 1 | Liver, Spleen | Similar HU (40-60), similar soft tissue noise |
| 2 | Kidney L + R | Distinct bright cortex + dark medulla pattern |
| 3 | Aorta, IVC, Heart | Blood vessels, motion/pulsation artifacts |
| 4 | Lung + vessels | Air-filled (-700 HU), totally different statistics |
| 5 | Vertebrae, Ribs | Bone (HU > 400), completely different noise |
| 6 | Soft tissue, Muscle | Everything else |

### The 2D/3D problem

Your data is stored as **2D slices** (individual files per CT cross-section).
TotalSegmentator needs the **full 3D volume** — a 2D slice might show a circular
dark blob, but only looking at consecutive slices in 3D reveals whether it is a
kidney, a lymph node, or a cyst.

Pipeline: collect all 2D slices → stack into 3D volume → save as NIfTI →
run TotalSegmentator → get 3D mask → split back into 2D `.npy` files.

### Label smoothing (ε = 0.1)

TotalSegmentator makes mistakes 5-15% of the time, especially at organ boundaries.
Training with hard labels (100% confident) on sometimes-wrong pseudo-labels causes
overconfidence. Instead we use label smoothing:

```
Hard label for a "liver" pixel:   [0,    1,     0,    0,    0,    0,    0   ]
Smooth label (ε=0.1, 7 classes):  [0.014, 0.914, 0.014, 0.014, 0.014, 0.014, 0.014]
Formula: smooth[k] = (1-ε) if k==true_class else ε/(num_classes-1)
```

This is applied automatically inside `SegmentationLoss` (Step 8) — you do NOT
need to pre-process the saved mask files. Save integer labels (0-6) on disk,
let the loss function handle smoothing at training time.

---

## STEP P0-A — Explore Raw Data Format

```
Write a standalone script called utils/explore_data.py.

Its only job is to inspect the raw CT files on disk and print everything
needed to write the dataset loader correctly.

DATA ROOT: /home/teaching/Music/Nigam_51/Project_51/data

Structure:
  /home/teaching/Music/Nigam_51/Project_51/data/
    C002/
      HDCT/    ← high-dose CT slices (one file per axial slice)
      LDCT/    ← low-dose CT slices (one file per axial slice)
    C004/
      HDCT/
      LDCT/
    C009/
      HDCT/
      LDCT/
    ...

The script must print ALL of the following. Use clear section headers.

─── SECTION 1: Patient inventory ───────────────────────────────────────────
List all patient folder names found (pattern: starts with 'C', contains HDCT/ and LDCT/).
Print total patient count.

─── SECTION 2: File format ──────────────────────────────────────────────────
For the first patient:
  - List all file extensions found in HDCT/ and LDCT/
  - Print count of files in each folder
  - Print first 5 and last 5 filenames (alphabetically sorted)
  - Detect format: .dcm/.IMA = DICOM, .npy = NumPy, .png/.tif = image

─── SECTION 3: Single file content ─────────────────────────────────────────
Load the FIRST file in HDCT/ for the first patient.

  If DICOM (.dcm or .IMA):
    import pydicom
    ds = pydicom.dcmread(filepath)
    arr = ds.pixel_array.astype(float)
    # Apply rescale to get Hounsfield Units:
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', -1024))
    arr = arr * slope + intercept
    # Also print DICOM metadata:
    print('RescaleSlope:', slope)
    print('RescaleIntercept:', intercept)
    print('PixelSpacing:', getattr(ds, 'PixelSpacing', 'N/A'))
    print('SliceThickness:', getattr(ds, 'SliceThickness', 'N/A'))
    print('Modality:', getattr(ds, 'Modality', 'N/A'))

  If .npy:
    arr = np.load(filepath)

  If .png/.tif:
    from PIL import Image
    arr = np.array(Image.open(filepath))

For ALL formats, print:
  shape, dtype, min, max, mean, std
  Is it already in HU range (min near -1000)? YES/NO
  Is it already normalized to [0,1] or [0,255]? YES/NO

─── SECTION 4: HDCT vs LDCT comparison ─────────────────────────────────────
Load FIRST file from HDCT/ and FIRST file from LDCT/ for the same patient.
Print side by side:
  HDCT: shape, dtype, min, max, mean, std
  LDCT: shape, dtype, min, max, mean, std
  Same shape?          YES/NO
  Filenames match?     YES/NO  (same filename exists in both HDCT and LDCT?)
  Mean absolute diff:  <value>
  Max absolute diff:   <value>
  Noise level (std of difference): <value>

─── SECTION 5: Slice count per patient ─────────────────────────────────────
For EVERY patient, print:
  Patient | HDCT count | LDCT count | Paired?
Print summary line: min/max/average slice count across all patients.

─── SECTION 6: Filename sorting pattern ────────────────────────────────────
For the first patient, print the first 5 HDCT filenames sorted two ways:
  (a) alphabetical (str sort)
  (b) numeric (extract digits, sort as integers)
Are they the same order? If not, numeric sort must be used to get correct
anatomical slice order (top-to-bottom through the body).

─── SECTION 7: File size and loading speed ──────────────────────────────────
Time how long it takes to load 10 consecutive HDCT slices:
  import time; t0=time.time(); [load 10 slices]; print(time.time()-t0, 'seconds')
This tells us whether we need caching or prefetching in the DataLoader.

Output everything to stdout. No plots. No file writes.
```

**Verify:** `python utils/explore_data.py` runs to completion and prints all 7 sections. **Paste the full output before writing Step P0-B or Step 11B.** The loader code depends entirely on what this reveals.

---

## STEP P0-B — TotalSegmentator Pipeline Script

```
Write utils/generate_masks.py — the complete pseudo-label generation pipeline.

This script runs ONCE, offline, before any training. It is NOT imported by the
training code. It is a standalone command-line tool.

────────────────────────────────────────────────────────────────────────────
DATA PATHS (hardcoded as defaults, overridable via argparse):
  Input HDCT:  /home/teaching/Music/Nigam_51/Project_51/data/{PATIENT}/HDCT/
  Output masks:/home/teaching/Music/Nigam_51/Project_51/data/masks/{PATIENT}/{IDX:04d}.npy
────────────────────────────────────────────────────────────────────────────

INSTALL REQUIREMENTS (add to requirements.txt and install before running):
  TotalSegmentator>=2.0.0
  SimpleITK>=2.2.0

────────────────────────────────────────────────────────────────────────────
IMPLEMENT THESE FUNCTIONS IN ORDER:
────────────────────────────────────────────────────────────────────────────

def load_slice_hu(filepath: str) -> np.ndarray:
    """
    Load a single CT slice and return a float32 HU array of shape [H, W].
    
    Handle all formats detected in Step P0-A.
    
    DICOM branch (.dcm or .IMA):
        import pydicom
        ds = pydicom.dcmread(filepath)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', -1024.0))
        return arr * slope + intercept    # now in Hounsfield Units

    NumPy branch (.npy):
        arr = np.load(filepath).astype(np.float32)
        # If values are already in HU range (min near -1000), return as-is.
        # If values are normalized [0,1], rescale: arr * 4000 - 1000
        return arr

    Image branch (.png / .tif / .tiff):
        from PIL import Image
        arr = np.array(Image.open(filepath)).astype(np.float32)
        # Assume 16-bit images are raw HU + 1000 offset (common convention):
        # arr = arr - 1000
        return arr
    
    Always returns float32 [H, W] with values in approximate HU range.
    """


def get_sorted_slice_paths(folder: str) -> list:
    """
    Return all slice file paths in folder, sorted in correct anatomical order.
    
    CRITICAL: Use NUMERIC sort, not alphabetical.
    Extract the integer from the filename stem and sort by it:
      key = lambda f: int(''.join(filter(str.isdigit, Path(f).stem)) or '0')
    
    Supports: .dcm, .IMA, .npy, .png, .tif, .tiff
    Returns: sorted list of absolute path strings.
    Raises ValueError if folder is empty.
    """


def stack_slices_to_volume(slice_paths: list) -> np.ndarray:
    """
    Load all slices and stack into a 3D volume.
    
    Returns: float32 array of shape [num_slices, H, W] in HU values.
    Prints progress: "  Loading slices: 50/200" every 50 slices.
    """


def volume_to_nifti(volume: np.ndarray, out_path: str,
                     spacing: tuple = (1.0, 1.0, 1.0)):
    """
    Convert a [D, H, W] float32 numpy volume to a NIfTI file.
    
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(volume)   # SimpleITK expects [D, H, W]
    img.SetSpacing(spacing)                # (x_mm, y_mm, z_mm) voxel size
    sitk.WriteImage(img, out_path)         # saves as .nii.gz
    
    spacing: use (1.0, 1.0, 1.0) if real spacing is unknown.
    If Step P0-A printed PixelSpacing and SliceThickness from DICOM,
    use those values: spacing = (PixelSpacing[0], PixelSpacing[1], SliceThickness)
    """


def run_totalsegmentator(input_nifti: str, output_nifti: str):
    """
    Run TotalSegmentator on the input NIfTI volume.
    
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(
        input=input_nifti,
        output=output_nifti,
        fast=True,        # use fast mode (slightly less accurate, much faster)
        quiet=True,       # suppress verbose output
        ml=True,          # multi-label output (single file, not per-organ files)
    )
    
    Output: a single NIfTI file where each voxel has an integer label 0-103.
    
    NOTE: First run downloads model weights (~1 GB) to ~/.totalsegmentator/
    Subsequent runs are fast (no download).
    
    If TotalSegmentator raises an error about missing weights or GPU,
    try adding: device='cpu' or device='gpu' argument.
    """


def remap_totalseg_to_7class(mask_104: np.ndarray) -> np.ndarray:
    """
    Remap TotalSegmentator's 104 labels to 7 denoising-relevant classes.
    
    Input:  [D, H, W] or [H, W] integer array, values 0-103
    Output: same shape, integer values 0-6, dtype int8
    
    MAPPING (anything not listed → class 6 soft tissue):
    
    Class 0 — background / air:
      TotalSeg label: 0
    
    Class 1 — liver / spleen:
      TotalSeg labels: 1 (spleen), 5 (liver)
    
    Class 2 — kidney:
      TotalSeg labels: 2 (kidney_right), 3 (kidney_left)
    
    Class 3 — vessels / heart:
      TotalSeg labels: 7 (aorta), 8 (inferior_vena_cava),
                       52 (pulmonary_vein),
                       53 (heart_atrium_left), 54 (heart_atrium_right),
                       55 (heart_myocardium),
                       56 (heart_ventricle_left), 57 (heart_ventricle_right)
    
    Class 4 — lung:
      TotalSeg labels: 10 (lung_upper_lobe_left),
                       11 (lung_lower_lobe_left),
                       12 (lung_upper_lobe_right),
                       13 (lung_middle_lobe_right),
                       14 (lung_lower_lobe_right)
    
    Class 5 — bone (vertebrae + ribs):
      TotalSeg labels: 26-50  (all vertebrae: cervical, thoracic, lumbar, sacrum)
                       58-81  (ribs: left_1 through right_12)
                       85 (sternum), 86 (costal_cartilages)
    
    Class 6 — soft tissue / other:
      All remaining labels (pancreas, stomach, adrenal glands, muscles, etc.)
    
    Implementation:
      mask_7 = np.full_like(mask_104, fill_value=6, dtype=np.int8)
      mask_7[mask_104 == 0] = 0
      for label in [1, 5]:                                  mask_7[mask_104==label]=1
      for label in [2, 3]:                                  mask_7[mask_104==label]=2
      for label in [7,8,52,53,54,55,56,57]:                mask_7[mask_104==label]=3
      for label in [10,11,12,13,14]:                       mask_7[mask_104==label]=4
      bone = list(range(26,51)) + list(range(58,83)) + [85,86]
      for label in bone:                                    mask_7[mask_104==label]=5
      return mask_7
    """


def save_mask_slices(mask_3d: np.ndarray, out_dir: Path):
    """
    Split a 3D mask [D, H, W] into 2D slices and save each as .npy.
    
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, slice_2d in enumerate(mask_3d):
        np.save(out_dir / f'{i:04d}.npy', slice_2d)  # int8, values 0-6
    
    Returns: number of slices saved.
    """


def generate_masks_for_patient(patient_id: str,
                                 data_root: Path,
                                 masks_root: Path,
                                 overwrite: bool = False) -> dict:
    """
    Full pipeline for one patient. Returns a status dict.
    
    Pipeline:
      1. Check if masks already exist → skip if overwrite=False
      2. hdct_dir = data_root / patient_id / 'HDCT'
         Check it exists, raise FileNotFoundError if not
      3. slice_paths = get_sorted_slice_paths(str(hdct_dir))
         Print: "  Found {N} HDCT slices"
      4. volume = stack_slices_to_volume(slice_paths)
         Print: "  Volume shape: {volume.shape}, HU range: [{min:.0f}, {max:.0f}]"
      5. tmp_in  = Path(f'/tmp/totalseg_{patient_id}_input.nii.gz')
         tmp_out = Path(f'/tmp/totalseg_{patient_id}_output.nii.gz')
      6. volume_to_nifti(volume, str(tmp_in))
      7. run_totalsegmentator(str(tmp_in), str(tmp_out))
      8. Load the output NIfTI:
            import SimpleITK as sitk
            mask_sitk = sitk.ReadImage(str(tmp_out))
            mask_3d = sitk.GetArrayFromImage(mask_sitk)  # [D, H, W] int32
         Verify shape matches input: assert mask_3d.shape == volume.shape
      9. mask_7 = remap_totalseg_to_7class(mask_3d)
         Print class distribution:
            unique, counts = np.unique(mask_7, return_counts=True)
            for cls, cnt in zip(unique, counts):
                pct = 100 * cnt / mask_7.size
                print(f"    class {cls} ({CLASS_NAMES[cls]}): {pct:.1f}%")
      10. out_dir = masks_root / patient_id
          n_saved = save_mask_slices(mask_7, out_dir)
          Print: "  Saved {n_saved} mask slices to {out_dir}"
      11. Clean up temp files:
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
      12. Return: {'patient': patient_id, 'status': 'done', 'n_slices': n_saved}
    
    Wrap step 6-11 in try/except. On error:
      Clean up temp files.
      Return: {'patient': patient_id, 'status': 'failed', 'error': str(e)}
    
    CLASS_NAMES = ['background', 'liver_spleen', 'kidney', 'vessel',
                   'lung', 'bone', 'soft_tissue']
    """


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate TotalSegmentator pseudo-labels for CT denoising training.'
    )
    parser.add_argument('--data_root',  default='/home/teaching/Music/Nigam_51/Project_51/data')
    parser.add_argument('--masks_root', default='/home/teaching/Music/Nigam_51/Project_51/data/masks')
    parser.add_argument('--patients', nargs='+', default=None,
        help='Patient IDs to process (e.g. C002 C004). Default: all.')
    parser.add_argument('--overwrite', action='store_true',
        help='Regenerate masks even if they already exist.')
    parser.add_argument('--workers', type=int, default=1,
        help='Number of patients to process in parallel (default 1 = sequential).')
    args = parser.parse_args()
    
    data_root  = Path(args.data_root)
    masks_root = Path(args.masks_root)
    
    # Discover patients
    all_patients = sorted([
        p.name for p in data_root.iterdir()
        if p.is_dir()
        and p.name[0] == 'C'
        and (p / 'HDCT').exists()
    ])
    if not all_patients:
        print(f"ERROR: No patient folders found in {data_root}")
        return
    
    to_process = args.patients if args.patients else all_patients
    invalid = [p for p in to_process if p not in all_patients]
    if invalid:
        print(f"WARNING: These patients not found and will be skipped: {invalid}")
        to_process = [p for p in to_process if p in all_patients]
    
    print(f"TotalSegmentator Pseudo-Label Generation")
    print(f"  Data root:   {data_root}")
    print(f"  Masks root:  {masks_root}")
    print(f"  Patients:    {len(to_process)} / {len(all_patients)} total")
    print(f"  Overwrite:   {args.overwrite}")
    print()
    
    import time
    results = []
    t_total = time.time()
    
    for i, patient_id in enumerate(to_process):
        print(f"[{i+1}/{len(to_process)}] {patient_id}")
        t0 = time.time()
        result = generate_masks_for_patient(
            patient_id, data_root, masks_root, args.overwrite
        )
        elapsed = time.time() - t0
        result['time_sec'] = elapsed
        results.append(result)
        status = result['status']
        if status == 'done':
            print(f"  ✓ Done in {elapsed:.0f}s")
        elif status == 'skipped':
            print(f"  → Skipped (already exists)")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'unknown')}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    done    = [r for r in results if r['status'] == 'done']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed  = [r for r in results if r['status'] == 'failed']
    total_slices = sum(r.get('n_slices', 0) for r in done)
    print(f"  Done:    {len(done)} patients, {total_slices} slices saved")
    print(f"  Skipped: {len(skipped)} (already had masks)")
    print(f"  Failed:  {len(failed)}")
    if failed:
        for r in failed:
            print(f"    {r['patient']}: {r.get('error','')}")
    print(f"  Total time: {time.time()-t_total:.0f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

**Verify:** `python utils/generate_masks.py --patients C002` runs on the machine with data. Expected output:
- Prints found slice count for C002
- Prints volume shape and HU range
- Prints class distribution (should show lung ~30-40%, bone ~5-10%, background ~20-30%)
- Creates `/home/teaching/Music/Nigam_51/Project_51/data/masks/C002/` with `.npy` files
- Prints "Done in Xs"
- Summary: "Done: 1 patients, N slices saved"

Then verify a saved mask: `python -c "import numpy as np; m = np.load('/home/teaching/Music/Nigam_51/Project_51/data/masks/C002/0000.npy'); print(m.shape, m.dtype, np.unique(m))"`
Expected: shape like `(512, 512)`, dtype `int8`, unique values subset of `[0 1 2 3 4 5 6]`

---

## STEP P0-C — Run on All Patients

```
After Step P0-B passes for C002, run on ALL patients:

  python utils/generate_masks.py

This will process C002, C004, C009, and all other patient folders found.
Expected runtime: ~20-60 seconds per patient (GPU faster, CPU slower).

After it finishes, run this verification script to confirm all masks are correct:

Write utils/verify_masks.py:

  DATA_ROOT  = '/home/teaching/Music/Nigam_51/Project_51/data'
  MASKS_ROOT = '/home/teaching/Music/Nigam_51/Project_51/data/masks'
  
  For every patient folder in DATA_ROOT:
    1. Count HDCT slices
    2. Count mask .npy files in MASKS_ROOT/{patient}/
    3. Check counts match
    4. Load 3 random mask slices (first, middle, last)
    5. For each: assert shape matches CT slice shape
                 assert dtype is int8
                 assert values are only in {0,1,2,3,4,5,6}
                 assert not all zeros (segmentation shouldn't be blank)
    6. Print: patient | hdct_slices | mask_slices | match | valid
  
  Print final summary:
    Total patients: N
    All counts matched: YES/NO
    Any blank masks found: YES/NO
    Total mask storage: X MB

Run: python utils/verify_masks.py
Expected: all rows show match=YES, valid=YES
```

**Verify:** All patients show YES/YES. If any fail, re-run `generate_masks.py --patients {PATIENT} --overwrite` for those specific patients.

---

## STEP P0-D — Visualise Masks (Sanity Check)

```
Write utils/visualise_masks.py to visually confirm the masks look correct.
This is a SANITY CHECK only — not required for training.

For a given patient and slice index, create a matplotlib figure with 3 panels:
  Panel 1: HDCT slice (grayscale, clipped to [-200, 400] HU for soft tissue window)
  Panel 2: LDCT slice (same window)
  Panel 3: Mask overlaid on HDCT (use a color-coded overlay)

Color map for the 7 classes:
  0 background  → transparent (alpha=0) or black
  1 liver/spleen → red      (#E53935)
  2 kidney       → orange   (#FB8C00)
  3 vessel/heart → blue     (#1E88E5)
  4 lung         → cyan     (#00ACC1)
  5 bone         → yellow   (#FDD835)
  6 soft tissue  → green    (#43A047)

Implementation:
  import matplotlib.pyplot as plt
  import matplotlib.colors as mcolors
  import numpy as np
  from pathlib import Path
  
  COLORS = {
      0: (0, 0, 0, 0),          # transparent
      1: (0.90, 0.22, 0.21, 0.6),
      2: (0.98, 0.55, 0.00, 0.6),
      3: (0.12, 0.53, 0.90, 0.6),
      4: (0.00, 0.67, 0.76, 0.6),
      5: (0.99, 0.85, 0.21, 0.6),
      6: (0.26, 0.63, 0.28, 0.6),
  }
  CLASS_NAMES = ['background','liver_spleen','kidney','vessel','lung','bone','soft_tissue']
  
  def visualise_slice(patient_id, slice_idx,
                       data_root='/home/teaching/Music/Nigam_51/Project_51/data',
                       masks_root='/home/teaching/Music/Nigam_51/Project_51/data/masks',
                       save_path=None):
      # Load HDCT, LDCT, mask
      # Create RGBA overlay image from mask
      # Plot 3 panels
      # If save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
      # Else: plt.show()
  
  # In __main__:
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--patient', default='C002')
  parser.add_argument('--slice',   type=int, default=100)
  parser.add_argument('--save',    default=None, help='Save to file instead of showing')
  args = parser.parse_args()
  visualise_slice(args.patient, args.slice, save_path=args.save)

Usage:
  # View interactively:
  python utils/visualise_masks.py --patient C002 --slice 100

  # Save to file (useful on headless servers):
  python utils/visualise_masks.py --patient C002 --slice 100 --save mask_check.png

Look at the output and confirm:
  - Lung regions are highlighted in cyan
  - Bone (spine/ribs) highlighted in yellow
  - Liver region highlighted in red
  - The mask aligns with the anatomy in the CT image
  - HDCT and LDCT show same anatomy (only different noise texture)
```

**Verify:** Run the script and visually inspect the overlay. The organ boundaries should roughly align with the CT anatomy. Perfect accuracy is not expected (these are pseudo-labels) but gross errors (e.g. lung labeled as bone) indicate a problem with the TotalSegmentator run.

---

## PART 0 COMPLETE — What you now have

```
/home/teaching/Music/Nigam_51/Project_51/data/
  C002/HDCT/*.{dcm|npy|...}    ← raw CT slices (unchanged)
  C002/LDCT/*.{dcm|npy|...}    ← raw CT slices (unchanged)
  C004/HDCT/...
  C004/LDCT/...
  C009/...
  masks/
    C002/
      0000.npy  ← int8 [512, 512], values 0-6
      0001.npy
      ...
    C004/
      ...
    C009/
      ...
```

These mask files are now the pseudo-labels for Stage 1 training.
The `CTSliceDataset` loader (Step 11B) reads them automatically.
Proceed to Stage 1 model implementation below.

---

# STAGE 1: VM-UNET TEACHER NETWORK

## STEP 0 — Project Setup & Folder Structure

```
Create the following folder structure for a PyTorch deep learning project called
"anatomy_ct_denoiser":

anatomy_ct_denoiser/
├── data/
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── vmamba_blocks.py       ← will contain VSS block, SS2D, VSSD
│   ├── vm_unet.py             ← will contain full VM-UNet (Stage 1)
│   ├── byol.py                ← will contain BYOL self-supervised module
│   └── stage1.py              ← will contain the combined Stage 1 model
├── losses/
│   └── stage1_losses.py
├── training/
│   └── train_stage1.py
├── utils/
│   └── masking.py
└── configs/
    └── stage1_config.yaml

Create empty __init__.py files where needed. Create a requirements.txt with:
torch>=2.0.0
torchvision
numpy
pyyaml
SimpleITK
nibabel
einops
timm

Do NOT write any model code yet. Just the folder structure and empty files.
```

**Verify:** Folder structure exists, no errors, `requirements.txt` is present.

---

## STEP 1 — Config File

```
In configs/stage1_config.yaml, write the full configuration for Stage 1 training.
Include the following parameters (all values from the architecture spec):

model:
  in_channels: 1           # CT images are grayscale
  num_classes: 7           # background, liver/spleen, kidney, vessel, lung, bone, soft tissue
  base_channels: 96        # channel count at scale 1
  depths: [2, 2, 2, 2]    # number of VSS blocks at each of 4 encoder scales
  patch_size: 4            # 4x4 patch embedding stride

training:
  batch_size: 8
  image_size: 512
  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer: adamw
  total_steps: 100000
  warmup_steps: 1000

  # Loss weights
  loss_seg_weight: 1.0
  loss_byol_weight: 0.1
  byol_start_epoch: 5

  label_smoothing: 0.1

byol:
  projector_hidden: 4096
  projector_out: 256
  predictor_hidden: 4096
  ema_tau_start: 0.996
  ema_tau_end: 1.0

data:
  num_workers: 4
  pin_memory: true

logging:
  log_every_n_steps: 100
  eval_every_n_steps: 1000
  save_every_n_steps: 5000
```

**Verify:** Load the YAML in Python with `import yaml; cfg = yaml.safe_load(open('configs/stage1_config.yaml'))` — no errors.

---

## STEP 2 — LayerNorm, Linear, and PatchEmbedding

```
In models/vmamba_blocks.py, implement these three foundational building blocks only.
Do NOT implement VSS block or SS2D yet.

1. A helper LayerNorm that works on both BCHW and BHWC tensor formats:

class LayerNorm2d(nn.Module):
    """LayerNorm for BCHW tensors (normalizes over channel dimension)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        ...

2. PatchEmbed: converts a raw CT image into a grid of patch features.

class PatchEmbed(nn.Module):
    """
    Splits the input image [B, 1, 512, 512] into non-overlapping 4x4 patches
    and projects each patch to a 96-dim feature vector.
    Uses Conv2d(in_channels=1, out_channels=96, kernel_size=4, stride=4).
    Output shape: [B, 96, 128, 128]  (BCHW format)
    """
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        ...

3. PatchMerging: downsamples spatial resolution by 2x and doubles channels.

class PatchMerging(nn.Module):
    """
    Takes 4 neighboring patches (top-left, top-right, bottom-left, bottom-right),
    concatenates them along channel dim → [B, 4C, H/2, W/2],
    then applies Linear(4C → 2C).
    
    Input:  [B, C, H, W]
    Output: [B, 2C, H/2, W/2]
    
    Both input and output are BCHW format.
    """
    def __init__(self, in_channels: int):
        ...

Include a test at the bottom of the file under if __name__ == '__main__':
    - Create a dummy input [2, 1, 512, 512]
    - Run PatchEmbed → assert output shape is [2, 96, 128, 128]
    - Run PatchMerging on [2, 96, 128, 128] → assert output shape is [2, 192, 64, 64]
    - Print "PatchEmbed and PatchMerging: PASSED"
```

**Verify:** `python models/vmamba_blocks.py` prints PASSED with no errors.

---

## STEP 3 — SS2D Selective Scan (Causal Version for Stage 1)

```
In models/vmamba_blocks.py, add the SS2D class below PatchMerging.

SS2D is the 2D Selective Scan used inside VSS blocks. It processes the image
in 4 scanning directions and aggregates the results.

Since we are implementing Stage 1 (VM-UNet Teacher), we use the CAUSAL version
(each position can only see previous positions in the scan order).

Important: A production Mamba kernel requires CUDA extensions (mamba_ssm package).
For now, implement a PURE PYTORCH version that is functionally correct but slower.
We will replace it with the fast kernel later. The interface must stay the same.

class SS2D(nn.Module):
    """
    2D Selective Scan (causal, 4-directional).
    
    Input:  x of shape [B, H, W, C]   (BHWC, channel-last)
    Output: same shape [B, H, W, C]
    
    Internally:
      1. For each of 4 directions (LR-TB, RL-BT, TB-LR, BT-RL):
         a. Flatten image to sequence of length H*W
         b. Run a selective SSM scan (causal)
         c. Reshape back to [B, H, W, C]
      2. Sum all 4 direction outputs
      3. Return the summed result
    
    The SSM parameters (A, B, C, D, delta) are input-dependent (selective):
      - delta (log-space dt): Linear(C → C), softplus activation
      - B: Linear(C → d_state)
      - C: Linear(C → d_state)
      - A: fixed negative parameter (log-space), shape [C, d_state]
      - D: scalar skip connection per channel
    
    Use d_state=16 as the state dimension.
    
    The core recurrence for the causal scan:
      h[t] = exp(delta[t] * A) * h[t-1] + delta[t] * B[t] * x[t]
      y[t] = C[t] @ h[t] + D * x[t]
    
    (This is the discretized ZOH version of the SSM.)
    
    For the 4 directions: apply scan to the flattened sequence of the image,
    where flattening order differs per direction:
      dir 0: rows then cols (normal reading order)
      dir 1: reverse of dir 0
      dir 2: cols then rows (transpose)
      dir 3: reverse of dir 2
    
    All 4 directions share the same SSM parameters.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        ...

Add to the __main__ test block:
    - Create dummy [2, 128, 128, 96] BHWC tensor
    - Run SS2D(d_model=96)
    - Assert output shape is [2, 128, 128, 96]
    - Print "SS2D: PASSED"
```

**Verify:** `python models/vmamba_blocks.py` still prints both PASSED lines.

---

## STEP 4 — VSS Block (Full VMamba Building Block)

```
In models/vmamba_blocks.py, add the VSSBlock class.

This is the complete VMamba visual state-space block. Every VSS block in the
encoder and decoder of Stage 1 follows this exact sequence of operations:

class VSSBlock(nn.Module):
    """
    VSS Block — the building block of VM-UNet.
    
    Input:  x of shape [B, H, W, C]   (BHWC format, channel-last)
    Output: same shape [B, H, W, C]
    
    Internal sequence:
      1. LayerNorm(x)                             → [B, H, W, C]
      2. Linear(C → 2C)                           → [B, H, W, 2C]
         Split into x_main [B,H,W,C] and x_gate [B,H,W,C]
      3. DepthwiseConv2d(C, C, kernel=3, pad=1) on x_main
         NOTE: depthwise conv needs BCHW, so permute before and after
      4. SS2D scan on x_main                      → [B, H, W, C]
      5. Gating: x_main = x_main * SiLU(x_gate)  → [B, H, W, C]
      6. Linear(C → C) on x_main                 → [B, H, W, C]
      7. Residual: output = x + x_main            → [B, H, W, C]
    
    Parameters:
      d_model (int): channel dimension C
      d_state (int): SSM state dimension, default 16
      drop_path (float): stochastic depth rate, default 0.0
    
    Use drop_path from timm: from timm.models.layers import DropPath
    If drop_path=0, just use identity (no drop).
    """
    def __init__(self, d_model: int, d_state: int = 16, drop_path: float = 0.0):
        ...

Add to the __main__ test block:
    - Create dummy [2, 128, 128, 96] BHWC tensor
    - Run VSSBlock(d_model=96)
    - Assert output shape is [2, 128, 128, 96]
    - Print "VSSBlock: PASSED"
```

**Verify:** `python models/vmamba_blocks.py` prints all 3 PASSED lines.

---

## STEP 5 — VM-UNet Encoder

```
In models/vm_unet.py, implement the Encoder portion of VM-UNet only.

The encoder has 4 scales. Each scale applies VSS blocks, then PatchMerging to
halve the resolution. The encoder outputs skip connections at each scale.

from models.vmamba_blocks import PatchEmbed, PatchMerging, VSSBlock

class VMUNetEncoder(nn.Module):
    """
    VM-UNet Encoder.
    
    Input:  [B, 1, 512, 512]  (BCHW)
    
    Architecture:
      PatchEmbed(in_channels=1, embed_dim=96, patch_size=4)
        → [B, 96, 128, 128]
      
      Scale 1: 2x VSSBlock(d_model=96) + save skip  → [B, 96, 128, 128]
                PatchMerging(96)                      → [B, 192, 64, 64]
      
      Scale 2: 2x VSSBlock(d_model=192) + save skip → [B, 192, 64, 64]
                PatchMerging(192)                     → [B, 384, 32, 32]
      
      Scale 3: 2x VSSBlock(d_model=384) + save skip → [B, 384, 32, 32]
                PatchMerging(384)                     → [B, 768, 16, 16]
      
      Bottleneck: 2x VSSBlock(d_model=768)           → [B, 768, 16, 16]
    
    IMPORTANT: VSSBlock works in BHWC format. After PatchEmbed (which outputs BCHW),
    convert: x = x.permute(0, 2, 3, 1) before VSSBlocks.
    After VSSBlocks at each scale, convert back to BCHW for PatchMerging.
    
    Returns:
      bottleneck: [B, 768, 16, 16]  (BCHW)
      skips: list of 3 tensors in order [skip1, skip2, skip3]
             skip1: [B, 96, 128, 128]
             skip2: [B, 192, 64, 64]
             skip3: [B, 384, 32, 32]
             (all BCHW format)
    
    NOTE: The bottleneck output (768 channels) is also called 'F' in the 
    architecture — it will be used for BYOL training.
    
    Parameters:
      in_channels: int = 1
      embed_dim: int = 96
      depths: list = [2, 2, 2, 2]   ← number of VSS blocks per scale (incl. bottleneck)
      patch_size: int = 4
    """
    def __init__(self, in_channels=1, embed_dim=96, depths=[2,2,2,2], patch_size=4):
        ...

Add a test in if __name__ == '__main__':
    enc = VMUNetEncoder()
    x = torch.randn(2, 1, 512, 512)
    bottleneck, skips = enc(x)
    assert bottleneck.shape == (2, 768, 16, 16), f"bottleneck wrong: {bottleneck.shape}"
    assert skips[0].shape == (2, 96, 128, 128)
    assert skips[1].shape == (2, 192, 64, 64)
    assert skips[2].shape == (2, 384, 32, 32)
    print("VMUNetEncoder: PASSED")
```

**Verify:** `python models/vm_unet.py` prints PASSED.

---

## STEP 6 — VM-UNet Decoder + Segmentation Head

```
In models/vm_unet.py, add the Decoder and the full VMUNet class.

class VMUNetDecoder(nn.Module):
    """
    VM-UNet Decoder with skip connections.
    
    At each scale: upsample 2x, concatenate skip, run VSSBlocks.
    
    Scale 3 up: bilinear 2x upsample + Linear(768→384)
                concat with skip3: [B, 384+384, 32, 32] = [B, 768, 32, 32]
                2x VSSBlock(d_model=768) → [B, 768, 32, 32]
                Linear(768→384) to reduce channels → [B, 384, 32, 32]
    
    Scale 2 up: bilinear 2x upsample + Linear(384→192)
                concat with skip2: [B, 192+192, 64, 64] = [B, 384, 64, 64]
                2x VSSBlock(d_model=384) → [B, 384, 64, 64]
                Linear(384→192) → [B, 192, 64, 64]
    
    Scale 1 up: bilinear 2x upsample + Linear(192→96)
                concat with skip1: [B, 96+96, 128, 128] = [B, 192, 128, 128]
                2x VSSBlock(d_model=192) → [B, 192, 128, 128]
                Linear(192→96) → [B, 96, 128, 128]
    
    Final: bilinear 4x upsample → [B, 96, 512, 512]
    
    IMPORTANT:
    - "bilinear Nx + Linear(C_in→C_out)" means:
        F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        then a 1x1 Conv2d (or Linear applied channel-wise) to change channels
    - VSSBlocks expect BHWC; decoder features are BCHW; permute as in encoder
    - The channel-reduction Linear after concat+VSSBlocks is a 1x1 conv:
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    Input: bottleneck [B, 768, 16, 16], skips [skip1, skip2, skip3] (all BCHW)
    Output: features [B, 96, 512, 512]  (BCHW)
    
    Also returns: decoder_features [B, 96, 512, 512] just before the final upsample.
    This is used for computing e_a (anatomy embeddings) in masked average pooling.
    """
    def __init__(self, embed_dim=96, depths=[2,2,2]):
        ...


class VMUNet(nn.Module):
    """
    Complete VM-UNet for Stage 1 segmentation.
    
    Combines Encoder + Decoder + Segmentation Head.
    
    Segmentation head:
      Conv2d(96, num_classes=7, kernel_size=1)  → logits [B, 7, 512, 512]
      Softmax(dim=1)                             → S [B, 7, 512, 512]
    
    Returns a dict with:
      'logits': [B, 7, 512, 512]  ← raw scores (used for loss)
      'S':      [B, 7, 512, 512]  ← softmax probabilities
      'F':      [B, 768, 16, 16]  ← bottleneck features (for BYOL)
      'decoder_features': [B, 96, 512, 512]  ← for computing e_a
    """
    def __init__(self, in_channels=1, num_classes=7, embed_dim=96,
                 depths=[2,2,2,2], patch_size=4):
        ...

Add test:
    model = VMUNet()
    x = torch.randn(2, 1, 512, 512)
    out = model(x)
    assert out['S'].shape == (2, 7, 512, 512)
    assert out['logits'].shape == (2, 7, 512, 512)
    assert out['F'].shape == (2, 768, 16, 16)
    assert out['decoder_features'].shape == (2, 96, 512, 512)
    # Check S sums to 1 along class dimension
    assert torch.allclose(out['S'].sum(dim=1), torch.ones(2, 512, 512), atol=1e-5)
    print("VMUNet: PASSED")
```

**Verify:** `python models/vm_unet.py` prints PASSED.

---

## STEP 7 — Masked Average Pooling (Computing e_a)

```
In utils/masking.py, implement the masked average pooling function that
computes anatomy embeddings e_a from decoder features and segmentation map S.

def masked_average_pooling(decoder_features: torch.Tensor,
                            S: torch.Tensor) -> torch.Tensor:
    """
    Compute per-organ feature embeddings using S as soft attention weights.
    
    For each organ class k:
      1. weight = S[:, k, :, :]                    → [B, H, W]
      2. weighted_features = weight.unsqueeze(1) * decoder_features  → [B, C, H, W]
      3. summed = weighted_features.sum(dim=[2, 3])                   → [B, C]
      4. total_weight = weight.sum(dim=[1, 2]).unsqueeze(1) + 1e-8   → [B, 1]
      5. e_a_k = summed / total_weight                                → [B, C]
    
    Stack over all classes:
      e_a = torch.stack([e_a_0, e_a_1, ..., e_a_6], dim=1)           → [B, 7, C]
    
    Args:
      decoder_features: [B, C, H, W]   BCHW format
      S:                [B, num_classes, H, W]  soft segmentation probabilities
    
    Returns:
      e_a: [B, num_classes, C]
    
    Physical meaning:
      e_a[b, k, :] = the average feature vector of organ k in image b,
                     weighted by how confident the model is that each pixel
                     belongs to organ k.
    """
    ...

Also add a helper to compute e_a from a VMUNet output dict:

def compute_anatomy_embeddings(vm_unet_output: dict) -> torch.Tensor:
    """
    Convenience wrapper.
    Extracts decoder_features and S from vm_unet output,
    returns e_a [B, 7, C].
    """
    return masked_average_pooling(
        vm_unet_output['decoder_features'],
        vm_unet_output['S']
    )

Add test in __main__:
    B, C, H, W = 2, 96, 512, 512
    num_classes = 7
    features = torch.randn(B, C, H, W)
    # Create a fake S where all pixels belong to class 1
    S = torch.zeros(B, num_classes, H, W)
    S[:, 1, :, :] = 1.0
    e_a = masked_average_pooling(features, S)
    assert e_a.shape == (B, num_classes, C)
    # e_a[:, 1, :] should equal the global average of features
    expected = features.mean(dim=[2, 3])  # [B, C]
    assert torch.allclose(e_a[:, 1, :], expected, atol=1e-5)
    print("masked_average_pooling: PASSED")
```

**Verify:** `python utils/masking.py` prints PASSED.

---

## STEP 8 — Stage 1 Loss Functions (L_seg)

```
In losses/stage1_losses.py, implement the segmentation loss for Stage 1.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for organ segmentation.
    
    Wraps F.cross_entropy with label_smoothing parameter.
    
    Args:
      num_classes: int = 7
      label_smoothing: float = 0.1
        Formula: smooth[k] = (1 - eps) if k == true_class else eps / (num_classes - 1)
        This prevents the model from being overconfident about noisy pseudo-labels.
      weight: optional [num_classes] tensor for class weighting
    
    Forward:
      logits: [B, 7, H, W]   ← raw scores from segmentation head (NOT softmax output)
      targets: [B, H, W]     ← integer class labels 0-6
    
    Returns:
      scalar loss value
    
    NOTE: Use logits (not S) for the loss. Cross-entropy applies softmax internally.
    """
    def __init__(self, num_classes=7, label_smoothing=0.1, weight=None):
        ...
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...


class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation — used as an optional auxiliary to cross-entropy.
    
    Dice(pred, target) = 1 - (2 * sum(pred * target) + smooth) /
                                (sum(pred) + sum(target) + smooth)
    
    Computed per class, then averaged.
    
    Args:
      smooth: float = 1.0   ← Laplace smoothing to avoid division by zero
    
    Forward:
      probs:   [B, 7, H, W]   ← softmax probabilities (S, not logits)
      targets: [B, H, W]      ← integer class labels 0-6
    
    Returns:
      scalar loss value (mean over classes)
    """
    def __init__(self, smooth=1.0):
        ...


def compute_dice_per_class(probs: torch.Tensor,
                            targets: torch.Tensor,
                            num_classes: int = 7) -> dict:
    """
    Compute Dice score per organ class. Used for evaluation/logging only.
    
    Args:
      probs:   [B, 7, H, W]  softmax output
      targets: [B, H, W]     integer ground truth labels
    
    Returns:
      dict: {'class_0': float, 'class_1': float, ..., 'mean': float}
      
    Class names for logging:
      0: background
      1: liver_spleen
      2: kidney
      3: vessel
      4: lung
      5: bone
      6: soft_tissue
    """
    CLASS_NAMES = ['background', 'liver_spleen', 'kidney', 'vessel',
                   'lung', 'bone', 'soft_tissue']
    ...

Add tests in __main__:
    B, C, H, W = 2, 7, 64, 64  # use 64x64 to keep test fast
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    
    seg_loss = SegmentationLoss()
    loss_val = seg_loss(logits, targets)
    assert loss_val.shape == ()  # scalar
    assert loss_val > 0
    print(f"SegmentationLoss: PASSED (loss={loss_val:.4f})")
    
    probs = F.softmax(logits, dim=1)
    dice_loss = DiceLoss()
    dice_val = dice_loss(probs, targets)
    assert 0 <= dice_val <= 1
    print(f"DiceLoss: PASSED (loss={dice_val:.4f})")
    
    per_class = compute_dice_per_class(probs, targets)
    assert 'mean' in per_class
    assert 'liver_spleen' in per_class
    print(f"compute_dice_per_class: PASSED (mean dice={per_class['mean']:.4f})")
```

**Verify:** `python losses/stage1_losses.py` prints all 3 PASSED lines.

---

## STEP 9 — BYOL Module

```
In models/byol.py, implement the complete BYOL self-supervised module
that makes Stage 1 features invariant to CT noise level.

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class ProjectorMLP(nn.Module):
    """
    2-layer MLP that maps bottleneck features to a compact vector.
    
    Input:  F [B, 768, 16, 16]
    Step 1: Global average pool → [B, 768]
    Step 2: Linear(768 → 4096) → BatchNorm → ReLU → Linear(4096 → 256)
    Output: z [B, 256]
    """
    def __init__(self, in_dim=768, hidden_dim=4096, out_dim=256):
        ...


class PredictorMLP(nn.Module):
    """
    2-layer MLP used ONLY in the online network.
    Predicts the target network's projection.
    
    Input:  z [B, 256]
    Linear(256 → 4096) → BatchNorm → ReLU → Linear(4096 → 256)
    Output: q [B, 256]
    """
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256):
        ...


class BYOLModule(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) for noise-invariant anatomy features.
    
    Architecture:
      Online network:  encoder (shared VMUNet) + projector + predictor
      Target network:  slow EMA copy of encoder + projector (NO predictor)
    
    The encoder is the VMUNet itself — we use its bottleneck features F.
    We do NOT maintain a separate copy of the full encoder here.
    The BYOLModule only owns the projector and predictor heads.
    The caller passes in F_online and F_target separately.
    
    Why this design? The VMUNet is already large. We don't want to store
    a full second copy. Instead:
      - The online projector/predictor are updated by gradients normally
      - The target projector is an EMA copy of the online projector
      - The caller maintains EMA for the full encoder backbone separately
    
    Args:
      feature_dim: int = 768     ← bottleneck channel count
      projector_hidden: int = 4096
      projector_out: int = 256
    
    Methods:
      forward(F_online, F_target) → loss scalar
        F_online: [B, 768, 16, 16]  gradient-connected
        F_target: [B, 768, 16, 16]  detached (from EMA encoder)
        Returns L_byol scalar
      
      update_target_projector(tau: float)
        EMA update: target_proj = tau * target_proj + (1-tau) * online_proj
    
    Loss formula:
      z_online = online_projector(F_online)          [B, 256]
      q_online = predictor(z_online)                 [B, 256]
      z_target = target_projector(F_target).detach() [B, 256]
      
      # Normalize both to unit sphere before cosine similarity
      q_norm = F.normalize(q_online, dim=-1)
      z_norm = F.normalize(z_target, dim=-1)
      
      L = 2 - 2 * (q_norm * z_norm).sum(dim=-1).mean()
      
      # Run both directions (view1→view2 and view2→view1)
      # The caller should call forward twice (once per direction) and sum
    
    NOTE on initialization:
      The last Linear in the predictor MLP should be initialized with
      weight=zeros and bias=zeros to implement "zero initialization."
      This makes the predictor output zero at init → stable training start.
    """
    def __init__(self, feature_dim=768, projector_hidden=4096, projector_out=256):
        super().__init__()
        self.online_projector = ProjectorMLP(feature_dim, projector_hidden, projector_out)
        self.target_projector = deepcopy(self.online_projector)
        self.predictor = PredictorMLP(projector_out, projector_hidden, projector_out)
        
        # Initialize target projector — no gradients
        for p in self.target_projector.parameters():
            p.requires_grad_(False)
        
        # Zero-init last predictor layer
        nn.init.zeros_(self.predictor.net[-1].weight)
        nn.init.zeros_(self.predictor.net[-1].bias)
    
    def forward(self, F_online: torch.Tensor, F_target: torch.Tensor) -> torch.Tensor:
        ...
    
    @torch.no_grad()
    def update_target_projector(self, tau: float):
        """EMA update of target projector from online projector."""
        for p_online, p_target in zip(self.online_projector.parameters(),
                                       self.target_projector.parameters()):
            p_target.data = tau * p_target.data + (1 - tau) * p_online.data


def get_ema_tau(current_step: int, total_steps: int,
                tau_start: float = 0.996, tau_end: float = 1.0) -> float:
    """
    Linearly interpolate EMA decay from tau_start to tau_end.
    
    tau = tau_start + (tau_end - tau_start) * (current_step / total_steps)
    """
    ...

Add test in __main__:
    byol = BYOLModule(feature_dim=768)
    F_online = torch.randn(2, 768, 16, 16, requires_grad=True)
    F_target = torch.randn(2, 768, 16, 16)
    loss = byol(F_online, F_target)
    assert loss.shape == ()
    assert 0 <= loss.item() <= 4.0, f"BYOL loss out of range: {loss.item()}"
    loss.backward()
    assert F_online.grad is not None
    print(f"BYOLModule forward+backward: PASSED (loss={loss.item():.4f})")
    
    tau = get_ema_tau(500, 100000)
    byol.update_target_projector(tau)
    print(f"EMA update: PASSED (tau={tau:.6f})")
```

**Verify:** `python models/byol.py` prints both PASSED lines.

---

## STEP 10 — Stage 1 Combined Model

```
In models/stage1.py, assemble the complete Stage 1 model that combines
VMUNet + BYOL + anatomy embedding computation.

from models.vm_unet import VMUNet
from models.byol import BYOLModule
from utils.masking import compute_anatomy_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Stage1Model(nn.Module):
    """
    Complete Stage 1: VM-UNet Teacher Network.
    
    Combines:
      - VMUNet backbone (encoder + decoder + segmentation head)
      - BYOL module for noise-invariant feature learning
      - Masked average pooling to compute e_a
    
    Two modes of operation:
    
    1. Training mode (forward with x and optional byol_view2):
       Returns everything needed for loss computation.
    
    2. Inference mode (called from Stage 2 training):
       Returns S and e_a only. Much faster.
    
    Args:
      in_channels: int = 1
      num_classes: int = 7
      embed_dim: int = 96
      depths: list = [2, 2, 2, 2]
      byol_feature_dim: int = 768
    
    Forward(x, byol_view2=None, return_byol=False):
      x: [B, 1, 512, 512]  ← CT image (NDCT or LDCT)
      byol_view2: [B, 1, 512, 512] or None  ← second augmented view for BYOL
      return_byol: bool  ← whether to compute BYOL loss
      
      Returns dict:
        'logits':           [B, 7, 512, 512]   always
        'S':                [B, 7, 512, 512]   always
        'e_a':              [B, 7, 96]         always
        'F':                [B, 768, 16, 16]   always
        'byol_loss':        scalar or None     only if return_byol=True and byol_view2 is given
        'decoder_features': [B, 96, 512, 512]  always
    
    Noise augmentation for BYOL views:
      When byol_view2 is None but return_byol=True, generate it internally:
        view1 = x + 0.02 * randn_like(x)   ← light noise (NDCT-like)
        view2 = x + 0.15 * randn_like(x)   ← heavy noise (LDCT-like)
      The main forward pass uses the original x (not the augmented views).
    
    IMPORTANT: The BYOL EMA update (update_target_projector) is NOT called here.
    It is called by the training loop after each optimizer step.
    """
    def __init__(self, in_channels=1, num_classes=7, embed_dim=96,
                 depths=[2,2,2,2], byol_feature_dim=768):
        super().__init__()
        self.backbone = VMUNet(in_channels=in_channels,
                                num_classes=num_classes,
                                embed_dim=embed_dim,
                                depths=depths)
        self.byol = BYOLModule(feature_dim=byol_feature_dim)
    
    def forward(self, x, byol_view2=None, return_byol=False):
        ...
    
    def get_anatomy_conditioning(self, x: torch.Tensor):
        """
        Inference-only method. Returns only S and e_a.
        Used by Stage 2 training loop with torch.no_grad().
        
        Returns:
          S:   [B, 7, 512, 512]
          e_a: [B, 7, 96]
        """
        out = self.forward(x, return_byol=False)
        return out['S'], out['e_a']


def load_stage1_frozen(checkpoint_path: str, device='cuda') -> Stage1Model:
    """
    Load a trained Stage 1 model and freeze ALL parameters.
    Used when setting up Stage 2 training.
    
    Args:
      checkpoint_path: path to .pth file saved during Stage 1 training
      device: 'cuda' or 'cpu'
    
    Returns:
      model: Stage1Model with requires_grad=False for all parameters
    """
    model = Stage1Model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)

Add test in __main__:
    model = Stage1Model()
    x = torch.randn(2, 1, 512, 512)
    
    # Test inference mode
    out = model(x, return_byol=False)
    assert out['S'].shape == (2, 7, 512, 512)
    assert out['e_a'].shape == (2, 7, 96)
    assert out['F'].shape == (2, 768, 16, 16)
    assert torch.allclose(out['S'].sum(dim=1), torch.ones(2, 512, 512), atol=1e-5)
    print("Stage1Model inference mode: PASSED")
    
    # Test BYOL mode
    out_byol = model(x, return_byol=True)
    assert out_byol['byol_loss'] is not None
    assert 0 <= out_byol['byol_loss'].item() <= 4.0
    print(f"Stage1Model BYOL mode: PASSED (byol_loss={out_byol['byol_loss'].item():.4f})")
    
    # Test anatomy conditioning
    S, e_a = model.get_anatomy_conditioning(x)
    assert S.shape == (2, 7, 512, 512)
    assert e_a.shape == (2, 7, 96)
    print("Stage1Model.get_anatomy_conditioning: PASSED")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Stage 1 parameters: {total_params/1e6:.1f}M")
```

**Verify:** `python models/stage1.py` prints all 3 PASSED lines and parameter count.

---

## STEP 11A — Data Exploration (Run BEFORE writing the loader)

```
Before writing any dataset code, run this exploration script to understand
the exact format of the CT data on disk.

Write a standalone script called data/explore_data.py that does the following:

DATA ROOT: /home/teaching/Music/Nigam_51/Project_51/data

Structure to explore:
  data_root/
    C002/
      HDCT/    ← high-dose CT (this is our NDCT — clean reference)
      LDCT/    ← low-dose CT (this is our LDCT — noisy input)
    C004/
      HDCT/
      LDCT/
    C009/
      HDCT/
      LDCT/
    ...        ← more patient folders following the same CXX naming pattern

The script should print ALL of the following information:

=== 1. Patient list ===
List every patient folder found (C002, C004, C009, ...).
Count total number of patients.

=== 2. File format detection ===
For the FIRST patient found, list ALL files inside HDCT/ and LDCT/:
  - Print the file extensions found (e.g. .dcm, .npy, .png, .IMA, .tiff)
  - Print the total count of files in each subfolder
  - Print the first 5 filenames sorted alphabetically

=== 3. File content inspection ===
Load the FIRST file found in HDCT/ of the first patient.
Detect format automatically:
  - If extension is .dcm or .IMA: use pydicom.dcmread()
  - If extension is .npy: use np.load()
  - If extension is .png or .tiff: use PIL or cv2
  - If extension is unknown: try numpy first, then pydicom

Print:
  - Loaded array shape (e.g. (512, 512) or (512, 512, 3))
  - Data type (float32, int16, uint16, uint8...)
  - Min value, Max value, Mean value
  - For DICOM: also print RescaleIntercept, RescaleSlope, PixelSpacing if available

=== 4. HDCT vs LDCT comparison ===
Load the FIRST file from HDCT/ and the FIRST file from LDCT/ for the same patient.
Print side-by-side:
  HDCT: shape=..., dtype=..., min=..., max=..., mean=..., std=...
  LDCT: shape=..., dtype=..., min=..., max=..., mean=..., std=...
  Are they the same shape? YES/NO
  Are filenames paired (same names in both folders)? YES/NO
  Difference stats: mean_abs_diff=..., max_abs_diff=...

=== 5. Slice count ===
For each patient, print:
  Patient | HDCT slices | LDCT slices | Match?
Print a summary: min/max/average slice count across all patients.

=== 6. Filename pattern ===
Print the sorted filenames of the first 5 slices in HDCT/ for the first patient.
Try to detect the naming pattern:
  - Are they numbered? (0001.dcm, 0002.dcm...)
  - Do they have a prefix? (IM-0001-0001.dcm...)
  - What is the zero-padding width?

Run this script first. Paste the output back so the dataset loader
in Step 11B can be written to exactly match your data format.

Requirements: pip install pydicom Pillow numpy (most are already installed)
```

**Verify:** `python data/explore_data.py` runs without crash and prints all 6 sections. **Paste the output here before proceeding to Step 11B.**

---

## STEP 11B — Dataset Loader (Written AFTER seeing Step 11A output)

```
IMPORTANT: Before writing this, you must have the output from Step 11A.
The loader below uses placeholders — replace [FORMAT], [SHAPE], [HU_RANGE]
with what you actually found.

In data/dataset.py, implement the CT dataset loader for Stage 1 training.

DATA ROOT: /home/teaching/Music/Nigam_51/Project_51/data

ACTUAL directory structure (your real data):
  /home/teaching/Music/Nigam_51/Project_51/data/
    C002/
      HDCT/    ← high-dose slices  (NDCT = clean reference = what model learns to produce)
      LDCT/    ← low-dose slices   (LDCT = noisy input = what model receives)
    C004/
      HDCT/
      LDCT/
    C009/
      HDCT/
      LDCT/
    ... (more CXX patient folders)

NOTE ON NAMING CONVENTION IN CODE:
  The architecture uses "NDCT" for clean/high-dose and "LDCT" for noisy/low-dose.
  Your data uses "HDCT" for high-dose. Map them as:
    HDCT folder → ndct in code (clean reference)
    LDCT folder → ldct in code (noisy input)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random


def load_slice(filepath: str) -> np.ndarray:
    """
    Load a single CT slice from disk and return as float32 numpy array.
    
    Handle the file format found in Step 11A. Implement the correct branch:
    
    Branch A — if files are DICOM (.dcm or .IMA):
        import pydicom
        ds = pydicom.dcmread(filepath)
        arr = ds.pixel_array.astype(np.float32)
        # Apply DICOM rescale: HU = pixel * RescaleSlope + RescaleIntercept
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', -1024))
        arr = arr * slope + intercept
        return arr  # now in Hounsfield Units
    
    Branch B — if files are .npy:
        arr = np.load(filepath).astype(np.float32)
        return arr
    
    Branch C — if files are .png or .tiff:
        from PIL import Image
        arr = np.array(Image.open(filepath)).astype(np.float32)
        return arr
    
    Choose the correct branch based on your Step 11A output.
    The function always returns a 2D float32 array of shape [H, W].
    """
    ...


def get_sorted_slice_files(folder: str) -> List[str]:
    """
    Return a sorted list of all slice file paths in a folder.
    Sorting must be by slice index (numeric order, not lexicographic).
    
    Example: ['0001.dcm', '0002.dcm', ..., '0200.dcm'] sorted correctly.
    
    If filenames contain numbers, extract the numeric part for sorting:
      sorted(files, key=lambda f: int(''.join(filter(str.isdigit, Path(f).stem))))
    
    Returns: list of absolute file paths, sorted by slice order.
    """
    ...


class CTSliceDataset(Dataset):
    """
    Dataset for paired HDCT/LDCT slices.
    Maps HDCT → 'ndct' (clean reference) and LDCT → 'ldct' (noisy input).
    
    REAL DATA STRUCTURE:
      /home/teaching/Music/Nigam_51/Project_51/data/
        C002/HDCT/*.dcm (or .npy, etc.)
        C002/LDCT/*.dcm
        C004/HDCT/...
        ...
    
    Masks are NOT present in the raw data. They are generated separately by
    TotalSegmentator (see TotalSegmentator pipeline step). Until masks exist,
    this dataset returns dummy zero masks so training can be tested.
    
    MASKS DIRECTORY (created after TotalSegmentator runs):
      /home/teaching/Music/Nigam_51/Project_51/data/masks/
        C002/
          0000.npy   ← integer [H, W] array with values 0-6
          0001.npy
          ...
        C004/
          ...
    
    Each __getitem__ returns:
      'ndct':       [1, H, W] float32  ← normalized, from HDCT folder
      'ldct':       [1, H, W] float32  ← normalized, from LDCT folder
      'mask':       [H, W]    int64    ← organ labels 0-6 (zeros if not yet generated)
      'patient_id': str                ← e.g. 'C002'
      'slice_idx':  int                ← slice index (0-based)
      'hdct_path':  str                ← full path to HDCT file (useful for debugging)
      'ldct_path':  str                ← full path to LDCT file
    
    PREPROCESSING:
      Clip to [-1000, 3000] HU, then normalize to [0, 1]:
        x_norm = (clip(x, -1000, 3000) - (-1000)) / (3000 - (-1000))
      
      NOTE: If Step 11A showed your data is already normalized (values in [0,1]
      or [0, 255]), adjust the clipping/normalization accordingly.
      Always use the actual value range found in Step 11A.
    
    PAIRING ASSUMPTION:
      The i-th file (sorted) in HDCT/ corresponds to the i-th file in LDCT/.
      This is standard for this dataset format. If Step 11A showed filenames
      are NOT the same across HDCT and LDCT, use filename matching instead.
    
    SPLIT: Patient-level split (not slice-level) to prevent data leakage.
      Given patients [C002, C004, C009, C010, ...]:
        train = first 80% of patients
        val   = next 10%
        test  = last 10%
      Sorted alphabetically before splitting for reproducibility.
    
    Args:
      data_root: str = '/home/teaching/Music/Nigam_51/Project_51/data'
      masks_root: str = '/home/teaching/Music/Nigam_51/Project_51/data/masks'
        (set to None or a non-existent path to use dummy zero masks)
      split: 'train', 'val', or 'test'
      split_ratio: (0.8, 0.1, 0.1)
      seed: int = 42
      augment: bool = True  ← only applies during 'train' split
    
    AUGMENTATION (training only, same transform applied to ndct + ldct + mask):
      - Random horizontal flip (p=0.5): torch.flip(x, dims=[-1])
      - Random vertical flip (p=0.5):   torch.flip(x, dims=[-2])
      Use the same random seed per sample so ndct/ldct/mask flip identically.
    """
    
    DATA_ROOT = '/home/teaching/Music/Nigam_51/Project_51/data'
    MASKS_ROOT = '/home/teaching/Music/Nigam_51/Project_51/data/masks'
    HU_MIN = -1000.0
    HU_MAX = 3000.0
    CLASS_NAMES = ['background', 'liver_spleen', 'kidney', 'vessel',
                   'lung', 'bone', 'soft_tissue']
    
    def __init__(self,
                 data_root: str = DATA_ROOT,
                 masks_root: str = MASKS_ROOT,
                 split: str = 'train',
                 split_ratio: Tuple = (0.8, 0.1, 0.1),
                 seed: int = 42,
                 augment: bool = True):
        self.data_root = Path(data_root)
        self.masks_root = Path(masks_root) if masks_root else None
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Find all patient folders (pattern: C followed by digits)
        all_patients = sorted([
            p.name for p in self.data_root.iterdir()
            if p.is_dir() and p.name.startswith('C')
            and (p / 'HDCT').exists() and (p / 'LDCT').exists()
        ])
        
        assert len(all_patients) > 0, f"No patient folders found in {data_root}"
        
        # Patient-level split
        random.seed(seed)
        n = len(all_patients)
        n_train = int(n * split_ratio[0])
        n_val   = int(n * split_ratio[1])
        if split == 'train':
            self.patients = all_patients[:n_train]
        elif split == 'val':
            self.patients = all_patients[n_train:n_train + n_val]
        else:  # test
            self.patients = all_patients[n_train + n_val:]
        
        # Build flat index: list of (patient_id, slice_idx, hdct_path, ldct_path)
        self.samples = []
        for patient in self.patients:
            hdct_files = get_sorted_slice_files(str(self.data_root / patient / 'HDCT'))
            ldct_files = get_sorted_slice_files(str(self.data_root / patient / 'LDCT'))
            assert len(hdct_files) == len(ldct_files), (
                f"Patient {patient}: HDCT has {len(hdct_files)} slices but "
                f"LDCT has {len(ldct_files)} slices — mismatch!"
            )
            for idx, (h, l) in enumerate(zip(hdct_files, ldct_files)):
                self.samples.append((patient, idx, h, l))
        
        print(f"[CTSliceDataset] split={split}, patients={len(self.patients)}, "
              f"slices={len(self.samples)}")
    
    def _normalize(self, arr: np.ndarray) -> torch.Tensor:
        """Clip to [HU_MIN, HU_MAX] and normalize to [0, 1]. Returns [1, H, W]."""
        arr = np.clip(arr, self.HU_MIN, self.HU_MAX)
        arr = (arr - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        return torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]
    
    def _load_mask(self, patient_id: str, slice_idx: int,
                   h: int, w: int) -> torch.Tensor:
        """
        Load organ mask if it exists, otherwise return zeros.
        Mask shape: [H, W] int64 with values 0-6.
        """
        if self.masks_root and self.masks_root.exists():
            mask_path = self.masks_root / patient_id / f'{slice_idx:04d}.npy'
            if mask_path.exists():
                mask = np.load(str(mask_path)).astype(np.int64)
                return torch.from_numpy(mask)
        return torch.zeros(h, w, dtype=torch.int64)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        patient_id, slice_idx, hdct_path, ldct_path = self.samples[idx]
        
        # Load raw slices
        hdct_arr = load_slice(hdct_path)
        ldct_arr = load_slice(ldct_path)
        
        # Normalize → tensors [1, H, W]
        ndct = self._normalize(hdct_arr)
        ldct = self._normalize(ldct_arr)
        
        h, w = ndct.shape[1], ndct.shape[2]
        mask = self._load_mask(patient_id, slice_idx, h, w)
        
        # Augmentation (same seed for ndct + ldct + mask)
        if self.augment:
            if random.random() < 0.5:
                ndct = torch.flip(ndct, dims=[-1])
                ldct = torch.flip(ldct, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if random.random() < 0.5:
                ndct = torch.flip(ndct, dims=[-2])
                ldct = torch.flip(ldct, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
        
        return {
            'ndct':       ndct,
            'ldct':       ldct,
            'mask':       mask,
            'patient_id': patient_id,
            'slice_idx':  slice_idx,
            'hdct_path':  hdct_path,
            'ldct_path':  ldct_path,
        }


def create_dataloaders(data_root: str = CTSliceDataset.DATA_ROOT,
                       masks_root: str = CTSliceDataset.MASKS_ROOT,
                       batch_size: int = 8,
                       num_workers: int = 4) -> Dict:
    """
    Create train/val/test DataLoaders for the real CT data.
    
    Returns dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = CTSliceDataset(data_root=data_root,
                            masks_root=masks_root,
                            split=split,
                            augment=(split == 'train'))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders


class DummyCTDataset(Dataset):
    """
    Generates random tensors in the correct format for testing the training loop
    before real data is available. Same interface as CTSliceDataset.
    """
    def __init__(self, length=100, image_size=512, num_classes=7):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            'ndct':       torch.randn(1, self.image_size, self.image_size),
            'ldct':       torch.randn(1, self.image_size, self.image_size),
            'mask':       torch.randint(0, self.num_classes,
                                        (self.image_size, self.image_size)),
            'patient_id': f'dummy_{idx // 50:03d}',
            'slice_idx':  idx % 50,
            'hdct_path':  '',
            'ldct_path':  '',
        }


if __name__ == '__main__':
    import sys
    
    # Test 1: DummyCTDataset (always runs — no real data needed)
    dummy = DummyCTDataset(length=16, image_size=64)
    loader = DataLoader(dummy, batch_size=4)
    batch = next(iter(loader))
    assert batch['ndct'].shape == (4, 1, 64, 64)
    assert batch['ldct'].shape == (4, 1, 64, 64)
    assert batch['mask'].shape == (4, 64, 64)
    print("DummyCTDataset: PASSED")
    
    # Test 2: Real data (only if path exists)
    real_root = '/home/teaching/Music/Nigam_51/Project_51/data'
    if Path(real_root).exists():
        print(f"\nReal data found at {real_root}. Testing CTSliceDataset...")
        ds = CTSliceDataset(data_root=real_root, split='train')
        print(f"  Train split: {len(ds)} slices")
        
        # Load one sample and check it
        sample = ds[0]
        assert sample['ndct'].shape[0] == 1, "ndct should have channel dim"
        assert sample['ldct'].shape == sample['ndct'].shape, "ndct/ldct shape mismatch"
        assert sample['ndct'].min() >= 0.0 and sample['ndct'].max() <= 1.0, \
            f"ndct out of [0,1]: min={sample['ndct'].min()}, max={sample['ndct'].max()}"
        assert sample['ldct'].min() >= 0.0 and sample['ldct'].max() <= 1.0, \
            f"ldct out of [0,1]: min={sample['ldct'].min()}, max={sample['ldct'].max()}"
        
        H, W = sample['ndct'].shape[1], sample['ndct'].shape[2]
        print(f"  Sample 0: patient={sample['patient_id']}, "
              f"slice={sample['slice_idx']}, shape=[1,{H},{W}]")
        print(f"  ndct range: [{sample['ndct'].min():.3f}, {sample['ndct'].max():.3f}]")
        print(f"  ldct range: [{sample['ldct'].min():.3f}, {sample['ldct'].max():.3f}]")
        print("CTSliceDataset (real data): PASSED")
        
        # Val and test splits
        val_ds   = CTSliceDataset(data_root=real_root, split='val')
        test_ds  = CTSliceDataset(data_root=real_root, split='test')
        total = len(ds) + len(val_ds) + len(test_ds)
        print(f"\nSplit summary: train={len(ds)}, val={len(val_ds)}, "
              f"test={len(test_ds)}, total={total}")
    else:
        print(f"\nReal data not found at {real_root}. Skipping real data test.")
        print("(Run on the machine with the CT data to test CTSliceDataset)")
```

**Verify:** `python data/dataset.py` prints "DummyCTDataset: PASSED" always. If run on the machine with data, also prints "CTSliceDataset (real data): PASSED" and slice counts.

---

## STEP 11C — TotalSegmentator Pseudo-Label Generation

```
Write a script utils/generate_masks.py that runs TotalSegmentator on the real data
to generate organ masks (pseudo-labels) required for Stage 1 training.

This script runs ONCE before training — not during training.

DATA PATHS:
  Input HDCT data: /home/teaching/Music/Nigam_51/Project_51/data/{PATIENT}/HDCT/
  Output masks:    /home/teaching/Music/Nigam_51/Project_51/data/masks/{PATIENT}/{SLICE_IDX:04d}.npy

WHY ONLY HDCT?
  TotalSegmentator runs only on HDCT (clean) images for better segmentation accuracy.
  The same masks are reused for LDCT since both scans cover identical anatomy
  (only noise differs, not structure).

PIPELINE FOR EACH PATIENT:

  Step 1: Collect all HDCT slice files (sorted) using get_sorted_slice_files()
          imported from data.dataset
  Step 2: Load each slice → HU float32 array using load_slice() from data.dataset
  Step 3: Stack into 3D volume:
            volume = np.stack([load_slice(f) for f in hdct_files])
            # shape: [num_slices, H, W], dtype float32, HU values
  Step 4: Write to temp NIfTI:
            import SimpleITK as sitk
            sitk_img = sitk.GetImageFromArray(volume)
            sitk_img.SetSpacing([1.0, 1.0, 1.0])
            tmp_in = f'/tmp/{patient_id}_in.nii.gz'
            sitk.WriteImage(sitk_img, tmp_in)
  Step 5: Run TotalSegmentator:
            from totalsegmentator.python_api import totalsegmentator
            tmp_out = f'/tmp/{patient_id}_seg.nii.gz'
            totalsegmentator(input=tmp_in, output=tmp_out, fast=True, quiet=True)
  Step 6: Load 3D mask:
            mask_3d = sitk.GetArrayFromImage(sitk.ReadImage(tmp_out))
            # shape: [num_slices, H, W], dtype int32, values 0-103
  Step 7: Remap 104 → 7 labels (see function below)
  Step 8: Save per-slice .npy files:
            out_dir = masks_root / patient_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, s in enumerate(mask_3d_remapped):
                np.save(out_dir / f'{i:04d}.npy', s.astype(np.int8))
  Step 9: Delete temp files

LABEL REMAPPING FUNCTION (copy this exactly):

def remap_totalseg_to_7class(mask_104: np.ndarray) -> np.ndarray:
    mask_7 = np.full_like(mask_104, fill_value=6, dtype=np.int8)  # default=soft tissue
    mask_7[mask_104 == 0] = 0                                       # background
    for l in [1, 5]:            mask_7[mask_104 == l] = 1           # liver, spleen
    for l in [2, 3]:            mask_7[mask_104 == l] = 2           # kidney L+R
    for l in [7,8,52,53,54,55,56,57]: mask_7[mask_104 == l] = 3    # vessels, heart
    for l in [10,11,12,13,14]:  mask_7[mask_104 == l] = 4           # lung lobes
    bone_labels = list(range(26,51)) + list(range(58,83)) + [85,86]
    for l in bone_labels:       mask_7[mask_104 == l] = 5           # vertebrae, ribs
    return mask_7

SCRIPT INTERFACE (argparse):

  python utils/generate_masks.py
    --data_root   /home/teaching/Music/Nigam_51/Project_51/data   (default)
    --masks_root  /home/teaching/Music/Nigam_51/Project_51/data/masks  (default)
    --patients    C002 C004   (optional: process specific patients only)
    --overwrite               (optional: redo even if masks exist)

The main() function should:
  1. Find all patient folders (names starting with C, containing HDCT/)
  2. Filter to --patients list if given
  3. For each patient: call generate_masks_for_patient(), catch and log any errors
  4. Print a summary at the end: X patients done, Y failed, total Z slices saved

INSTALL: pip install TotalSegmentator SimpleITK
(First run auto-downloads model weights ~1GB)
```

**Verify:** Run `python utils/generate_masks.py --patients C002` on the machine with data. Confirm `/home/teaching/Music/Nigam_51/Project_51/data/masks/C002/` is created with `.npy` files. Load one and confirm `np.unique()` shows only values 0-6.

---

## STEP 12 — Stage 1 Training Loop

```
In training/train_stage1.py, implement the complete training loop for Stage 1.

This is the most complex step. Implement it section by section.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import os
import time
import math
from pathlib import Path

from models.stage1 import Stage1Model
from losses.stage1_losses import SegmentationLoss, compute_dice_per_class
from models.byol import get_ema_tau
from data.dataset import CTSliceDataset, DummyCTDataset, create_dataloaders


def train_stage1(config_path: str,
                 data_root: str = '/home/teaching/Music/Nigam_51/Project_51/data',
                 masks_root: str = '/home/teaching/Music/Nigam_51/Project_51/data/masks',
                 checkpoint_dir: str = '/home/teaching/Music/Nigam_51/Project_51/checkpoints/stage1',
                 resume_from: str = None,
                 use_dummy_data: bool = False,
                 max_steps: int = None):
    """
    Complete Stage 1 training.
    
    DATA:
      Real data: /home/teaching/Music/Nigam_51/Project_51/data/{C002,C004,...}/{HDCT,LDCT}/
      Masks:     /home/teaching/Music/Nigam_51/Project_51/data/masks/{C002,...}/{slice}.npy
                 (generated by utils/generate_masks.py — must exist before real training)
      
      If use_dummy_data=True: use DummyCTDataset (no files needed, for loop testing)
      If use_dummy_data=False: use create_dataloaders(data_root, masks_root)
    
    Training has two phases:
      Phase 1 (epoch 0 to byol_start_epoch-1):
        Loss = 1.0 * L_seg
      Phase 2 (epoch byol_start_epoch onward):
        Loss = 1.0 * L_seg + 0.1 * L_byol
    
    Each training step:
      1. Load batch: ndct, ldct, mask
      2. Forward: out = model(ndct, return_byol=(epoch >= byol_start_epoch))
         NOTE: Stage 1 trains on 'ndct' (HDCT/clean) images, not ldct
      3. L_seg from out['logits'] and batch['mask']
      4. If BYOL active: L_byol from out['byol_loss']
      5. total_loss = L_seg + byol_weight * L_byol
      6. Backward + AdamW step
      7. EMA update: model.byol.update_target_projector(tau)
    
    Logging (every 100 steps):
      step, epoch, L_seg, L_byol (if active), learning_rate
    
    Validation Dice (every 1000 steps):
      Run model on up to 20 val batches (no_grad)
      Print Dice per class: liver_spleen, kidney, lung, mean
      Save checkpoint if mean Dice improved
    
    Checkpoints (every 5000 steps + on best val Dice):
      Saved to checkpoint_dir/stage1_step_{N}.pth and stage1_best.pth
      Format: { step, epoch, model_state_dict, optimizer_state_dict,
                best_val_dice, config }
    
    Optimizer: AdamW, lr=1e-4, weight_decay=0.01
    LR Schedule: linear warmup (1000 steps) then cosine decay to 0
    
    max_steps: if set, stop training after this many steps (for smoke testing)
    """
    ...


if __name__ == '__main__':
    # Quick smoke test: run 10 steps on dummy data
    # This verifies the entire training loop works end-to-end
    import tempfile
    import sys
    
    print("Running Stage 1 training smoke test (10 steps on dummy data)...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        train_stage1(
            config_path='configs/stage1_config.yaml',
            checkpoint_dir=tmpdir,
            use_dummy_data=True,
            # Override config inline for fast test:
        )
    
    # NOTE: The above will run for all total_steps in config, which is too long
    # for a test. Modify train_stage1 to accept a max_steps override:
    # train_stage1(..., max_steps=10)
    # Then the test completes in <30 seconds.
    
    # The smoke test should:
    # - Complete 10 gradient steps without error
    # - Print loss values (should decrease or at least be finite)
    # - Save 0 checkpoints (no checkpoint at step 10)
    # - Print "Smoke test: PASSED"

IMPORTANT: Add max_steps parameter to train_stage1() so the test can run quickly.
When max_steps is set and step >= max_steps, break out of the training loop.

The smoke test at the end of __main__ should call:
    train_stage1(config_path='configs/stage1_config.yaml',
                 use_dummy_data=True,
                 max_steps=10,
                 checkpoint_dir=tmpdir)
and print "Smoke test: PASSED" if no exception is raised.
```

**Verify:** `python training/train_stage1.py` prints 10 steps of loss values and "Smoke test: PASSED".

---

## STEP 13 — Integration Test (Full Stage 1 Pipeline)

```
Create a file tests/test_stage1_pipeline.py that runs a complete integration
test of all Stage 1 components working together.

This test should NOT require any real CT data — use DummyCTDataset.
Use image_size=64 (not 512) to make it fast.

Test 1: Shape flow test
  - Create Stage1Model
  - Create a batch of dummy data (B=2, image_size=64)
  - Run forward pass
  - Assert all output shapes are correct
  - Print "Test 1 PASSED: Shape flow"

Test 2: Loss computation test
  - Run forward with return_byol=False
  - Compute SegmentationLoss on logits and mask
  - Assert loss is finite and > 0
  - Print "Test 2 PASSED: Loss computation"

Test 3: BYOL loss test
  - Run forward with return_byol=True
  - Assert byol_loss is finite and in [0, 4]
  - Backpropagate total_loss = L_seg + 0.1 * L_byol
  - Assert no NaN gradients in any parameter
  - Print "Test 3 PASSED: BYOL with backward"

Test 4: EMA update test
  - Store a copy of target projector weights
  - Call model.byol.update_target_projector(tau=0.99)
  - Assert target weights changed (but not equal to online weights)
  - Print "Test 4 PASSED: EMA update"

Test 5: Anatomy conditioning test
  - Run model.get_anatomy_conditioning(x)
  - Assert S sums to 1 per pixel
  - Assert e_a has no NaN values
  - Assert e_a[:, k, :] differs across classes k (not all the same)
  - Print "Test 5 PASSED: Anatomy conditioning"

Test 6: Frozen model test
  - Save a checkpoint to a temp file
  - Load with load_stage1_frozen()
  - Assert requires_grad is False for all parameters
  - Run forward pass — assert no gradients tracked
  - Print "Test 6 PASSED: Frozen model"

At the end, print:
  "============================="
  "All Stage 1 integration tests PASSED"
  "============================="
```

**Verify:** `python tests/test_stage1_pipeline.py` prints all 6 tests PASSED.

---

## STEP 14 — Final Checklist

Before moving to Stage 2, confirm:

```
=== Unit tests (all machines, no data needed) ===
python models/vmamba_blocks.py       ← PatchEmbed, PatchMerging, SS2D, VSSBlock
python models/vm_unet.py             ← VMUNetEncoder, VMUNet
python utils/masking.py              ← masked_average_pooling
python losses/stage1_losses.py       ← SegmentationLoss, DiceLoss
python models/byol.py                ← BYOLModule, get_ema_tau
python models/stage1.py              ← Stage1Model, load_stage1_frozen
python data/dataset.py               ← DummyCTDataset (always), CTSliceDataset (if data present)
python training/train_stage1.py      ← 10-step smoke test (use_dummy_data=True)
python tests/test_stage1_pipeline.py ← all 6 integration tests

=== Data preparation (machine with data at /home/teaching/Music/Nigam_51/Project_51/data) ===
# Step 1: Explore data format
python data/explore_data.py

# Step 2: Generate masks for all patients (run once, takes ~10-30 min)
python utils/generate_masks.py
# Expected output: masks/ folder with one subfolder per patient

# Step 3: Verify masks
python -c "
import numpy as np, os
from pathlib import Path
masks_root = Path('/home/teaching/Music/Nigam_51/Project_51/data/masks')
patients = list(masks_root.iterdir())
print(f'Patients with masks: {len(patients)}')
for p in patients[:3]:
    slices = list(p.glob('*.npy'))
    m = np.load(slices[0])
    print(f'  {p.name}: {len(slices)} slices, classes={np.unique(m).tolist()}')
"

# Step 4: Full dataloader test
python -c "
from data.dataset import create_dataloaders
loaders = create_dataloaders()
batch = next(iter(loaders['train']))
print('Train batch ndct:', batch['ndct'].shape)
print('Train batch mask:', batch['mask'].shape)
print('Patients in batch:', batch['patient_id'])
"

=== Model size check ===
python -c "
from models.stage1 import Stage1Model
m = Stage1Model()
params = sum(p.numel() for p in m.parameters()) / 1e6
print(f'Stage 1 total parameters: {params:.1f}M')
"
# Expected: roughly 40-80M

=== Training (machine with data + GPU) ===
# Start real training:
python training/train_stage1.py

# Or with explicit paths:
python -c "
from training.train_stage1 import train_stage1
train_stage1(
    config_path='configs/stage1_config.yaml',
    data_root='/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root='/home/teaching/Music/Nigam_51/Project_51/data/masks',
    checkpoint_dir='checkpoints/stage1',
)
"

If all pass, Stage 1 is complete and ready.
The trained checkpoint at /home/teaching/Music/Nigam_51/Project_51/checkpoints/stage1/stage1_best.pth is the
input to Stage 2 training (loaded via load_stage1_frozen()).
```

---

## Quick Reference: Output Shapes

| Variable | Shape | Format | Used for |
|---|---|---|---|
| Input CT | [B, 1, 512, 512] | BCHW | Raw input |
| After PatchEmbed | [B, 96, 128, 128] | BCHW | Encoder scale 1 |
| Skip 1 | [B, 96, 128, 128] | BCHW | Decoder |
| Skip 2 | [B, 192, 64, 64] | BCHW | Decoder |
| Skip 3 | [B, 384, 32, 32] | BCHW | Decoder |
| F (Bottleneck) | [B, 768, 16, 16] | BCHW | BYOL |
| BYOL z, q | [B, 256] | — | BYOL loss |
| decoder_features | [B, 96, 512, 512] | BCHW | e_a computation |
| logits | [B, 7, 512, 512] | BCHW | L_seg loss |
| S | [B, 7, 512, 512] | BCHW | Stage 2 input |
| e_a | [B, 7, 96] | — | Stage 2 input |
| VSS Block I/O | [B, H, W, C] | BHWC | Must permute |

## Class Label Map

| ID | Organs | Why Grouped |
|---|---|---|
| 0 | Background, air | Quantum noise dominates at -1000 HU |
| 1 | Liver, Spleen | Similar HU (40-60), similar noise |
| 2 | Kidney L + R | Bright cortex + dark medulla pattern |
| 3 | Aorta, IVC | Blood vessels, motion artifacts |
| 4 | Lung + vessels | Air-filled (-700 HU), totally different |
| 5 | Vertebrae, Ribs | Bone (HU > 400) |
| 6 | Soft tissue, Muscle | Everything else |
