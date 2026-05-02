
#!/usr/bin/env python3
"""
utils/generate_masks.py

TotalSegmentator Pseudo-Label Generation Pipeline
==================================================
Runs ONCE, offline, before any training.
NOT imported by training code — standalone CLI tool.

Usage:
    python utils/generate_masks.py
    python utils/generate_masks.py --patients C002 C004
    python utils/generate_masks.py --overwrite
    python utils/generate_masks.py --data_root /path/to/data --masks_root /path/to/masks

Requirements (add to requirements.txt):
    TotalSegmentator>=2.0.0
    SimpleITK>=2.2.0
    pydicom>=2.3.0
    Pillow>=9.0.0
"""

import sys
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CLASS NAMES for 7-class remapping
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'background',    # 0
    'liver_spleen',  # 1
    'kidney',        # 2
    'vessel',        # 3
    'lung',          # 4
    'bone',          # 5
    'soft_tissue',   # 6
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: load_slice_hu
# ─────────────────────────────────────────────────────────────────────────────
def load_slice_hu(filepath: str) -> np.ndarray:
    """
    Load a single CT slice and return a float32 HU array of shape [H, W].

    Handles all formats: .dcm, .IMA, .npy, .png, .tif, .tiff
    Always returns float32 [H, W] with values in approximate HU range.
    """
    p = Path(filepath)
    suffix = p.suffix.lower()

    # ── DICOM branch ──────────────────────────────────────────────────────
    if suffix in ('.dcm', '.ima'):
        import pydicom
        ds = pydicom.dcmread(filepath)
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, 'RescaleSlope',     1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', -1024.0))
        return arr * slope + intercept          # Hounsfield Units

    # ── NumPy branch ──────────────────────────────────────────────────────
    elif suffix == '.npy':
        arr = np.load(filepath).astype(np.float32)
        # Detect normalised [0,1] data by checking whether min is near 0
        # and max is near 1 (HU data has min near -1000).
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_min >= -5.0 and arr_max <= 1.05:
            # Normalised → rescale back to HU
            arr = arr * 4000.0 - 1000.0
        # Otherwise already in HU range — return as-is
        return arr

    # ── Image branch ──────────────────────────────────────────────────────
    elif suffix in ('.png', '.tif', '.tiff'):
        from PIL import Image
        arr = np.array(Image.open(filepath)).astype(np.float32)
        # 16-bit convention: raw HU + 1000 offset
        arr = arr - 1000.0
        return arr

    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}' for file {filepath}\n"
            f"Supported formats: .dcm, .IMA, .npy, .png, .tif, .tiff"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: get_sorted_slice_paths
# ─────────────────────────────────────────────────────────────────────────────
def get_sorted_slice_paths(folder: str) -> list:
    """
    Return all slice file paths in folder, sorted in correct anatomical order.

    Uses NUMERIC sort (not alphabetical) by extracting the integer from
    the filename stem.

    Supports: .dcm, .IMA, .npy, .png, .tif, .tiff
    Returns: sorted list of absolute path strings.
    Raises ValueError if folder is empty.
    """
    SUPPORTED = {'.dcm', '.ima', '.npy', '.png', '.tif', '.tiff'}
    folder_path = Path(folder)

    paths = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ]

    if not paths:
        raise ValueError(
            f"No supported slice files found in '{folder}'.\n"
            f"Supported extensions: {', '.join(sorted(SUPPORTED))}"
        )

    # Numeric sort: extract digits from stem, default to 0 if none found
    def numeric_key(f: Path) -> int:
        digits = ''.join(filter(str.isdigit, f.stem))
        return int(digits) if digits else 0

    paths.sort(key=numeric_key)
    return [str(p.resolve()) for p in paths]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: stack_slices_to_volume
# ─────────────────────────────────────────────────────────────────────────────
def stack_slices_to_volume(slice_paths: list) -> np.ndarray:
    """
    Load all slices and stack into a 3D volume.

    Returns: float32 array of shape [num_slices, H, W] in HU values.
    Prints progress every 50 slices.
    """
    slices = []
    n_total = len(slice_paths)

    for i, path in enumerate(slice_paths):
        slc = load_slice_hu(path)
        slices.append(slc)

        # Progress every 50 slices (1-indexed display)
        if (i + 1) % 50 == 0:
            print(f"  Loading slices: {i + 1}/{n_total}")

    volume = np.stack(slices, axis=0).astype(np.float32)  # [D, H, W]
    return volume


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4: volume_to_nifti
# ─────────────────────────────────────────────────────────────────────────────
def volume_to_nifti(volume: np.ndarray, out_path: str,
                    spacing: tuple = (1.0, 1.0, 1.0)):
    """
    Convert a [D, H, W] float32 numpy volume to a NIfTI file (.nii.gz).

    Parameters
    ----------
    volume   : float32 ndarray [D, H, W]
    out_path : destination path (should end in .nii.gz)
    spacing  : (x_mm, y_mm, z_mm) voxel size; use (1,1,1) if unknown.
               For DICOM data use (PixelSpacing[0], PixelSpacing[1],
               SliceThickness) from the header.
    """
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(volume)   # SimpleITK expects [D, H, W]
    img.SetSpacing(spacing)                # (x_mm, y_mm, z_mm)
    sitk.WriteImage(img, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5: run_totalsegmentator
# ─────────────────────────────────────────────────────────────────────────────
def run_totalsegmentator(input_nifti: str, output_nifti: str):
    """
    Run TotalSegmentator on the input NIfTI volume.

    Output: a single NIfTI file where each voxel carries an integer label 0-103.
    First run downloads model weights (~1 GB) to ~/.totalsegmentator/
    """
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(
        input=input_nifti,
        output=output_nifti,
        fast=True,    # faster inference, slightly less accurate
        quiet=True,   # suppress verbose output
        ml=True,      # multi-label: single output file, not per-organ files
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 6: remap_totalseg_to_7class
# ─────────────────────────────────────────────────────────────────────────────
def remap_totalseg_to_7class(mask_104: np.ndarray) -> np.ndarray:
    """
    Remap TotalSegmentator's 104 labels → 7 denoising-relevant classes.

    Input:  [D, H, W] or [H, W] integer array, values 0-103
    Output: same shape, dtype int8, values 0-6

    Class 0  background / air            — TotalSeg label 0
    Class 1  liver / spleen              — labels 1, 5
    Class 2  kidney                      — labels 2, 3
    Class 3  vessels / heart             — labels 7, 8, 52-57
    Class 4  lung                        — labels 10-14
    Class 5  bone (vertebrae + ribs)     — labels 26-50, 58-82, 85, 86
    Class 6  soft tissue / other         — everything else
    """
    # Start with everything as soft tissue (class 6)
    mask_7 = np.full_like(mask_104, fill_value=6, dtype=np.int8)

    # Class 0 — background / air
    mask_7[mask_104 == 0] = 0

    # Class 1 — liver / spleen
    for label in [1, 5]:
        mask_7[mask_104 == label] = 1

    # Class 2 — kidney (right + left)
    for label in [2, 3]:
        mask_7[mask_104 == label] = 2

    # Class 3 — vessels / heart
    for label in [7, 8, 52, 53, 54, 55, 56, 57]:
        mask_7[mask_104 == label] = 3

    # Class 4 — lung (all lobes)
    for label in [10, 11, 12, 13, 14]:
        mask_7[mask_104 == label] = 4

    # Class 5 — bone
    # Vertebrae (26-50), ribs (58-82), sternum (85), costal cartilages (86)
    bone_labels = list(range(26, 51)) + list(range(58, 83)) + [85, 86]
    for label in bone_labels:
        mask_7[mask_104 == label] = 5

    # Class 6 is already set as the default — no extra work needed

    return mask_7


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 7: save_mask_slices
# ─────────────────────────────────────────────────────────────────────────────
def save_mask_slices(mask_3d: np.ndarray, out_dir: Path) -> int:
    """
    Split a 3D mask [D, H, W] into 2D slices and save each as .npy (int8).

    Returns: number of slices saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, slice_2d in enumerate(mask_3d):
        np.save(out_dir / f'{i:04d}.npy', slice_2d)   # int8, values 0-6

    return len(mask_3d)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 8: generate_masks_for_patient  (full pipeline for one patient)
# ─────────────────────────────────────────────────────────────────────────────
def generate_masks_for_patient(
    patient_id: str,
    data_root:  Path,
    masks_root: Path,
    overwrite:  bool = False,
) -> dict:
    """
    Full pseudo-label pipeline for one patient.

    Steps
    -----
    1.  Skip if masks already exist and overwrite=False
    2.  Locate HDCT folder
    3.  Sort slices
    4.  Stack into 3-D volume
    5.  Write temp NIfTI
    6.  Run TotalSegmentator
    7.  Load output mask
    8.  Remap 104 → 7 classes
    9.  Save 2-D slices
    10. Clean up temp files

    Returns a status dict with keys:
        patient, status ('done'|'skipped'|'failed'), n_slices, time_sec, error
    """
    import SimpleITK as sitk

    out_dir = masks_root / patient_id

    # ── Step 1: Skip if already done ─────────────────────────────────────
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        return {'patient': patient_id, 'status': 'skipped', 'n_slices': 0}

    # ── Step 2: Locate HDCT directory ────────────────────────────────────
    hdct_dir = data_root / patient_id / 'HDCT'
    if not hdct_dir.exists():
        raise FileNotFoundError(
            f"HDCT directory not found: {hdct_dir}"
        )

    # Temp file paths
    tmp_in  = Path(f'/tmp/totalseg_{patient_id}_input.nii.gz')
    tmp_out = Path(f'/tmp/totalseg_{patient_id}_output.nii.gz')

    try:
        # ── Step 3: Get sorted slice paths ───────────────────────────────
        slice_paths = get_sorted_slice_paths(str(hdct_dir))
        n_slices = len(slice_paths)
        print(f"  Found {n_slices} HDCT slices")

        # ── Step 4: Stack into 3-D volume ─────────────────────────────────
        volume = stack_slices_to_volume(slice_paths)
        print(
            f"  Volume shape: {volume.shape}, "
            f"HU range: [{volume.min():.0f}, {volume.max():.0f}]"
        )

        # ── Step 5: Write input NIfTI ─────────────────────────────────────
        volume_to_nifti(volume, str(tmp_in), spacing=(1.0, 1.0, 1.0))

        # ── Step 6: TotalSegmentator ──────────────────────────────────────
        print("  Running TotalSegmentator …")
        run_totalsegmentator(str(tmp_in), str(tmp_out))

        # ── Step 7: Load output mask ──────────────────────────────────────
        mask_sitk = sitk.ReadImage(str(tmp_out))
        mask_3d   = sitk.GetArrayFromImage(mask_sitk)   # [D, H, W] int32

        # Sanity check: spatial dimensions must match the volume
        assert mask_3d.shape == volume.shape, (
            f"Shape mismatch: volume {volume.shape} vs mask {mask_3d.shape}"
        )

        # ── Step 8: Remap 104 → 7 classes ────────────────────────────────
        mask_7 = remap_totalseg_to_7class(mask_3d)

        # Print class distribution
        unique, counts = np.unique(mask_7, return_counts=True)
        print("  Class distribution:")
        for cls, cnt in zip(unique, counts):
            pct = 100.0 * cnt / mask_7.size
            print(f"    class {cls} ({CLASS_NAMES[cls]}): {pct:.1f}%")

        # ── Step 9: Save 2-D slices ───────────────────────────────────────
        n_saved = save_mask_slices(mask_7, out_dir)
        print(f"  Saved {n_saved} mask slices to {out_dir}")

        # ── Step 10: Clean up temp files ──────────────────────────────────
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)

        return {
            'patient':  patient_id,
            'status':   'done',
            'n_slices': n_saved,
        }

    except Exception as e:
        # Clean up on error
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)

        return {
            'patient': patient_id,
            'status':  'failed',
            'error':   str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description=(
            'Generate TotalSegmentator pseudo-labels for CT denoising training.\n'
            'Runs ONCE, offline, before any training.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data_root',
        default='/home/teaching/Music/Nigam_51/Project_51/data',
        help='Root directory containing patient folders (default: %(default)s)',
    )
    parser.add_argument(
        '--masks_root',
        default='/home/teaching/Music/Nigam_51/Project_51/data/masks',
        help='Output directory for mask slices (default: %(default)s)',
    )
    parser.add_argument(
        '--patients',
        nargs='+',
        default=None,
        metavar='PATIENT_ID',
        help=(
            'Patient IDs to process, e.g. --patients C002 C004. '
            'Default: all patients found in data_root.'
        ),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate masks even if they already exist.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help=(
            'Number of patients to process in parallel. '
            'Default: 1 (sequential). '
            'NOTE: TotalSegmentator itself uses all available CPU/GPU cores, '
            'so parallel workers > 1 only helps if you have multiple GPUs.'
        ),
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    masks_root = Path(args.masks_root)

    # ── Discover patients ─────────────────────────────────────────────────
    if not data_root.exists():
        print(f"ERROR: data_root does not exist: {data_root}")
        sys.exit(1)

    all_patients = sorted([
        p.name
        for p in data_root.iterdir()
        if p.is_dir()
        and p.name.startswith('C')          # patient IDs start with 'C'
        and (p / 'HDCT').exists()
    ])

    if not all_patients:
        print(f"ERROR: No patient folders found in {data_root}")
        print("  Expected structure: {data_root}/{PATIENT_ID}/HDCT/")
        sys.exit(1)

    # ── Validate requested patients ───────────────────────────────────────
    to_process = args.patients if args.patients else all_patients
    invalid    = [p for p in to_process if p not in all_patients]

    if invalid:
        print(f"WARNING: These patients were not found and will be skipped:")
        for pid in invalid:
            print(f"  {pid}")
        to_process = [p for p in to_process if p in all_patients]

    if not to_process:
        print("ERROR: No valid patients to process.")
        sys.exit(1)

    # ── Banner ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TotalSegmentator Pseudo-Label Generation")
    print("=" * 60)
    print(f"  Data root   : {data_root}")
    print(f"  Masks root  : {masks_root}")
    print(f"  Patients    : {len(to_process)} / {len(all_patients)} total")
    print(f"  Overwrite   : {args.overwrite}")
    print(f"  Workers     : {args.workers}")
    print()

    # ── Process patients ──────────────────────────────────────────────────
    results   = []
    t_total   = time.time()

    if args.workers == 1:
        # Sequential (default, safest)
        for i, patient_id in enumerate(to_process):
            print(f"[{i + 1}/{len(to_process)}] {patient_id}")
            t0 = time.time()
            result = generate_masks_for_patient(
                patient_id, data_root, masks_root, args.overwrite
            )
            elapsed          = time.time() - t0
            result['time_sec'] = round(elapsed, 1)
            results.append(result)

            status = result['status']
            if status == 'done':
                print(f"  ✓ Done in {elapsed:.0f}s")
            elif status == 'skipped':
                print(f"  → Skipped (masks already exist)")
            else:
                print(f"  ✗ FAILED: {result.get('error', 'unknown error')}")
            print()

    else:
        # Parallel (one patient per worker)
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for patient_id in to_process:
                fut = executor.submit(
                    generate_masks_for_patient,
                    patient_id, data_root, masks_root, args.overwrite
                )
                futures[fut] = patient_id

            for fut in as_completed(futures):
                patient_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        'patient': patient_id,
                        'status':  'failed',
                        'error':   str(exc),
                    }

                result.setdefault('time_sec', 0)
                results.append(result)

                status = result['status']
                idx    = len(results)
                print(f"[{idx}/{len(to_process)}] {patient_id}")
                if status == 'done':
                    print(f"  ✓ Done in {result['time_sec']:.0f}s")
                elif status == 'skipped':
                    print(f"  → Skipped (masks already exist)")
                else:
                    print(f"  ✗ FAILED: {result.get('error', 'unknown error')}")
                print()

    # ── Summary ───────────────────────────────────────────────────────────
    done    = [r for r in results if r['status'] == 'done']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed  = [r for r in results if r['status'] == 'failed']

    total_slices = sum(r.get('n_slices', 0) for r in done)
    total_time   = time.time() - t_total

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Done    : {len(done)} patients,  {total_slices} slices saved")
    print(f"  Skipped : {len(skipped)} (already had masks)")
    print(f"  Failed  : {len(failed)}")

    if failed:
        print()
        print("  Failed patients:")
        for r in failed:
            print(f"    {r['patient']}: {r.get('error', '')}")

    print(f"\n  Total wall-clock time: {total_time:.0f}s")
    print("=" * 60)

    # Non-zero exit code if any patient failed
    if failed:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
