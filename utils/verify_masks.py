#!/usr/bin/env python3
"""
utils/verify_masks.py

Post-generation verification script for TotalSegmentator pseudo-labels.
Run after utils/generate_masks.py has completed.

Checks every patient in DATA_ROOT:
  1. HDCT slice count == mask slice count
  2. Mask shape matches CT slice shape
  3. Mask dtype is int8
  4. Mask values are only in {0,1,2,3,4,5,6}
  5. Mask is not all zeros (segmentation is not blank)

Usage:
    python utils/verify_masks.py
    python utils/verify_masks.py --data_root /path/to/data --masks_root /path/to/masks
    python utils/verify_masks.py --patients C002 C004
    python utils/verify_masks.py --verbose        # show per-slice details
    python utils/verify_masks.py --fix_dtype      # cast masks to int8 if wrong dtype
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT  = '/home/teaching/Music/Nigam_51/Project_51/data'
MASKS_ROOT = '/home/teaching/Music/Nigam_51/Project_51/data/masks'

VALID_LABELS  = frozenset(range(7))   # {0,1,2,3,4,5,6}
EXPECTED_DTYPE = np.int8

CLASS_NAMES = [
    'background',   # 0
    'liver_spleen', # 1
    'kidney',       # 2
    'vessel',       # 3
    'lung',         # 4
    'bone',         # 5
    'soft_tissue',  # 6
]

SUPPORTED_CT_EXTS = {'.dcm', '.ima', '.npy', '.png', '.tif', '.tiff'}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_key(p: Path) -> int:
    """Extract leading integer from filename stem for numeric sorting."""
    digits = ''.join(filter(str.isdigit, p.stem))
    return int(digits) if digits else 0


def _get_ct_paths(hdct_dir: Path) -> list:
    """Return numerically-sorted list of CT slice paths in hdct_dir."""
    paths = [
        p for p in hdct_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_CT_EXTS
    ]
    paths.sort(key=_numeric_key)
    return paths


def _get_mask_paths(mask_dir: Path) -> list:
    """Return numerically-sorted list of .npy mask paths in mask_dir."""
    paths = [
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() == '.npy'
    ]
    paths.sort(key=_numeric_key)
    return paths


def _load_ct_shape(ct_path: Path) -> tuple:
    """
    Return (H, W) shape of a CT slice without loading full HU conversion.
    Handles .dcm/.IMA, .npy, .png/.tif/.tiff.
    """
    suffix = ct_path.suffix.lower()

    if suffix in ('.dcm', '.ima'):
        import pydicom
        ds  = pydicom.dcmread(str(ct_path), stop_before_pixels=False)
        arr = ds.pixel_array
        return arr.shape[-2], arr.shape[-1]   # (H, W) even for RGB DICOMs

    elif suffix == '.npy':
        arr = np.load(str(ct_path), mmap_mode='r')
        return arr.shape[-2], arr.shape[-1]

    else:  # .png / .tif / .tiff
        from PIL import Image
        with Image.open(str(ct_path)) as img:
            w, h = img.size    # PIL returns (width, height)
        return h, w


def _human_bytes(n_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


# ─────────────────────────────────────────────────────────────────────────────
# PER-PATIENT VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_patient(
    patient_id:  str,
    data_root:   Path,
    masks_root:  Path,
    verbose:     bool = False,
    fix_dtype:   bool = False,
) -> dict:
    """
    Verify masks for one patient.

    Returns
    -------
    dict with keys:
        patient        str
        hdct_slices    int   — number of CT slices found
        mask_slices    int   — number of mask .npy files found
        count_match    bool  — hdct_slices == mask_slices
        shape_ok       bool  — all checked masks have correct (H, W)
        dtype_ok       bool  — all checked masks have dtype int8
        labels_ok      bool  — all checked masks contain only {0..6}
        not_blank      bool  — no checked mask is all-zero
        valid          bool  — all four checks pass
        issues         list  — human-readable problem strings
        class_dist     dict  — aggregated class distribution {cls: count}
        mask_bytes     int   — total bytes consumed by mask files
    """
    result = {
        'patient':     patient_id,
        'hdct_slices': 0,
        'mask_slices': 0,
        'count_match': False,
        'shape_ok':    True,
        'dtype_ok':    True,
        'labels_ok':   True,
        'not_blank':   True,
        'valid':       False,
        'issues':      [],
        'class_dist':  {c: 0 for c in range(7)},
        'mask_bytes':  0,
    }

    hdct_dir = data_root  / patient_id / 'HDCT'
    mask_dir = masks_root / patient_id

    # ── Check directories exist ───────────────────────────────────────────
    if not hdct_dir.exists():
        result['issues'].append(f"HDCT directory missing: {hdct_dir}")
        return result

    if not mask_dir.exists():
        result['issues'].append(f"Mask directory missing: {mask_dir}")
        return result

    # ── Count files ───────────────────────────────────────────────────────
    ct_paths   = _get_ct_paths(hdct_dir)
    mask_paths = _get_mask_paths(mask_dir)

    n_ct   = len(ct_paths)
    n_mask = len(mask_paths)

    result['hdct_slices'] = n_ct
    result['mask_slices'] = n_mask

    if n_ct == 0:
        result['issues'].append("No CT slices found in HDCT directory")
        return result

    if n_mask == 0:
        result['issues'].append("No mask .npy files found in mask directory")
        return result

    if n_ct != n_mask:
        result['issues'].append(
            f"Count mismatch: {n_ct} CT slices vs {n_mask} mask files"
        )
    else:
        result['count_match'] = True

    # ── Pick indices to check: first, middle, last ────────────────────────
    check_indices = sorted({0, n_mask // 2, n_mask - 1})

    # Also get CT shape from the same three positions (if CT count matches)
    ct_check_paths = []
    for idx in check_indices:
        if idx < n_ct:
            ct_check_paths.append((idx, ct_paths[idx]))

    # ── Load CT shapes for reference ──────────────────────────────────────
    ct_shapes = {}
    for idx, ct_path in ct_check_paths:
        try:
            h, w = _load_ct_shape(ct_path)
            ct_shapes[idx] = (h, w)
        except Exception as exc:
            result['issues'].append(
                f"  Could not read CT shape at index {idx}: {exc}"
            )

    # ── Verify each selected mask slice ───────────────────────────────────
    for idx in check_indices:
        if idx >= n_mask:
            continue

        mask_path = mask_paths[idx]
        try:
            mask = np.load(str(mask_path))
        except Exception as exc:
            result['issues'].append(
                f"  Cannot load mask at index {idx} ({mask_path.name}): {exc}"
            )
            result['shape_ok']  = False
            result['dtype_ok']  = False
            result['labels_ok'] = False
            result['not_blank'] = False
            continue

        # ── Shape check ───────────────────────────────────────────────────
        if mask.ndim != 2:
            result['issues'].append(
                f"  Mask [{idx}] has {mask.ndim}D shape {mask.shape}; expected 2D"
            )
            result['shape_ok'] = False
        elif idx in ct_shapes:
            expected_hw = ct_shapes[idx]
            if mask.shape != expected_hw:
                result['issues'].append(
                    f"  Mask [{idx}] shape {mask.shape} != CT shape {expected_hw}"
                )
                result['shape_ok'] = False

        # ── Dtype check ───────────────────────────────────────────────────
        if mask.dtype != EXPECTED_DTYPE:
            if fix_dtype:
                # Attempt silent cast and re-save
                try:
                    mask_fixed = mask.astype(np.int8)
                    np.save(str(mask_path), mask_fixed)
                    mask = mask_fixed
                    if verbose:
                        print(
                            f"    [FIX] Cast mask [{idx}] "
                            f"{mask_path.name} to int8"
                        )
                except Exception as exc:
                    result['issues'].append(
                        f"  Could not fix dtype at index {idx}: {exc}"
                    )
                    result['dtype_ok'] = False
            else:
                result['issues'].append(
                    f"  Mask [{idx}] dtype {mask.dtype} != int8"
                )
                result['dtype_ok'] = False

        # ── Label validity check ──────────────────────────────────────────
        unique_vals = set(np.unique(mask).tolist())
        invalid_vals = unique_vals - VALID_LABELS
        if invalid_vals:
            result['issues'].append(
                f"  Mask [{idx}] contains invalid label(s): {sorted(invalid_vals)}"
            )
            result['labels_ok'] = False

        # ── Non-blank check ───────────────────────────────────────────────
        if np.all(mask == 0):
            result['issues'].append(
                f"  Mask [{idx}] ({mask_path.name}) is entirely zero (blank)"
            )
            result['not_blank'] = False

        # ── Accumulate class distribution ─────────────────────────────────
        for cls_id in range(7):
            result['class_dist'][cls_id] += int(np.sum(mask == cls_id))

        if verbose:
            vals_str = ', '.join(
                f"{CLASS_NAMES[v]}:{np.sum(mask == v)}"
                for v in sorted(unique_vals)
                if v in VALID_LABELS
            )
            print(
                f"    slice {idx:04d}: shape={mask.shape}, "
                f"dtype={mask.dtype}, classes=[{vals_str}]"
            )

    # ── Total mask storage ────────────────────────────────────────────────
    result['mask_bytes'] = sum(p.stat().st_size for p in mask_paths)

    # ── Overall validity ──────────────────────────────────────────────────
    result['valid'] = (
        result['count_match']
        and result['shape_ok']
        and result['dtype_ok']
        and result['labels_ok']
        and result['not_blank']
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bool_flag(value: bool, true_str: str = 'YES', false_str: str = 'NO') -> str:
    return true_str if value else false_str


def _print_table_row(
    patient:     str,
    hdct:        int,
    masks:       int,
    count_match: bool,
    valid:       bool,
    issues:      list,
):
    """Print one row of the summary table."""
    match_str = _bool_flag(count_match, 'YES', 'NO ')
    valid_str = _bool_flag(valid,       'YES', 'NO ')

    # Truncate patient id to fixed width
    pid = patient[:10].ljust(10)

    print(
        f"  {pid} | "
        f"hdct={hdct:>4} | "
        f"masks={masks:>4} | "
        f"match={match_str} | "
        f"valid={valid_str}"
        + (f"  ← {issues[0]}" if issues else "")
    )


def _print_class_distribution(class_dist: dict, total_pixels: int):
    """Print per-class pixel percentages."""
    print()
    print("  Aggregated class distribution across verified slices:")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = class_dist.get(cls_id, 0)
        pct   = 100.0 * count / total_pixels if total_pixels > 0 else 0.0
        bar   = '█' * int(pct / 2)   # 1 block per 2 %
        print(f"    class {cls_id} {name:<14} {pct:5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Verify TotalSegmentator pseudo-label masks.\n'
            'Run after utils/generate_masks.py has finished.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data_root',
        default=DATA_ROOT,
        help='Root directory containing patient folders (default: %(default)s)',
    )
    parser.add_argument(
        '--masks_root',
        default=MASKS_ROOT,
        help='Root directory for mask .npy files (default: %(default)s)',
    )
    parser.add_argument(
        '--patients',
        nargs='+',
        default=None,
        metavar='PATIENT_ID',
        help='Specific patient IDs to verify. Default: all.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-slice details for every checked slice.',
    )
    parser.add_argument(
        '--fix_dtype',
        action='store_true',
        help=(
            'Automatically cast masks with wrong dtype to int8 and re-save. '
            'Use with caution — modifies files on disk.'
        ),
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    masks_root = Path(args.masks_root)

    # ── Sanity-check top-level directories ───────────────────────────────
    if not data_root.exists():
        print(f"ERROR: data_root does not exist: {data_root}")
        sys.exit(1)

    if not masks_root.exists():
        print(f"ERROR: masks_root does not exist: {masks_root}")
        print("       Did you run utils/generate_masks.py yet?")
        sys.exit(1)

    # ── Discover patients ─────────────────────────────────────────────────
    all_patients = sorted([
        p.name
        for p in data_root.iterdir()
        if p.is_dir()
        and p.name.startswith('C')
        and (p / 'HDCT').exists()
    ])

    if not all_patients:
        print(f"ERROR: No patient folders found in {data_root}")
        sys.exit(1)

    if args.patients:
        unknown = [p for p in args.patients if p not in all_patients]
        if unknown:
            print(f"WARNING: Unknown patient IDs will be skipped: {unknown}")
        to_verify = [p for p in args.patients if p in all_patients]
    else:
        to_verify = all_patients

    if not to_verify:
        print("ERROR: No patients to verify.")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Mask Verification Report")
    print("=" * 70)
    print(f"  data_root  : {data_root}")
    print(f"  masks_root : {masks_root}")
    print(f"  Patients   : {len(to_verify)} / {len(all_patients)} total")
    if args.fix_dtype:
        print("  fix_dtype  : ENABLED — will overwrite files with wrong dtype")
    print()
    print(
        f"  {'PATIENT':10s} | "
        f"{'HDCT':>9} | "
        f"{'MASKS':>10} | "
        f"{'MATCH':>8} | "
        f"{'VALID':>8}"
    )
    print("  " + "-" * 62)

    # ── Run verification ──────────────────────────────────────────────────
    results = []
    for patient_id in to_verify:
        if args.verbose:
            print(f"\n  [{patient_id}]")

        result = verify_patient(
            patient_id = patient_id,
            data_root  = data_root,
            masks_root = masks_root,
            verbose    = args.verbose,
            fix_dtype  = args.fix_dtype,
        )
        results.append(result)

        _print_table_row(
            patient     = result['patient'],
            hdct        = result['hdct_slices'],
            masks       = result['mask_slices'],
            count_match = result['count_match'],
            valid       = result['valid'],
            issues      = result['issues'],
        )

        # Print extra issues (beyond the first one shown inline)
        if len(result['issues']) > 1:
            for issue in result['issues'][1:]:
                print(f"{'':>15}  ↳ {issue}")

    # ── Aggregated statistics ─────────────────────────────────────────────
    n_total         = len(results)
    n_valid         = sum(1 for r in results if r['valid'])
    n_count_mismatch= sum(1 for r in results if not r['count_match'])
    n_blank         = sum(1 for r in results if not r['not_blank'])
    n_dtype_bad     = sum(1 for r in results if not r['dtype_ok'])
    n_labels_bad    = sum(1 for r in results if not r['labels_ok'])
    n_shape_bad     = sum(1 for r in results if not r['shape_ok'])

    total_hdct_slices  = sum(r['hdct_slices']  for r in results)
    total_mask_slices  = sum(r['mask_slices']   for r in results)
    total_mask_bytes   = sum(r['mask_bytes']    for r in results)

    # Aggregate class distribution across all patients
    agg_class_dist = {c: 0 for c in range(7)}
    for r in results:
        for cls_id, cnt in r['class_dist'].items():
            agg_class_dist[cls_id] += cnt
    total_dist_pixels = sum(agg_class_dist.values())

    all_counts_match = (n_count_mismatch == 0)
    any_blank        = (n_blank > 0)
    all_valid        = (n_valid == n_total)

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total patients            : {n_total}")
    print(f"  All counts matched        : {_bool_flag(all_counts_match)}")
    print(f"  Any blank masks found     : {_bool_flag(any_blank, 'YES ⚠', 'NO')}")
    print(f"  All masks fully valid     : {_bool_flag(all_valid)}")
    print()
    print(f"  Patients with issues:")
    print(f"    Count mismatch          : {n_count_mismatch}")
    print(f"    Wrong shape             : {n_shape_bad}")
    print(f"    Wrong dtype (not int8)  : {n_dtype_bad}")
    print(f"    Invalid label values    : {n_labels_bad}")
    print(f"    Blank masks             : {n_blank}")
    print()
    print(f"  Total CT slices           : {total_hdct_slices:,}")
    print(f"  Total mask files          : {total_mask_slices:,}")
    print(f"  Total mask storage        : {_human_bytes(total_mask_bytes)}")

    if total_dist_pixels > 0:
        _print_class_distribution(agg_class_dist, total_dist_pixels)

    # ── Detailed failure listing ──────────────────────────────────────────
    failed = [r for r in results if not r['valid']]
    if failed:
        print()
        print("  Patients requiring attention:")
        for r in failed:
            print(f"    {r['patient']}:")
            for issue in r['issues']:
                print(f"      • {issue}")

    print("=" * 70)

    # ── Recommendations ───────────────────────────────────────────────────
    if not all_valid:
        print()
        print("RECOMMENDATIONS:")
        if n_count_mismatch:
            print(
                "  • Count mismatches: re-run generate_masks.py with --overwrite "
                "for affected patients."
            )
        if n_blank:
            print(
                "  • Blank masks: TotalSegmentator may have failed silently. "
                "Re-run with --overwrite and check for GPU/memory errors."
            )
        if n_dtype_bad and not args.fix_dtype:
            print(
                "  • Wrong dtype: re-run verify_masks.py with --fix_dtype to "
                "automatically correct int8 casting."
            )
        if n_labels_bad:
            print(
                "  • Invalid labels: re-run generate_masks.py --overwrite. "
                "This usually means remap_totalseg_to_7class produced unexpected values."
            )
        sys.exit(1)
    else:
        print()
        print("✓  All masks verified successfully. Ready for training.")
        print()


if __name__ == '__main__':
    main()



    