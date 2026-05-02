#!/usr/bin/env python3
"""
utils/explore_data.py
─────────────────────
Standalone data-inspection script for the anatomy-aware CT denoising project.
Run ONCE before writing the dataset loader to understand the raw files on disk.

Usage:
    python utils/explore_data.py

Requires: numpy, pydicom, Pillow   (pip install numpy pydicom Pillow)
No GPU, no PyTorch, no training code needed.
"""

import os
import sys
import time
import pathlib
import importlib

import numpy as np

# ── soft imports (report version or warn) ─────────────────────────────────────
def _try_import(name: str):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {name} {ver}")
        return mod
    except ImportError:
        print(f"  [MISSING] {name} — install with: pip install {name}")
        return None

print(f"Python {sys.version.split()[0]}")
print("Checking dependencies:")
pydicom = _try_import("pydicom")
PIL     = _try_import("PIL")
print()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT      = pathlib.Path("/home/teaching/Music/Nigam_51/Project_51/data")
PATIENT_PREFIX = "C"          # patient folders start with this letter
HDCT_DIR       = "HDCT"
LDCT_DIR       = "LDCT"
TIMING_SLICES  = 10           # how many slices to time in Section 7
HU_MIN, HU_MAX = -1200, 3200  # sane HU range for a CT scan

DIVIDER = "─" * 72

def section(title: str) -> None:
    """Print a bold section header."""
    print()
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_format(filepath: pathlib.Path) -> str:
    """Return a string tag for the file format based on extension."""
    ext = filepath.suffix.lower()
    if ext in (".dcm", ".ima"):
        return "dicom"
    if ext == ".npy":
        return "npy"
    if ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        return "image"
    return "unknown"


def load_file(filepath: pathlib.Path) -> tuple[np.ndarray, dict]:
    """
    Load one CT slice from any supported format.

    Returns
    -------
    arr  : float64 ndarray, always in HU (or as-stored for non-DICOM)
    meta : dict of format-specific metadata (empty for npy/image)
    """
    fmt  = detect_format(filepath)
    meta = {}

    if fmt == "dicom":
        if pydicom is None:
            raise ImportError("pydicom not installed — cannot read DICOM files")
        ds          = pydicom.dcmread(str(filepath))
        arr         = ds.pixel_array.astype(np.float64)
        slope       = float(getattr(ds, "RescaleSlope",     1))
        intercept   = float(getattr(ds, "RescaleIntercept", -1024))
        arr         = arr * slope + intercept          # → Hounsfield Units
        meta = {
            "RescaleSlope":    slope,
            "RescaleIntercept": intercept,
            "PixelSpacing":    getattr(ds, "PixelSpacing",    "N/A"),
            "SliceThickness":  getattr(ds, "SliceThickness",  "N/A"),
            "Modality":        getattr(ds, "Modality",        "N/A"),
            "PatientID":       getattr(ds, "PatientID",       "N/A"),
            "SliceLocation":   getattr(ds, "SliceLocation",   "N/A"),
        }

    elif fmt == "npy":
        arr = np.load(str(filepath)).astype(np.float64)

    elif fmt == "image":
        if PIL is None:
            raise ImportError("Pillow not installed — cannot read image files")
        from PIL import Image
        arr = np.array(Image.open(str(filepath))).astype(np.float64)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    return arr, meta


def print_array_stats(arr: np.ndarray, label: str = "") -> None:
    """Print shape / dtype / statistics and HU sanity flags."""
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}shape  = {arr.shape}")
    print(f"{prefix}dtype  = {arr.dtype}")
    print(f"{prefix}min    = {arr.min():.4f}")
    print(f"{prefix}max    = {arr.max():.4f}")
    print(f"{prefix}mean   = {arr.mean():.4f}")
    print(f"{prefix}std    = {arr.std():.4f}")

    # HU range sanity check
    in_hu = (arr.min() >= HU_MIN) and (arr.max() <= HU_MAX) and (arr.min() < -100)
    print(f"{prefix}HU range (min < -100, within [{HU_MIN},{HU_MAX}])? "
          f"{'YES' if in_hu else 'NO'}")

    # normalisation check
    in_01    = (0.0 <= arr.min()) and (arr.max() <= 1.0)
    in_0255  = (0.0 <= arr.min()) and (arr.max() <= 255.0) and arr.max() > 1.0
    if in_01:
        print(f"{prefix}Normalised to [0,1]?   YES")
    elif in_0255:
        print(f"{prefix}Normalised to [0,255]? YES (uint8-like)")
    else:
        print(f"{prefix}Normalised?            NO  (raw HU or other scale)")


def numeric_sort_key(p: pathlib.Path) -> int:
    """Extract the leading integer from a filename for correct slice ordering."""
    import re
    digits = re.findall(r"\d+", p.stem)
    return int(digits[-1]) if digits else 0


def list_slices(folder: pathlib.Path) -> list[pathlib.Path]:
    """Return all non-hidden files in folder, sorted numerically."""
    files = [f for f in folder.iterdir()
             if f.is_file() and not f.name.startswith(".")]
    return sorted(files, key=numeric_sort_key)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Patient inventory
# ══════════════════════════════════════════════════════════════════════════════

def section1_patient_inventory() -> list[pathlib.Path]:
    section("SECTION 1: Patient inventory")

    if not DATA_ROOT.exists():
        print(f"  ERROR: DATA_ROOT does not exist: {DATA_ROOT}")
        print("  Please update the DATA_ROOT variable at the top of this script.")
        sys.exit(1)

    candidates = sorted(
        [d for d in DATA_ROOT.iterdir()
         if d.is_dir() and d.name.startswith(PATIENT_PREFIX)],
        key=lambda d: d.name,
    )

    patients = []
    skipped  = []
    for d in candidates:
        has_hdct = (d / HDCT_DIR).is_dir()
        has_ldct = (d / LDCT_DIR).is_dir()
        if has_hdct and has_ldct:
            patients.append(d)
        else:
            skipped.append((d.name, has_hdct, has_ldct))

    # pretty-print in columns of 5
    names = [p.name for p in patients]
    for i in range(0, len(names), 5):
        print("  " + "   ".join(f"{n:<8}" for n in names[i:i+5]))

    print(f"\n  Total patients found : {len(patients)}")
    if skipped:
        print(f"  Skipped (missing subdir):")
        for name, h, l in skipped:
            print(f"    {name}  HDCT={'OK' if h else 'MISSING'}  "
                  f"LDCT={'OK' if l else 'MISSING'}")

    return patients

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — File format
# ══════════════════════════════════════════════════════════════════════════════

def section2_file_format(patients: list[pathlib.Path]) -> None:
    section("SECTION 2: File format")

    p = patients[0]
    print(f"  Inspecting first patient: {p.name}")

    for sub in (HDCT_DIR, LDCT_DIR):
        folder = p / sub
        files  = sorted(folder.iterdir(), key=lambda f: f.name)
        files  = [f for f in files if f.is_file() and not f.name.startswith(".")]

        exts   = {f.suffix.lower() for f in files}
        count  = len(files)
        names  = [f.name for f in files]

        print(f"\n  {sub}/ — {count} files   extensions: {exts}")

        first5 = names[:5]
        last5  = names[-5:] if count > 5 else []
        print(f"    First 5 : {' | '.join(first5)}")
        if last5:
            print(f"    Last  5 : {' | '.join(last5)}")

        # Detect dominant format
        if exts & {".dcm", ".ima"}:
            fmt_label = "DICOM (.dcm / .IMA)"
        elif ".npy" in exts:
            fmt_label = "NumPy array (.npy)"
        elif exts & {".png", ".tif", ".tiff", ".jpg"}:
            fmt_label = "Image file (.png / .tif / .jpg)"
        else:
            fmt_label = f"Unknown ({exts})"

        print(f"    Detected format : {fmt_label}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Single file content
# ══════════════════════════════════════════════════════════════════════════════

def section3_single_file(patients: list[pathlib.Path]) -> None:
    section("SECTION 3: Single file content")

    p    = patients[0]
    hdct = p / HDCT_DIR
    files = list_slices(hdct)

    if not files:
        print("  ERROR: No files found in HDCT/")
        return

    filepath = files[0]
    fmt      = detect_format(filepath)

    print(f"  Loading: {filepath.relative_to(DATA_ROOT)}")
    print(f"  Format : {fmt}")
    print()

    arr, meta = load_file(filepath)

    # DICOM-specific metadata
    if meta:
        print("  ── DICOM metadata ──────────────────────────────────────────────")
        for key, val in meta.items():
            print(f"  {key:<20}: {val}")
        print()

    print("  ── Array statistics ────────────────────────────────────────────")
    print_array_stats(arr)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HDCT vs LDCT comparison
# ══════════════════════════════════════════════════════════════════════════════

def section4_hdct_vs_ldct(patients: list[pathlib.Path]) -> None:
    section("SECTION 4: HDCT vs LDCT comparison")

    p     = patients[0]
    hfiles = list_slices(p / HDCT_DIR)
    lfiles = list_slices(p / LDCT_DIR)

    if not hfiles or not lfiles:
        print("  ERROR: one or both folders are empty")
        return

    hfile = hfiles[0]
    lfile = lfiles[0]

    print(f"  HDCT file : {hfile.name}")
    print(f"  LDCT file : {lfile.name}")
    print()

    harr, hmeta = load_file(hfile)
    larr, lmeta = load_file(lfile)

    print("  ── HDCT ─────────────────────────────────────────────────────────")
    print_array_stats(harr, "HDCT")
    print()
    print("  ── LDCT ─────────────────────────────────────────────────────────")
    print_array_stats(larr, "LDCT")
    print()

    same_shape = (harr.shape == larr.shape)
    names_match = (hfile.name == lfile.name)

    print(f"  Same shape?           {'YES' if same_shape else 'NO  ← WARNING'}")
    print(f"  Filenames match?      {'YES' if names_match else 'NO  (check ordering)'}")

    if same_shape:
        diff = np.abs(harr.astype(np.float64) - larr.astype(np.float64))
        print(f"  Mean abs diff (MAD) : {diff.mean():.4f} HU")
        print(f"  Max  abs diff       : {diff.max():.4f} HU")
        noise = (harr - larr)
        print(f"  Noise std (HDCT-LDCT): {noise.std():.4f} HU")
        snr = harr.std() / (noise.std() + 1e-8)
        print(f"  Approx SNR ratio    : {snr:.2f}  (HDCT std / noise std)")
    else:
        print("  Cannot compute diff — shapes differ")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Slice count per patient
# ══════════════════════════════════════════════════════════════════════════════

def section5_slice_counts(patients: list[pathlib.Path]) -> None:
    section("SECTION 5: Slice count per patient")

    header = f"  {'Patient':<12}| {'HDCT':>6} | {'LDCT':>6} | Paired?"
    print(header)
    print("  " + "─" * (len(header) - 2))

    hdct_counts = []
    ldct_counts = []
    unpaired    = []

    for p in patients:
        hf = list_slices(p / HDCT_DIR)
        lf = list_slices(p / LDCT_DIR)
        hc, lc = len(hf), len(lf)
        paired = (hc == lc)

        hdct_counts.append(hc)
        ldct_counts.append(lc)
        if not paired:
            unpaired.append(p.name)

        flag = "YES" if paired else "NO  ← MISMATCH"
        print(f"  {p.name:<12}| {hc:>6} | {lc:>6} | {flag}")

    print()
    all_counts = hdct_counts + ldct_counts
    total_hdct = sum(hdct_counts)
    total_ldct = sum(ldct_counts)
    print(f"  HDCT totals : min={min(hdct_counts)}  max={max(hdct_counts)}  "
          f"mean={np.mean(hdct_counts):.1f}  total={total_hdct}")
    print(f"  LDCT totals : min={min(ldct_counts)}  max={max(ldct_counts)}  "
          f"mean={np.mean(ldct_counts):.1f}  total={total_ldct}")

    if unpaired:
        print(f"\n  WARNING — {len(unpaired)} patients have mismatched counts: "
              f"{', '.join(unpaired)}")
    else:
        print("  All patients are fully paired (HDCT count == LDCT count). ✓")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Filename sorting pattern
# ══════════════════════════════════════════════════════════════════════════════

def section6_sorting(patients: list[pathlib.Path]) -> None:
    section("SECTION 6: Filename sorting pattern")

    p     = patients[0]
    hdct  = p / HDCT_DIR
    files = [f for f in hdct.iterdir() if f.is_file() and not f.name.startswith(".")]

    alpha_sorted   = sorted(files, key=lambda f: f.name)
    numeric_sorted = sorted(files, key=numeric_sort_key)

    n = 5
    alpha_names   = [f.name for f in alpha_sorted[:n]]
    numeric_names = [f.name for f in numeric_sorted[:n]]

    print(f"  First {n} files — alphabetical sort:")
    print(f"    {' | '.join(alpha_names)}")
    print(f"  First {n} files — numeric sort (extract digits):")
    print(f"    {' | '.join(numeric_names)}")

    same_order = (
        [f.name for f in alpha_sorted] == [f.name for f in numeric_sorted]
    )

    if same_order:
        print("\n  Orders are IDENTICAL — alphabetical sort is safe. ✓")
        print("  (filenames probably have zero-padded integers: 0001.dcm, 0002.dcm ...)")
    else:
        print("\n  ⚠  Orders DIFFER — use numeric sort in the dataset loader!")
        print("  (filenames probably lack zero-padding: 1.dcm, 2.dcm, 10.dcm ...)")
        print("  In the loader, use: sorted(files, key=lambda f: numeric_sort_key(f))")

    # Also check for duplicates / gaps
    keys = [numeric_sort_key(f) for f in numeric_sorted]
    if len(keys) != len(set(keys)):
        print("\n  ⚠  DUPLICATE slice indices detected! Check your data.")
    else:
        expected = list(range(keys[0], keys[0] + len(keys)))
        if keys != expected:
            gaps = sorted(set(expected) - set(keys))
            print(f"\n  ⚠  Gaps in slice indices: {gaps[:10]}"
                  f"{'...' if len(gaps) > 10 else ''}")
        else:
            print(f"  Slice indices are contiguous ({keys[0]} → {keys[-1]}). ✓")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — File size and loading speed
# ══════════════════════════════════════════════════════════════════════════════

def section7_loading_speed(patients: list[pathlib.Path]) -> None:
    section("SECTION 7: File size and loading speed")

    p     = patients[0]
    hdct  = p / HDCT_DIR
    files = list_slices(hdct)[:TIMING_SLICES]

    if not files:
        print("  ERROR: no files to time")
        return

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"  Timing {len(files)} consecutive HDCT slices...")
    print(f"  Total disk size of {len(files)} files: "
          f"{total_bytes / 1024:.1f} KB  "
          f"({total_bytes / 1024 / 1024:.2f} MB)")

    # warm-up (OS disk cache)
    _ = load_file(files[0])

    t0 = time.perf_counter()
    loaded = []
    for f in files:
        arr, _ = load_file(f)
        loaded.append(arr)
    elapsed = time.perf_counter() - t0

    per_file_ms = elapsed / len(files) * 1000
    total_files = len(list_slices(hdct))  # full HDCT count

    print(f"  Total time for {len(files)} files : {elapsed:.3f} s")
    print(f"  Per-file time             : {per_file_ms:.1f} ms")
    print(f"  Memory for {len(files)} arrays   : "
          f"{sum(a.nbytes for a in loaded) / 1024 / 1024:.1f} MB")
    proj = per_file_ms * total_files / 1000
    print(f"  Projected for all {total_files} slices: {proj:.1f} s per epoch "
          f"(single-threaded)")

    print()
    if per_file_ms < 10:
        print("  ✓ Very fast (<10 ms/file) — default DataLoader settings fine.")
    elif per_file_ms < 50:
        print("  ✓ Acceptable (10–50 ms/file) — use num_workers=4.")
    elif per_file_ms < 200:
        print("  ⚠ Slow (50–200 ms/file) — use num_workers=8 and pin_memory=True.")
    else:
        print("  ✗ Very slow (>200 ms/file) — convert to .npy on SSD before training!")
        print("    Run: python utils/convert_to_npy.py")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import datetime
    print("=" * 72)
    print("  explore_data.py — Anatomy-Aware CT Denoising Project")
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print(f"  Run start : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Section 1 must succeed (returns patient list for all other sections)
    patients = section1_patient_inventory()

    runners = [
        section2_file_format,
        section3_single_file,
        section4_hdct_vs_ldct,
        section5_slice_counts,
        section6_sorting,
        section7_loading_speed,
    ]

    for fn in runners:
        try:
            fn(patients)
        except Exception as exc:
            print(f"\n  [ERROR in {fn.__name__}]: {exc}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 72)
    print("  explore_data.py — COMPLETE")
    print(f"  Patients  : {len(patients)}")
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()