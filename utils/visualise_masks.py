#!/usr/bin/env python3
"""
utils/visualise_masks.py

Sanity-check visualisation for TotalSegmentator pseudo-labels.
NOT required for training — visual confirmation only.

Usage
-----
# Interactive display:
python utils/visualise_masks.py --patient C002 --slice 100

# Save to file (headless servers / remote sessions):
python utils/visualise_masks.py --patient C002 --slice 100 --save mask_check.png

# Override default data paths:
python utils/visualise_masks.py --patient C002 --slice 100 \\
    --data_root  /my/data \\
    --masks_root /my/data/masks \\
    --save       /my/output/check.png

What to look for
----------------
  ✓  Lung regions      → cyan
  ✓  Bone (spine/ribs) → yellow
  ✓  Liver / spleen    → red
  ✓  Kidney            → orange
  ✓  Vessels / heart   → blue
  ✓  Soft tissue       → green
  ✓  Mask boundaries align with anatomy in the CT image
  ✓  HDCT and LDCT show the same anatomy (different noise texture only)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# RGBA colours for the 7 classes.
# Class 0 (background) is fully transparent so the CT image shows through.
COLORS = {
    0: (0.00, 0.00, 0.00, 0.00),   # background  — transparent
    1: (0.90, 0.22, 0.21, 0.60),   # liver/spleen — red      #E53935
    2: (0.98, 0.55, 0.00, 0.60),   # kidney       — orange   #FB8C00
    3: (0.12, 0.53, 0.90, 0.60),   # vessel/heart — blue     #1E88E5
    4: (0.00, 0.67, 0.76, 0.60),   # lung         — cyan     #00ACC1
    5: (0.99, 0.85, 0.21, 0.60),   # bone         — yellow   #FDD835
    6: (0.26, 0.63, 0.28, 0.60),   # soft tissue  — green    #43A047
}

CLASS_NAMES = [
    'background',   # 0
    'liver_spleen', # 1
    'kidney',       # 2
    'vessel',       # 3
    'lung',         # 4
    'bone',         # 5
    'soft_tissue',  # 6
]

# Soft-tissue window for display (HU)
HU_MIN = -200
HU_MAX =  400


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load one 2-D slice (handles DICOM, NumPy, PNG/TIFF)
# ─────────────────────────────────────────────────────────────────────────────
def _load_slice_hu(filepath: str) -> np.ndarray:
    """
    Load a single CT slice → float32 [H, W] in HU.
    Mirrors the logic in generate_masks.py so the two scripts stay consistent.
    """
    p = Path(filepath)
    suffix = p.suffix.lower()

    if suffix in ('.dcm', '.ima'):
        import pydicom
        ds  = pydicom.dcmread(str(p))
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, 'RescaleSlope',     1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', -1024.0))
        return arr * slope + intercept

    elif suffix == '.npy':
        arr = np.load(str(p)).astype(np.float32)
        # Detect normalised [0, 1] data
        if arr.min() >= -5.0 and arr.max() <= 1.05:
            arr = arr * 4000.0 - 1000.0
        return arr

    elif suffix in ('.png', '.tif', '.tiff'):
        from PIL import Image
        arr = np.array(Image.open(str(p))).astype(np.float32)
        return arr - 1000.0          # 16-bit HU + 1000 convention

    else:
        raise ValueError(f"Unsupported slice format: '{suffix}' ({p})")


def _sorted_slice_paths(folder: Path) -> list:
    """Return slice paths in numeric order (same logic as generate_masks.py)."""
    SUPPORTED = {'.dcm', '.ima', '.npy', '.png', '.tif', '.tiff'}
    paths = [f for f in folder.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED]
    if not paths:
        raise FileNotFoundError(f"No slice files found in: {folder}")

    def _num_key(f: Path) -> int:
        digits = ''.join(filter(str.isdigit, f.stem))
        return int(digits) if digits else 0

    paths.sort(key=_num_key)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build RGBA overlay image from integer mask
# ─────────────────────────────────────────────────────────────────────────────
def _mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D integer mask [H, W] with values 0-6
    into an RGBA image [H, W, 4] using the COLORS look-up table.
    """
    H, W   = mask.shape
    rgba   = np.zeros((H, W, 4), dtype=np.float32)

    for cls_id, color in COLORS.items():
        where = (mask == cls_id)
        rgba[where, 0] = color[0]   # R
        rgba[where, 1] = color[1]   # G
        rgba[where, 2] = color[2]   # B
        rgba[where, 3] = color[3]   # A

    return rgba


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build the legend patches
# ─────────────────────────────────────────────────────────────────────────────
def _make_legend_patches(classes_present: set) -> list:
    """
    Return a list of matplotlib Patch objects for the legend.
    Only includes classes actually present in the mask (skip empty classes).
    Background is always excluded from the legend (it is transparent).
    """
    patches = []
    for cls_id in range(1, 7):          # skip background (0)
        if cls_id not in classes_present:
            continue
        r, g, b, _ = COLORS[cls_id]
        patch = mpatches.Patch(
            facecolor=(r, g, b),
            edgecolor='white',
            linewidth=0.5,
            alpha=0.85,
            label=f"{cls_id}: {CLASS_NAMES[cls_id]}",
        )
        patches.append(patch)
    return patches


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def visualise_slice(
    patient_id: str,
    slice_idx:  int,
    data_root:  str = '/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root: str = '/home/teaching/Music/Nigam_51/Project_51/data/masks',
    save_path:  str = None,
):
    """
    Create a 3-panel figure for one patient / slice combination.

    Panel 1 — HDCT slice          (grayscale, soft-tissue window)
    Panel 2 — LDCT slice          (grayscale, soft-tissue window)
    Panel 3 — Mask overlaid on HDCT (colour-coded RGBA overlay)

    Parameters
    ----------
    patient_id : str   e.g. 'C002'
    slice_idx  : int   0-based slice index
    data_root  : str   root that contains {PATIENT}/HDCT/ and {PATIENT}/LDCT/
    masks_root : str   root that contains {PATIENT}/{IDX:04d}.npy
    save_path  : str | None
        If given, save the figure to this path instead of calling plt.show().
        Useful on headless servers.  Supports any matplotlib-supported format
        (.png, .pdf, .svg, …).
    """
    data_root  = Path(data_root)
    masks_root = Path(masks_root)

    # ── 1. Resolve file paths ─────────────────────────────────────────────
    hdct_dir = data_root  / patient_id / 'HDCT'
    ldct_dir = data_root  / patient_id / 'LDCT'
    mask_dir = masks_root / patient_id

    # Validate directories
    missing = []
    if not hdct_dir.exists():
        missing.append(f"HDCT dir  : {hdct_dir}")
    if not ldct_dir.exists():
        missing.append(f"LDCT dir  : {ldct_dir}")
    if not mask_dir.exists():
        missing.append(f"Mask dir  : {mask_dir}")
    if missing:
        raise FileNotFoundError(
            "The following paths do not exist:\n  " + "\n  ".join(missing)
        )

    # Get sorted slice lists
    hdct_paths = _sorted_slice_paths(hdct_dir)
    ldct_paths = _sorted_slice_paths(ldct_dir)

    n_hdct = len(hdct_paths)
    n_ldct = len(ldct_paths)

    # Validate slice index
    max_slice = min(n_hdct, n_ldct) - 1
    if not (0 <= slice_idx <= max_slice):
        raise IndexError(
            f"slice_idx={slice_idx} is out of range. "
            f"Patient {patient_id} has {n_hdct} HDCT / {n_ldct} LDCT slices "
            f"(valid range: 0–{max_slice})."
        )

    # Mask file: {masks_root}/{patient_id}/{slice_idx:04d}.npy
    mask_file = mask_dir / f'{slice_idx:04d}.npy'
    if not mask_file.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_file}\n"
            f"Have you run generate_masks.py for patient {patient_id}?"
        )

    # ── 2. Load data ──────────────────────────────────────────────────────
    print(f"Loading HDCT slice {slice_idx} from {hdct_paths[slice_idx]}")
    hdct_slice = _load_slice_hu(str(hdct_paths[slice_idx]))   # [H, W] float32

    print(f"Loading LDCT slice {slice_idx} from {ldct_paths[slice_idx]}")
    ldct_slice = _load_slice_hu(str(ldct_paths[slice_idx]))   # [H, W] float32

    print(f"Loading mask from {mask_file}")
    mask = np.load(str(mask_file)).astype(np.int8)             # [H, W] int8

    # ── 3. Clip HDCT / LDCT to soft-tissue window ─────────────────────────
    hdct_display = np.clip(hdct_slice, HU_MIN, HU_MAX)
    ldct_display = np.clip(ldct_slice, HU_MIN, HU_MAX)

    # ── 4. Build RGBA overlay ─────────────────────────────────────────────
    overlay_rgba = _mask_to_rgba(mask)                          # [H, W, 4]

    # ── 5. Collect classes present in this slice (for the legend) ─────────
    classes_present = set(np.unique(mask).tolist())
    legend_patches  = _make_legend_patches(classes_present)

    # ── 6. Compute per-class pixel percentages for the info box ───────────
    total_pixels = mask.size
    class_stats = {}
    for cls_id in range(7):
        cnt = int(np.sum(mask == cls_id))
        pct = 100.0 * cnt / total_pixels
        class_stats[cls_id] = (cnt, pct)

    # ── 7. Build figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        facecolor='#1a1a1a',
    )
    fig.suptitle(
        f"Patient: {patient_id}   |   Slice: {slice_idx}   |   "
        f"HU window: [{HU_MIN}, {HU_MAX}]",
        color='white',
        fontsize=13,
        fontweight='bold',
        y=1.01,
    )

    _common_imshow_kwargs = dict(
        cmap='gray',
        vmin=HU_MIN,
        vmax=HU_MAX,
        interpolation='nearest',
        aspect='equal',
    )

    # ── Panel 1: HDCT ─────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.imshow(hdct_display, **_common_imshow_kwargs)
    ax1.set_title(
        'HDCT (clean)',
        color='white', fontsize=11, pad=6, fontweight='semibold',
    )
    _style_axis(ax1)
    _add_stats_box(
        ax1,
        lines=[
            f"HU min : {hdct_slice.min():.0f}",
            f"HU max : {hdct_slice.max():.0f}",
            f"Shape  : {hdct_slice.shape[0]}×{hdct_slice.shape[1]}",
        ],
    )

    # ── Panel 2: LDCT ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.imshow(ldct_display, **_common_imshow_kwargs)
    ax2.set_title(
        'LDCT (noisy)',
        color='white', fontsize=11, pad=6, fontweight='semibold',
    )
    _style_axis(ax2)
    _add_stats_box(
        ax2,
        lines=[
            f"HU min : {ldct_slice.min():.0f}",
            f"HU max : {ldct_slice.max():.0f}",
            f"Shape  : {ldct_slice.shape[0]}×{ldct_slice.shape[1]}",
        ],
    )

    # ── Panel 3: Mask overlay on HDCT ─────────────────────────────────────
    ax3 = axes[2]
    ax3.imshow(hdct_display, **_common_imshow_kwargs)
    ax3.imshow(overlay_rgba, interpolation='nearest', aspect='equal')
    ax3.set_title(
        'Pseudo-label mask (overlay on HDCT)',
        color='white', fontsize=11, pad=6, fontweight='semibold',
    )
    _style_axis(ax3)

    # Legend inside panel 3
    if legend_patches:
        leg = ax3.legend(
            handles=legend_patches,
            loc='lower right',
            fontsize=7.5,
            framealpha=0.75,
            facecolor='#111111',
            edgecolor='#555555',
            labelcolor='white',
            title='Classes present',
            title_fontsize=8,
        )
        leg.get_title().set_color('white')

    # Class-distribution text box inside panel 3 (top-left)
    dist_lines = []
    for cls_id in range(7):
        cnt, pct = class_stats[cls_id]
        if pct > 0.01:                    # skip truly-empty classes
            dist_lines.append(f"{CLASS_NAMES[cls_id]:14s} {pct:5.1f}%")
    if dist_lines:
        dist_text = '\n'.join(dist_lines)
        ax3.text(
            0.01, 0.99, dist_text,
            transform=ax3.transAxes,
            va='top', ha='left',
            fontsize=6.5,
            fontfamily='monospace',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='#111111',
                edgecolor='#444444',
                alpha=0.80,
            ),
        )

    plt.tight_layout(pad=1.5)

    # ── 8. Output ─────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\n✓ Figure saved to: {save_path.resolve()}")
        plt.close(fig)
    else:
        print("\nDisplaying figure …  (close the window to exit)")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _style_axis(ax):
    """Apply a dark-theme style to an axis."""
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(0.8)


def _add_stats_box(ax, lines: list):
    """Add a small monospace text box with image statistics."""
    text = '\n'.join(lines)
    ax.text(
        0.01, 0.01, text,
        transform=ax.transAxes,
        va='bottom', ha='left',
        fontsize=7,
        fontfamily='monospace',
        color='#dddddd',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='#111111',
            edgecolor='#444444',
            alpha=0.80,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE: visualise multiple slices at once
# ─────────────────────────────────────────────────────────────────────────────
def visualise_multiple_slices(
    patient_id:  str,
    slice_indices: list,
    data_root:   str = '/home/teaching/Music/Nigam_51/Project_51/data',
    masks_root:  str = '/home/teaching/Music/Nigam_51/Project_51/data/masks',
    save_dir:    str = None,
):
    """
    Convenience wrapper: generate one figure per slice index.

    If save_dir is given, each figure is saved as:
        {save_dir}/{patient_id}_slice{idx:04d}.png

    Parameters
    ----------
    slice_indices : list[int]  e.g. [0, 50, 100, 150, 200]
    """
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    for idx in slice_indices:
        print(f"\n── Slice {idx} ──────────────────────────────────────────")
        sp = None
        if save_dir_path:
            sp = str(save_dir_path / f"{patient_id}_slice{idx:04d}.png")
        try:
            visualise_slice(
                patient_id  = patient_id,
                slice_idx   = idx,
                data_root   = data_root,
                masks_root  = masks_root,
                save_path   = sp,
            )
        except Exception as exc:
            print(f"  ✗ Failed for slice {idx}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Sanity-check visualisation for TotalSegmentator pseudo-labels.\n'
            'Creates a 3-panel figure: HDCT | LDCT | Mask overlay.\n\n'
            'What to confirm:\n'
            '  ✓  Lung    → cyan\n'
            '  ✓  Bone    → yellow\n'
            '  ✓  Liver   → red\n'
            '  ✓  Kidney  → orange\n'
            '  ✓  Vessels → blue\n'
            '  ✓  Mask boundaries align with CT anatomy\n'
            '  ✓  HDCT and LDCT show the same anatomy (different noise only)\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--patient',
        default='C002',
        metavar='PATIENT_ID',
        help='Patient ID to visualise (default: C002)',
    )
    parser.add_argument(
        '--slice',
        type=int,
        default=100,
        metavar='IDX',
        help='0-based slice index to display (default: 100)',
    )
    parser.add_argument(
        '--slices',
        type=int,
        nargs='+',
        default=None,
        metavar='IDX',
        help=(
            'Multiple slice indices to visualise, e.g. --slices 50 100 150. '
            'Overrides --slice. Requires --save_dir.'
        ),
    )
    parser.add_argument(
        '--save',
        default=None,
        metavar='FILE',
        help=(
            'Save figure to this file instead of showing interactively. '
            'Supports .png, .pdf, .svg, etc. '
            'Example: --save mask_check.png'
        ),
    )
    parser.add_argument(
        '--save_dir',
        default=None,
        metavar='DIR',
        help=(
            'When using --slices, save each figure into this directory. '
            'Files are named {PATIENT}_slice{IDX:04d}.png'
        ),
    )
    parser.add_argument(
        '--data_root',
        default='/home/teaching/Music/Nigam_51/Project_51/data',
        metavar='PATH',
        help='Root data directory (default: %(default)s)',
    )
    parser.add_argument(
        '--masks_root',
        default='/home/teaching/Music/Nigam_51/Project_51/data/masks',
        metavar='PATH',
        help='Root masks directory (default: %(default)s)',
    )

    args = parser.parse_args()

    # ── Batch mode ────────────────────────────────────────────────────────
    if args.slices is not None:
        visualise_multiple_slices(
            patient_id    = args.patient,
            slice_indices = args.slices,
            data_root     = args.data_root,
            masks_root    = args.masks_root,
            save_dir      = args.save_dir,
        )

    # ── Single-slice mode ─────────────────────────────────────────────────
    else:
        try:
            visualise_slice(
                patient_id = args.patient,
                slice_idx  = args.slice,
                data_root  = args.data_root,
                masks_root = args.masks_root,
                save_path  = args.save,
            )
        except FileNotFoundError as exc:
            print(f"\n✗ File not found:\n  {exc}", file=sys.stderr)
            sys.exit(1)
        except IndexError as exc:
            print(f"\n✗ Index error:\n  {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"\n✗ Unexpected error:\n  {exc}", file=sys.stderr)
            raise
