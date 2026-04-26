"""Generate pseudo-label masks from Mayo high-dose slices using TotalSegmentator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_totalsegmentator(repo_root: Path):
    """Import TotalSegmentator API with local-repo fallback."""
    local_totalseg = repo_root / "TotalSegmentator"
    if local_totalseg.exists():
        sys.path.insert(0, str(local_totalseg))
    from totalsegmentator.python_api import totalsegmentator  # pylint: disable=import-outside-toplevel

    return totalsegmentator


def stack_slices(patient_files: list[Path]) -> np.ndarray:
    """Stack sorted 2D slice files into a [N, 512, 512] volume."""
    slices = [np.load(str(path)).astype(np.float32) for path in sorted(patient_files)]
    volume = np.stack(slices, axis=0)
    if volume.shape[1:] != (512, 512):
        raise ValueError(f"Expected slices of shape (512, 512), got {volume.shape[1:]}")
    return volume


def remap_labels(mask_104: np.ndarray) -> np.ndarray:
    """Remap TotalSegmentator labels to 7 anatomy-aware classes."""
    remapped = np.zeros_like(mask_104, dtype=np.int64)

    cls1 = {1, 2}
    cls2 = {3, 4}
    cls3 = {5, 6}
    cls4 = {10, 11}
    cls5 = set(range(18, 26)) | set(range(66, 78))
    known = cls1 | cls2 | cls3 | cls4 | cls5

    remapped[np.isin(mask_104, list(cls1))] = 1
    remapped[np.isin(mask_104, list(cls2))] = 2
    remapped[np.isin(mask_104, list(cls3))] = 3
    remapped[np.isin(mask_104, list(cls4))] = 4
    remapped[np.isin(mask_104, list(cls5))] = 5
    remapped[(mask_104 > 0) & (~np.isin(mask_104, list(known)))] = 6
    return remapped


def process_patient(
    patient_id: str,
    slice_files: list[Path],
    mask_dir: Path,
    run_totalseg,
    save_raw_dir: Path | None = None,
) -> None:
    """Run pseudo-label generation for one patient and save per-slice masks."""
    (mask_dir / patient_id).mkdir(parents=True, exist_ok=True)
    volume = stack_slices(slice_files)

    if save_raw_dir is not None:
        save_raw_dir.mkdir(parents=True, exist_ok=True)
        input_nifti = save_raw_dir / f"{patient_id}_ndct_input.nii.gz"
        output_nifti = save_raw_dir / f"{patient_id}_totalseg_ml_output.nii.gz"
    else:
        input_nifti = mask_dir / f"{patient_id}_ndct_input.nii.gz"
        output_nifti = mask_dir / f"{patient_id}_totalseg_ml_output.nii.gz"

    sitk.WriteImage(sitk.GetImageFromArray(volume), str(input_nifti))
    # Ask TotalSegmentator for one multilabel volume file.
    run_totalseg(str(input_nifti), str(output_nifti), ml=True)

    read_path = output_nifti
    if output_nifti.is_dir():
        # Backward compatibility: some TotalSegmentator setups write a directory.
        candidates = [
            output_nifti / "segmentations.nii.gz",
            output_nifti / "multilabel.nii.gz",
        ]
        read_path = next((p for p in candidates if p.exists()), None)
        if read_path is None:
            nii_files = sorted(output_nifti.glob("*.nii.gz"))
            if len(nii_files) == 1:
                read_path = nii_files[0]
            else:
                raise RuntimeError(
                    f"TotalSegmentator output directory does not contain a unique NIfTI file: {output_nifti}"
                )

    seg_nii = sitk.ReadImage(str(read_path))
    seg_104 = sitk.GetArrayFromImage(seg_nii).astype(np.int64)
    seg_7 = remap_labels(seg_104)

    for src_path, seg_slice in zip(sorted(slice_files), seg_7):
        out_path = mask_dir / patient_id / src_path.name
        np.save(str(out_path), seg_slice.astype(np.int64))


def collect_patient_files(mayo_root: Path, high_subdir: str = "100%") -> dict[str, list[Path]]:
    """Group Mayo high-dose slice files by patient id."""
    patient_map: dict[str, list[Path]] = {}
    for patient_dir in sorted(p for p in mayo_root.iterdir() if p.is_dir()):
        hd_dir = patient_dir / high_subdir
        if not hd_dir.exists():
            continue
        files = sorted(hd_dir.glob("*.npy"))
        if files:
            patient_map[patient_dir.name] = files
    return patient_map


def main() -> None:
    """Parse CLI args and launch pseudo-label generation."""
    parser = argparse.ArgumentParser(description="Generate anatomy pseudo-labels.")
    parser.add_argument("--mayo_root", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, default=Path("data/masks_7cls"))
    parser.add_argument("--high_subdir", type=str, default="100%")
    parser.add_argument("--save_raw_dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Process only first patient.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_totalseg = load_totalsegmentator(repo_root)

    patient_files = collect_patient_files(args.mayo_root, args.high_subdir)
    patient_ids = sorted(patient_files.keys())
    if args.dry_run:
        patient_ids = patient_ids[:1]

    for patient_id in patient_ids:
        process_patient(
            patient_id=patient_id,
            slice_files=patient_files[patient_id],
            mask_dir=args.mask_dir,
            run_totalseg=run_totalseg,
            save_raw_dir=args.save_raw_dir,
        )


if __name__ == "__main__":
    main()

