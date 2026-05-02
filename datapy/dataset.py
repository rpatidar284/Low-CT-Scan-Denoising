
```python
"""
data/dataset.py

CT Dataset Loader for Stage 1 Training.
========================================
Provides paired HDCT/LDCT slice loading with:
  - Automatic file-format detection (DICOM / .npy / PNG / TIFF)
  - Patient-level train/val/test splits (no data leakage)
  - HU normalisation to [0, 1]
  - Consistent augmentation across ndct / ldct / mask
  - Graceful fallback to zero masks when TotalSegmentator labels are absent

Directory layout expected on disk
----------------------------------
  data/
    C002/
      HDCT/   ← high-dose slices  (mapped to 'ndct' in code)
      LDCT/   ← low-dose slices   (mapped to 'ldct' in code)
    C004/
      HDCT/
      LDCT/
    ...
  data/masks/           ← created by TotalSegmentator pipeline
    C002/
      0000.npy          ← int64 [H, W] with labels 0-6
      0001.npy
      ...

Naming convention
-----------------
  HDCT folder → 'ndct' in all variable / key names  (clean reference)
  LDCT folder → 'ldct' in all variable / key names  (noisy input)
"""

import os
import re
import random
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# File-format utilities
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(folder: str) -> str:
    """
    Inspect *folder* and return the dominant file extension.

    Preference order: .dcm / .IMA  >  .npy  >  .png  >  .tiff / .tif

    Returns one of: 'dicom', 'npy', 'png', 'tiff'.
    Raises FileNotFoundError when the folder is empty or unrecognised.
    """
    folder = Path(folder)
    counts: Dict[str, int] = {}
    for p in folder.iterdir():
        ext = p.suffix.lower()
        if ext in ('.dcm', '.ima'):
            counts['dicom'] = counts.get('dicom', 0) + 1
        elif ext == '.npy':
            counts['npy'] = counts.get('npy', 0) + 1
        elif ext == '.png':
            counts['png'] = counts.get('png', 0) + 1
        elif ext in ('.tiff', '.tif'):
            counts['tiff'] = counts.get('tiff', 0) + 1

    if not counts:
        raise FileNotFoundError(
            f"No recognised image files found in {folder}. "
            f"Expected .dcm, .IMA, .npy, .png, or .tiff/.tif."
        )

    # Return whichever format appears most often
    return max(counts, key=counts.__getitem__)


def load_slice(filepath: str) -> np.ndarray:
    """
    Load one CT slice and return a 2-D float32 ndarray in Hounsfield Units
    (or the native float range for non-DICOM formats).

    Format is detected automatically from the file extension so that the
    dataset works without any manual configuration regardless of whether
    the data was stored as DICOM, NumPy, PNG, or TIFF.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the slice file.

    Returns
    -------
    arr : np.ndarray  shape [H, W], dtype float32

    Branches
    --------
    A  .dcm / .IMA  — pydicom + rescale to HU
    B  .npy         — np.load, cast to float32
    C  .png         — PIL.Image, cast to float32
    D  .tiff/.tif   — PIL.Image, cast to float32
    """
    ext = Path(filepath).suffix.lower()

    # ── Branch A: DICOM ───────────────────────────────────────────────────
    if ext in ('.dcm', '.ima'):
        import pydicom  # type: ignore
        ds  = pydicom.dcmread(filepath)
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, 'RescaleSlope',     1))
        intercept = float(getattr(ds, 'RescaleIntercept', -1024))
        arr = arr * slope + intercept   # now in Hounsfield Units
        return arr

    # ── Branch B: NumPy ───────────────────────────────────────────────────
    if ext == '.npy':
        return np.load(filepath).astype(np.float32)

    # ── Branch C / D: PIL raster ──────────────────────────────────────────
    if ext in ('.png', '.tiff', '.tif'):
        from PIL import Image  # type: ignore
        return np.array(Image.open(filepath)).astype(np.float32)

    raise ValueError(
        f"Unsupported file extension '{ext}' for slice at {filepath}. "
        f"Supported: .dcm, .IMA, .npy, .png, .tiff, .tif"
    )


def get_sorted_slice_files(folder: str) -> List[str]:
    """
    Return a **numerically** sorted list of all slice file paths in *folder*.

    Numeric sort prevents the common lexicographic ordering bug where
    '10.dcm' < '2.dcm'.  Files are sorted by the integer embedded in
    the stem (e.g. '0042' → 42).  If no digits are found the raw stem
    is used as a fallback key so the function never crashes.

    Parameters
    ----------
    folder : str
        Directory containing the slice files.

    Returns
    -------
    List[str]
        Absolute file paths, sorted by slice order.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Slice folder not found: {folder}")

    # Collect all files (skip hidden files and sub-directories)
    supported = {'.dcm', '.ima', '.npy', '.png', '.tiff', '.tif'}
    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and not p.name.startswith('.')
        and p.suffix.lower() in supported
    ]

    def _sort_key(p: Path):
        digits = re.sub(r'\D', '', p.stem)   # strip all non-digits
        return int(digits) if digits else p.stem

    files.sort(key=_sort_key)
    return [str(f) for f in files]


# ─────────────────────────────────────────────────────────────────────────────
# CTSliceDataset
# ─────────────────────────────────────────────────────────────────────────────

class CTSliceDataset(Dataset):
    """
    Paired HDCT / LDCT slice dataset for Stage 1 VM-UNet training.

    Maps HDCT → 'ndct' (clean reference) and LDCT → 'ldct' (noisy input).

    Each sample contains
    --------------------
    'ndct'       : [1, H, W] float32  normalised to [0, 1]
    'ldct'       : [1, H, W] float32  normalised to [0, 1]
    'mask'       : [H, W]    int64    organ labels 0-6  (zeros if absent)
    'patient_id' : str                e.g. 'C002'
    'slice_idx'  : int                0-based slice index within patient
    'hdct_path'  : str                full path to HDCT file
    'ldct_path'  : str                full path to LDCT file

    Normalisation
    -------------
    Clip to [HU_MIN, HU_MAX] = [-1000, 3000] then linearly scale to [0, 1]:
        x_norm = (clip(x, -1000, 3000) − (−1000)) / (3000 − (−1000))

    If your Step-11A audit showed the raw data is already in [0, 1] or
    [0, 255], override HU_MIN / HU_MAX accordingly.

    Split strategy
    --------------
    Patient-level split (prevents data leakage between folds):
        train = first  80 % of patients (sorted alphabetically)
        val   = next   10 %
        test  = last   10 %

    Augmentation (training only)
    ----------------------------
    Both flips are applied to ndct + ldct + mask identically:
        Random horizontal flip  p = 0.5
        Random vertical flip    p = 0.5

    Parameters
    ----------
    data_root   : str   Root directory containing patient folders.
    masks_root  : str   Directory with TotalSegmentator masks.
                        Pass None or a non-existent path to use zero masks.
    split       : str   'train', 'val', or 'test'.
    split_ratio : tuple (train_frac, val_frac, test_frac). Default (0.8, 0.1, 0.1).
    seed        : int   Random seed for reproducible splits. Default 42.
    augment     : bool  Enable augmentation (only active for 'train' split).
    """

    DATA_ROOT  = '/home/teaching/Music/Nigam_51/Project_51/data'
    MASKS_ROOT = '/home/teaching/Music/Nigam_51/Project_51/data/masks'

    HU_MIN = -1000.0
    HU_MAX =  3000.0

    CLASS_NAMES = [
        'background', 'liver_spleen', 'kidney', 'vessel',
        'lung', 'bone', 'soft_tissue',
    ]

    def __init__(
        self,
        data_root:   str   = DATA_ROOT,
        masks_root:  str   = MASKS_ROOT,
        split:       str   = 'train',
        split_ratio: Tuple = (0.8, 0.1, 0.1),
        seed:        int   = 42,
        augment:     bool  = True,
    ):
        super().__init__()

        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test'; got '{split}'."
        assert abs(sum(split_ratio) - 1.0) < 1e-6, \
            f"split_ratio must sum to 1.0; got {split_ratio}."

        self.data_root  = Path(data_root)
        self.masks_root = Path(masks_root) if masks_root else None
        self.split      = split
        self.augment    = augment and (split == 'train')
        self.seed       = seed

        # ── Discover patient folders ──────────────────────────────────────
        all_patients: List[str] = sorted([
            p.name
            for p in self.data_root.iterdir()
            if p.is_dir()
            and re.match(r'^C\d+', p.name)          # C002, C004, …
            and (p / 'HDCT').exists()
            and (p / 'LDCT').exists()
        ])

        if len(all_patients) == 0:
            raise FileNotFoundError(
                f"No valid patient folders (C###/HDCT + C###/LDCT) found "
                f"under {data_root}."
            )

        # ── Patient-level split ───────────────────────────────────────────
        n       = len(all_patients)
        n_train = int(n * split_ratio[0])
        n_val   = max(1, int(n * split_ratio[1]))  # at least 1 patient per fold
        # Guard: don't overshoot when n is small
        n_train = min(n_train, n - n_val - 1)
        n_train = max(n_train, 1)

        if split == 'train':
            self.patients = all_patients[:n_train]
        elif split == 'val':
            self.patients = all_patients[n_train: n_train + n_val]
        else:  # test
            self.patients = all_patients[n_train + n_val:]

        # ── Build flat sample index ───────────────────────────────────────
        # Each entry: (patient_id, slice_idx, hdct_path, ldct_path)
        self.samples: List[Tuple[str, int, str, str]] = []

        for patient in self.patients:
            hdct_dir = str(self.data_root / patient / 'HDCT')
            ldct_dir = str(self.data_root / patient / 'LDCT')

            hdct_files = get_sorted_slice_files(hdct_dir)
            ldct_files = get_sorted_slice_files(ldct_dir)

            if len(hdct_files) == 0:
                raise FileNotFoundError(
                    f"Patient {patient}: HDCT folder is empty ({hdct_dir})."
                )

            # ── Pairing strategy ──────────────────────────────────────────
            # Primary: match by filename stem (robust when slice counts differ).
            # Fallback: positional pairing (assumes same count & order).
            hdct_by_stem = {Path(f).stem: f for f in hdct_files}
            ldct_by_stem = {Path(f).stem: f for f in ldct_files}
            common_stems = sorted(
                hdct_by_stem.keys() & ldct_by_stem.keys(),
                key=lambda s: int(re.sub(r'\D', '', s)) if re.search(r'\d', s) else s,
            )

            if common_stems:
                # Filename-matched pairing
                pairs = [
                    (hdct_by_stem[s], ldct_by_stem[s])
                    for s in common_stems
                ]
            else:
                # Positional pairing — warn if counts differ
                if len(hdct_files) != len(ldct_files):
                    raise AssertionError(
                        f"Patient {patient}: HDCT has {len(hdct_files)} slices "
                        f"but LDCT has {len(ldct_files)} — cannot pair by "
                        f"position. Check data integrity."
                    )
                pairs = list(zip(hdct_files, ldct_files))

            for idx, (h_path, l_path) in enumerate(pairs):
                self.samples.append((patient, idx, h_path, l_path))

        print(
            f"[CTSliceDataset] split={split:5s} | "
            f"patients={len(self.patients):3d} | "
            f"slices={len(self.samples):5d}"
        )

    # ── Normalisation ─────────────────────────────────────────────────────

    def _normalize(self, arr: np.ndarray) -> torch.Tensor:
        """
        Clip to [HU_MIN, HU_MAX] and linearly scale to [0, 1].

        Returns
        -------
        Tensor  [1, H, W]  float32
        """
        arr = np.clip(arr, self.HU_MIN, self.HU_MAX)
        arr = (arr - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        return torch.from_numpy(arr).float().unsqueeze(0)   # [1, H, W]

    # ── Mask loading ──────────────────────────────────────────────────────

    def _load_mask(
        self,
        patient_id: str,
        slice_idx:  int,
        h:          int,
        w:          int,
    ) -> torch.Tensor:
        """
        Load the TotalSegmentator organ mask for (patient_id, slice_idx).

        Returns a zero tensor of shape [H, W] if the mask file does not exist,
        allowing the training loop to run before segmentation labels are ready.

        Returns
        -------
        Tensor  [H, W]  int64  with values in {0, 1, 2, 3, 4, 5, 6}
        """
        if self.masks_root is not None and self.masks_root.exists():
            mask_path = self.masks_root / patient_id / f'{slice_idx:04d}.npy'
            if mask_path.exists():
                mask = np.load(str(mask_path)).astype(np.int64)
                return torch.from_numpy(mask)   # [H, W]

        return torch.zeros(h, w, dtype=torch.int64)

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        patient_id, slice_idx, hdct_path, ldct_path = self.samples[idx]

        # ── Load raw slices ───────────────────────────────────────────────
        hdct_arr = load_slice(hdct_path)   # [H, W] float32
        ldct_arr = load_slice(ldct_path)   # [H, W] float32

        # ── Normalise → tensors ───────────────────────────────────────────
        ndct = self._normalize(hdct_arr)   # [1, H, W]
        ldct = self._normalize(ldct_arr)   # [1, H, W]

        h, w = ndct.shape[1], ndct.shape[2]

        # ── Load (or synthesise) mask ─────────────────────────────────────
        mask = self._load_mask(patient_id, slice_idx, h, w)   # [H, W]

        # ── Augmentation (training only) ──────────────────────────────────
        # Use a per-sample RNG so ndct / ldct / mask receive identical flips.
        if self.augment:
            rng = random.Random(self.seed ^ idx)   # deterministic per (seed, idx)

            if rng.random() < 0.5:
                ndct = torch.flip(ndct, dims=[-1])
                ldct = torch.flip(ldct, dims=[-1])
                mask = torch.flip(mask, dims=[-1])

            if rng.random() < 0.5:
                ndct = torch.flip(ndct, dims=[-2])
                ldct = torch.flip(ldct, dims=[-2])
                mask = torch.flip(mask, dims=[-2])

        return {
            'ndct':       ndct,          # [1, H, W]  float32  [0, 1]
            'ldct':       ldct,          # [1, H, W]  float32  [0, 1]
            'mask':       mask,          # [H, W]     int64    0-6
            'patient_id': patient_id,
            'slice_idx':  slice_idx,
            'hdct_path':  hdct_path,
            'ldct_path':  ldct_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# create_dataloaders — convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_root:   str = CTSliceDataset.DATA_ROOT,
    masks_root:  str = CTSliceDataset.MASKS_ROOT,
    batch_size:  int = 8,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create train / val / test DataLoaders for the CT data.

    Parameters
    ----------
    data_root   : str  Root directory containing patient folders.
    masks_root  : str  Directory with TotalSegmentator masks.
    batch_size  : int  Mini-batch size.  Default 8.
    num_workers : int  DataLoader worker processes.  Default 4.

    Returns
    -------
    dict  {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    loaders: Dict[str, DataLoader] = {}

    for split in ('train', 'val', 'test'):
        ds = CTSliceDataset(
            data_root  = data_root,
            masks_root = masks_root,
            split      = split,
            augment    = (split == 'train'),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = (split == 'train'),
            # Avoid CUDA multi-processing issues with some DICOM libraries
            persistent_workers = (num_workers > 0),
        )

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# DummyCTDataset — for unit tests and CI without real data
# ─────────────────────────────────────────────────────────────────────────────

class DummyCTDataset(Dataset):
    """
    Generates random tensors in the correct format for testing the training
    loop before real data is available.  Identical interface to CTSliceDataset.

    Parameters
    ----------
    length     : int  Number of samples.  Default 100.
    image_size : int  Spatial resolution (square).  Default 512.
    num_classes: int  Number of organ classes.  Default 7.
    """

    def __init__(
        self,
        length:      int = 100,
        image_size:  int = 512,
        num_classes: int = 7,
    ):
        self.length      = length
        self.image_size  = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        H = W = self.image_size
        return {
            'ndct':       torch.rand(1, H, W),                              # [0,1]
            'ldct':       torch.rand(1, H, W),
            'mask':       torch.randint(0, self.num_classes, (H, W)),
            'patient_id': f'dummy_{idx // 50:03d}',
            'slice_idx':  idx % 50,
            'hdct_path':  '',
            'ldct_path':  '',
        }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from pathlib import Path

    print("=" * 60)
    print("data/dataset.py — self-test")
    print("=" * 60)

    # ── Test 1: DummyCTDataset ────────────────────────────────────────────
    print("\n── DummyCTDataset ────────────────────────────────────────────")
    dummy = DummyCTDataset(length=16, image_size=64)
    loader = DataLoader(dummy, batch_size=4, num_workers=0)
    batch  = next(iter(loader))

    assert batch['ndct'].shape == (4, 1, 64, 64), batch['ndct'].shape
    assert batch['ldct'].shape == (4, 1, 64, 64), batch['ldct'].shape
    assert batch['mask'].shape == (4, 64, 64),     batch['mask'].shape
    assert batch['ndct'].dtype == torch.float32
    assert batch['ldct'].dtype == torch.float32
    assert batch['mask'].dtype == torch.int64

    # Values in [0, 1]
    assert batch['ndct'].min() >= 0.0 and batch['ndct'].max() <= 1.0
    assert batch['ldct'].min() >= 0.0 and batch['ldct'].max() <= 1.0

    # Mask values in valid range
    assert batch['mask'].min() >= 0
    assert batch['mask'].max() < 7

    print(f"  ndct  : {list(batch['ndct'].shape)}  dtype={batch['ndct'].dtype}  "
          f"range=[{batch['ndct'].min():.2f}, {batch['ndct'].max():.2f}]  ✓")
    print(f"  ldct  : {list(batch['ldct'].shape)}  ✓")
    print(f"  mask  : {list(batch['mask'].shape)}  dtype={batch['mask'].dtype}  ✓")
    print("DummyCTDataset: PASSED")

    # ── Test 2: Real CTSliceDataset ───────────────────────────────────────
    real_root = CTSliceDataset.DATA_ROOT
    print(f"\n── CTSliceDataset (real data at {real_root}) ─────────────────")

    if not Path(real_root).exists():
        print(f"  Real data not found — skipping real-data tests.")
        print("  (Run on the machine with CT data to validate CTSliceDataset.)")
    else:
        # ── 2a. Discover patients and splits ─────────────────────────────
        train_ds = CTSliceDataset(data_root=real_root, split='train')
        val_ds   = CTSliceDataset(data_root=real_root, split='val')
        test_ds  = CTSliceDataset(data_root=real_root, split='test')

        total = len(train_ds) + len(val_ds) + len(test_ds)
        print(f"\n  Split summary :")
        print(f"    train : {len(train_ds):5d} slices "
              f"({len(train_ds.patients)} patients)")
        print(f"    val   : {len(val_ds):5d} slices "
              f"({len(val_ds.patients)} patients)")
        print(f"    test  : {len(test_ds):5d} slices "
              f"({len(test_ds.patients)} patients)")
        print(f"    total : {total:5d} slices")

        # No patient should appear in two splits
        train_p = set(train_ds.patients)
        val_p   = set(val_ds.patients)
        test_p  = set(test_ds.patients)
        assert len(train_p & val_p)  == 0, "Train/val overlap!"
        assert len(train_p & test_p) == 0, "Train/test overlap!"
        assert len(val_p   & test_p) == 0, "Val/test overlap!"
        print(f"    No patient overlap across splits  ✓")

        # ── 2b. Single-sample correctness ────────────────────────────────
        print("\n  Single-sample check (index 0):")
        sample = train_ds[0]

        assert 'ndct'       in sample
        assert 'ldct'       in sample
        assert 'mask'       in sample
        assert 'patient_id' in sample
        assert 'slice_idx'  in sample
        assert 'hdct_path'  in sample
        assert 'ldct_path'  in sample

        ndct = sample['ndct']
        ldct = sample['ldct']
        mask = sample['mask']

        assert ndct.dim() == 3 and ndct.shape[0] == 1, \
            f"ndct should be [1,H,W], got {ndct.shape}"
        assert ldct.shape == ndct.shape, \
            f"ldct/ndct shape mismatch: {ldct.shape} vs {ndct.shape}"
        assert mask.shape == ndct.shape[1:], \
            f"mask shape {mask.shape} != spatial {ndct.shape[1:]}"

        # Normalisation range
        assert ndct.min() >= -1e-5 and ndct.max() <= 1.0 + 1e-5, \
            f"ndct out of [0,1]: min={ndct.min():.4f} max={ndct.max():.4f}"
        assert ldct.min() >= -1e-5 and ldct.max() <= 1.0 + 1e-5, \
            f"ldct out of [0,1]: min={ldct.min():.4f} max={ldct.max():.4f}"

        # Dtype
        assert ndct.dtype == torch.float32
        assert ldct.dtype == torch.float32
        assert mask.dtype == torch.int64

        # Mask values
        assert mask.min() >= 0
        assert mask.max() <= 6

        H, W = ndct.shape[1], ndct.shape[2]
        print(f"    patient_id : {sample['patient_id']}")
        print(f"    slice_idx  : {sample['slice_idx']}")
        print(f"    shape      : [1, {H}, {W}]")
        print(f"    ndct range : [{ndct.min():.4f}, {ndct.max():.4f}]  ✓")
        print(f"    ldct range : [{ldct.min():.4f}, {ldct.max():.4f}]  ✓")
        print(f"    mask dtype : {mask.dtype}  unique={mask.unique().tolist()}  ✓")
        print(f"    hdct_path  : {sample['hdct_path']}")

        # ── 2c. Augmentation reproducibility ─────────────────────────────
        print("\n  Augmentation reproducibility:")
        # Fetch the same sample twice — with deterministic per-sample RNG
        # the result must be identical (flips are seeded by seed ^ idx).
        s1 = train_ds[0]
        s2 = train_ds[0]
        assert torch.equal(s1['ndct'], s2['ndct']), \
            "Same index returned different ndct — augmentation not deterministic!"
        assert torch.equal(s1['mask'], s2['mask']), \
            "Same index returned different mask — augmentation not deterministic!"
        print(f"    Deterministic augmentation (same idx → same output)  ✓")

        # ndct and ldct must have received the same flip
        # (we can only verify spatial consistency, not which flip was applied)
        assert s1['ndct'].shape == s1['ldct'].shape
        print(f"    ndct / ldct spatial shapes match  ✓")

        # ── 2d. DataLoader batch ─────────────────────────────────────────
        print("\n  DataLoader batch check (batch_size=2, num_workers=0):")
        dl    = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dl))

        assert batch['ndct'].shape == (2, 1, H, W), batch['ndct'].shape
        assert batch['ldct'].shape == (2, 1, H, W)
        assert batch['mask'].shape == (2, H, W)
        print(f"    batch ndct : {list(batch['ndct'].shape)}  ✓")
        print(f"    batch ldct : {list(batch['ldct'].shape)}  ✓")
        print(f"    batch mask : {list(batch['mask'].shape)}  ✓")

        # ── 2e. Zero-mask fallback ────────────────────────────────────────
        print("\n  Zero-mask fallback (masks_root='/nonexistent'):")
        ds_nomask = CTSliceDataset(
            data_root  = real_root,
            masks_root = '/nonexistent/masks',
            split      = 'train',
            augment    = False,
        )
        s_nm = ds_nomask[0]
        assert s_nm['mask'].sum() == 0, \
            "Expected all-zero mask when masks_root is missing."
        print(f"    Zero mask returned when masks_root absent  ✓")

        print("\nCTSliceDataset (real data): PASSED")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)
```