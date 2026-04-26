#!/usr/bin/env bash
set -euo pipefail

# Step 0 (optional): run TotalSegmentator on NDCT volumes
# bash scripts/run_totalseg_offline.sh data/ndct_volumes_nifti data/totalseg_raw

export PYTHONPATH=src

# Step 1: convert TotalSegmentator masks to 7 classes and slice-level files
python -m anatomy_denoise.data.prepare_totalseg \
  --totalseg_dir data/totalseg_raw \
  --output_masks_dir data/masks_7cls \
  --manifest_path data/totalseg_manifest.json

# Step 2: build train/val splits
python scripts/build_splits.py --ids_dir data/ldct --out_dir data/splits --val_ratio 0.1

# Step 3: train Stage 1
bash scripts/train_stage1.sh

# Step 4: train Stage 2 (frozen Stage 1)
bash scripts/train_stage2.sh

# Step 5: evaluate
bash scripts/evaluate.sh

