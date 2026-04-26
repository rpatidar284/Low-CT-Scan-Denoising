#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m anatomy_denoise.trainers.evaluate \
  --data_root data \
  --val_split data/splits/val.txt \
  --stage1_ckpt outputs/stage1/stage1_epoch_100.pt \
  --stage2_ckpt outputs/stage2/stage2_step_250000.pt

