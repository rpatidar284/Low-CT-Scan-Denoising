#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m anatomy_denoise.trainers.train_stage2 \
  --data_root data \
  --train_split data/splits/train.txt \
  --stage1_ckpt outputs/stage1/stage1_epoch_100.pt \
  --output_dir outputs/stage2

