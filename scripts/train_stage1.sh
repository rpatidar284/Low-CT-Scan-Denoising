#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m anatomy_denoise.trainers.train_stage1 \
  --data_root data \
  --train_split data/splits/train.txt \
  --output_dir outputs/stage1

