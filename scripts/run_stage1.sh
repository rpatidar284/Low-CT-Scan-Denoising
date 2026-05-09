#!/bin/bash
# Stage 1 Training Script
# Run in ldct_mamba conda environment
set -e

cd /home/teaching/Music/Nigam_51/Project_51

echo "============================================"
echo "Stage 1 Training - VM-UNet Teacher"
echo "Config: configs/stage1_config.yaml"
echo "Steps: 20000 | Image size: 256 | Batch: 4"
echo "============================================"

export CUDA_VISIBLE_DEVICES=0

python -c "
from training.train_stage1 import train_stage1
train_stage1(
    config_path='configs/stage1_config.yaml',
    checkpoint_dir='/home/teaching/Music/Nigam_51/Project_51/outputs/stage1',
)
"
