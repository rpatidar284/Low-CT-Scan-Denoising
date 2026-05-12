#!/bin/bash
set -e
cd /home/teaching/Music/Nigam_51/Project_51
echo "============================================"
echo "Stage 2 — VSSD Anatomy-Conditioned Denoiser"
echo "128px | Batch: 14 | Steps: 10000 | ~23h"
echo "============================================"
export CUDA_VISIBLE_DEVICES=0
python -c "
from training.train_stage2 import train_stage2
train_stage2(
    stage1_checkpoint='outputs/stage1/stage1_best.pth',
    image_size=128, batch_size=4, total_steps=10000,
    checkpoint_dir='outputs/stage2',
    lr=1e-4, warmup_steps=1000,
    res_weight=1.0, noise_weight=0.5,
)
"
