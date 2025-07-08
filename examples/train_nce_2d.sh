#!/bin/bash
# This script runs training with Noise Contrastive Estimation (NCE)

# python train.py \
#   --dataset checkerboard \
#   --loss_type nce \
#   --epochs 2000 \
#   --lr 1e-4 \
#   --nce_noise_ratio 5 \
#   --output_dir ./outputs/nce_checkerboard

python train.py \
  --dataset checkerboard \
  --loss_type adaptive_nce \
  --epochs 2000 \
  --lr 1e-4 \
  --nce_noise_ratio 5 \
  --output_dir ./outputs/nce_adaptive_checkerboard
