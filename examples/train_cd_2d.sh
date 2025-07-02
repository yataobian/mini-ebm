#!/bin/bash
# This script runs training with Contrastive Divergence (CD)

python train.py \
  --dataset checkerboard \
  --loss_type cd \
  --epochs 2000 \
  --lr 1e-4 \
  --cd_k 10 \
  --langevin_step 0.1 \
  --langevin_noise 0.01 \
  --output_dir ./outputs/cd_checkerboard
