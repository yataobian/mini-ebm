#!/bin/bash
# This script runs training with Denoising Score Matching (DSM) 

python train.py \
  --dataset checkerboard \
  --loss_type dsm \
  --epochs 2000 \
  --lr 1e-4 \
  --dsm_sigma 0.2 \
  --output_dir ./outputs/dsm_checkerboard
