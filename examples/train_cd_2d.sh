#!/bin/bash
# This script runs training with Contrastive Divergence (CD)

python train.py \
  --dataset two_moons \
  --loss_type cd \
  --epochs 200 \
  --lr 1e-4 \
  --cd_k 10 \
  --langevin_step 0.1 \
  --output_dir ./outputs/cd_two_moons

python train.py \
  --dataset gmm \
  --loss_type cd \
  --epochs 200 \
  --lr 1e-4 \
  --cd_k 10 \
  --langevin_step 0.01 \
  --output_dir ./outputs/cd_gmm
  
