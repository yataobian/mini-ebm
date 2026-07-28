#!/bin/bash
# Train PCD-k (Persistent Contrastive Divergence) on 2D toy datasets

python train.py \
  --dataset two_moons \
  --loss_type pcd \
  --cd_k 10 \
  --langevin_step 0.1 \
  --pcd_n_persistent 100 \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-4 \
  --output_dir ./outputs/pcd_two_moons

python train.py \
  --dataset gmm \
  --loss_type pcd \
  --cd_k 10 \
  --langevin_step 0.1 \
  --pcd_n_persistent 100 \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-4 \
  --output_dir ./outputs/pcd_gmm
