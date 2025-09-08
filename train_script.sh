#!/bin/bash -l

#PBS -N DIFFUSION_TRAIN
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=24:ngpus=1:gpu_id=H100:mem=160gb

set -euo pipefail

# ─── Setup working directory ──────────────────────────────────────────────
cd /home/n10813934/gitRepos/VGS_diffusion_policy   # adjust if repo path differs

# ─── Activate your virtual environment ────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robodiff

# ─── Add W&B environment variables ────────────────────────────────────────
source ~/.wandb_secrets

# ─── Point to your dataset ────────────────────────────────────────────────
export XARM6_DATASET_PATH=/home/n10813934/data/diffusion/converted_padded   # your .zarr cache dir
export HYDRA_FULL_ERROR=1

# ─── Run diffusion policy training ────────────────────────────────────────
# For smoke-test (10–20 min):
# python train.py --config-name=train_xarm6_diffusion_unet_image_workspace \
#   training.num_epochs=1 training.max_train_steps=6000 training.resume=False \
#   training.checkpoint_every=9999 training.rollout_every=9999 training.sample_every=9999 \
#   training.max_val_steps=128 logging.mode=disabled

# For full run:
python train.py --config-name=train_xarm6_diffusion_unet_real_pretrained_workspace

exit