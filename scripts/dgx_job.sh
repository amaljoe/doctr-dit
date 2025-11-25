#!/bin/bash
#SBATCH --job-name=dgx-job
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

# Print node info
hostname
nvidia-smi

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
cd workspace/doctr-dit/
source .venv/bin/activate

jupyter lab --ip=0.0.0.0 --port=8888

sleep infinity
