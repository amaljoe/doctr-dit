#!/bin/bash
#SBATCH --job-name=l40-job
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Print node info
hostname
nvidia-smi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

sleep infinity

