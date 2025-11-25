#!/bin/bash
#SBATCH --job-name=l40-job
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Print node info
hostname
nvidia-smi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd workspace/doctr-dit/
source .venv/bin/activate

jupyter lab --ip=0.0.0.0 --port=8888

sleep infinity

