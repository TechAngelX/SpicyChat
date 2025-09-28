#!/bin/bash
#SBATCH --job-name=cooking-gpu
#SBATCH --partition=tesla
#SBATCH --gres=gpu:1
#SBATCH --mem=40G  # Increased memory slightly to try and prevent 'Killed' error
#SBATCH --time=6:00:00
# Using the hardcoded absolute path to ensure log files are created
#SBATCH --output=/tmp/rangel/spicychat/logs/cooking_%j.out
#SBATCH --error=/tmp/rangel/spicychat/logs/cooking_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Activate the environment
source spicyVenv/bin/activate

# Create the temporary cache directory
export HF_HOME="/tmp/cooking_cache_${SLURM_JOB_ID}"
mkdir -p $HF_HOME

echo "Step 1: Training model..."
python scripts/train_cooking_model.py

rm -rf "$HF_HOME"
echo "Job completed at: $(date)"
