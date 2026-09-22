#!/bin/bash
# slurm-runner.sh - Wrapper to run GitHub Actions runner with Slurm GPU allocation

set -e

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# Load necessary modules
module load cuda/12.x 2>/dev/null || true
module load gcc/11.x 2>/dev/null || true

# Submit job to Slurm with GPU allocation
srun --gpus=1 \
     --cpus-per-task=4 \
     --mem=200G \
     --time=12:00:00 \
     --partition=gpu \
     bash -c "cd $RUNNER_DIR && ./run.sh"
