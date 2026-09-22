#!/bin/bash
# submit-runner.sh - Submit GitHub Actions runner as a Slurm batch job

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=github-runner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/slurm/slurm-%j.log

module load cuda/12.x
cd /gpfs/home/jensencc/actions-runner
./run.sh
SLURM

# Replace placeholder with actual runner directory
sed -i "s|/gpfs/home/jensencc/actions-runner|$RUNNER_DIR|g" $(ls -t slurm-*.sbatch | head -1) 2>/dev/null || true
