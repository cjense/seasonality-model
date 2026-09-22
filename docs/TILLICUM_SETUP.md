# Tillicum GPU Runner Setup Guide

This guide explains how to set up a GitHub Actions self-hosted runner on UW's Tillicum GPU cluster **without sudo access** and **running on-demand only** (no persistent background service).

## Prerequisites

- GitHub personal access token (from Settings > Developer settings > Personal access tokens)
- SSH access to Tillicum
- Repository with admin permissions to add runners
- **No sudo access required** ✓
- **No persistent runner needed** - run only when you push code ✓

## Installation Steps

### 1. SSH into Tillicum and set up the runner

```bash
ssh <your-uw-netid>@tillicum.cs.washington.edu

# Create runner directory
mkdir -p ~/actions-runner
cd ~/actions-runner

# Download runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# Configure runner
./config.sh \
  --url https://github.com/cjense/seasonality-model \
  --token YOUR_GITHUB_TOKEN \
  --name tillicum-slurm-gpu-runner \
  --labels self-hosted,slurm,gpu,tillicum,cuda,Linux,X64 \
  --unattended
```

### 2. Set up the Slurm wrapper script

```bash
# Make scripts executable
chmod +x scripts/slurm-runner.sh
chmod +x scripts/submit-runner.sh

# Copy scripts to runner directory (or create symlinks)
cp scripts/slurm-runner.sh ~/actions-runner/
```

### 3. Create a quick-start script (no sudo needed)

Since you don't have sudo and only want to run on-demand, create this simple script:

```bash
# Create a script to start the runner on-demand
cat > ~/actions-runner/start-runner.sh <<'EOF'
#!/bin/bash
# Start runner with Slurm GPU allocation
# Run this manually when you push code and want to run workflows

cd ~/actions-runner

# Load modules
module load cuda/12.x 2>/dev/null || true
module load gcc/11.x 2>/dev/null || true

echo "Starting GitHub Actions runner with Slurm GPU allocation..."
echo "This will stay running for 12 hours or until jobs complete."
echo "Press Ctrl+C to stop (won't stop running jobs)."
echo ""

# Run with Slurm allocation
srun --gpus=1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=12:00:00 \
     --partition=gpu \
     ./run.sh
EOF

chmod +x ~/actions-runner/start-runner.sh
```

### 3b. (Alternative) Set up as a systemd service (requires sudo, skip if you don't have it)

```bash
# Create systemd service file
cat > /tmp/actions-runner.service <<'EOF'
[Unit]
Description=GitHub Actions Runner (Slurm-based)
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=YOUR_RUNNER_PATH
ExecStart=YOUR_RUNNER_PATH/slurm-runner.sh
Restart=always
RestartSec=10
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="HOME=/home/YOUR_USERNAME"

[Install]
WantedBy=multi-user.target
EOF

# Replace placeholders
sed -i "s|YOUR_USERNAME|$(whoami)|g" /tmp/actions-runner.service
sed -i "s|YOUR_RUNNER_PATH|$HOME/actions-runner|g" /tmp/actions-runner.service

# Install service
sudo mv /tmp/actions-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable actions-runner
sudo systemctl start actions-runner
```

### 4. Test the setup

```bash
# Test the start-runner script
cd ~/actions-runner
./start-runner.sh

# In another terminal, check if runner is listening:
squeue -u $(whoami)
ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null

# The runner should connect to GitHub and wait for jobs
# Press Ctrl+C after verifying
```

## Workflow: How to use on-demand runner

### Step 1: Start the runner when you're ready to test

```bash
# SSH into Tillicum
ssh <netid>@tillicum.cs.washington.edu

# Start the runner (this keeps it running for up to 12 hours)
cd ~/actions-runner
./start-runner.sh

# Leave this terminal open, it will listen for GitHub events
# You'll see output like: "Listening for Jobs"
```

### Step 2: Push your code to GitHub

```bash
# In another terminal, on your local machine
git push origin main
```

### Step 3: GitHub Actions runs your workflow

- Your workflow will trigger automatically
- The runner on Tillicum picks up the job
- Model trains, SHAP runs, figures generated
- Results posted to your PR/commit
- After job completes, runner waits for next job

### Step 4: Stop the runner when done

```bash
# Back in the Tillicum terminal where runner is running
Ctrl+C  # This gracefully stops the runner

# Verify Slurm job ended
squeue -u $(whoami)  # Should show no jobs
```

## Troubleshooting

### Runner not connecting to GitHub

```bash
# Check if runner process is running
ps aux | grep -i runner

# Check runner logs in real-time (while running)
cd ~/actions-runner
tail -f _diag/*/Runner_*.log
```

### Check Slurm allocation

```bash
# See your running jobs
squeue -u $(whoami) -l

# Get detailed info about a job
sinfo -p gpu
```

### View GitHub runner status

```bash
cd ~/actions-runner
# Check if runner is registered (while running)
curl -s https://api.github.com/repos/YOUR_USER/YOUR_REPO/actions/runners \
  -H "Authorization: token YOUR_GITHUB_TOKEN" | jq .
```

### Manual GPU verification

```bash
# Test GPU access before starting runner
srun --gpus=1 --pty nvidia-smi

# Or submit a test job
sbatch <<'EOF'
#!/bin/bash
#SBATCH --gpus=1
nvidia-smi
EOF
```

### Runner times out (job runs for 12 hours max)

- Default time limit is 12 hours (`--time=12:00:00`)
- Edit `start-runner.sh` to change this
- For short jobs, 12 hours is usually plenty
- Long jobs will need to split across multiple workflow runs

## Configuration

### Adjust Slurm resources

Edit `scripts/slurm-runner.sh` to change:
- `--gpus=1` - Number of GPUs (default: 1)
- `--cpus-per-task=4` - CPU cores (default: 4)
- `--mem=16G` - Memory allocation (default: 16GB)
- `--time=12:00:00` - Maximum job duration (default: 12 hours)
- `--partition=gpu` - Slurm partition name

### Load different CUDA versions

Update the module load in `scripts/slurm-runner.sh`:
```bash
module load cuda/11.x  # or your preferred version
```

## Workflow Configuration

The workflow file is pre-configured to:
- Request Slurm GPU resources (`runs-on: [self-hosted, slurm, gpu, tillicum, cuda]`)
- Verify GPU availability before running
- Set `CUDA_VISIBLE_DEVICES=0` to use first GPU
- Run all steps in the seasonality-model pixi environment

## Cost Optimization

Since you pay for Tillicum usage, here are ways to keep costs down:

1. **Only run when needed**: Only start `start-runner.sh` when you're actively testing
2. **Use smaller resources**: Edit `start-runner.sh` to request fewer GPUs/CPUs if your model allows
3. **Set time limits**: Runner currently set for 12 hours max - adjust if you need less
4. **Stop after testing**: `Ctrl+C` immediately after your workflow completes

### Example: Run for only 2 hours

Edit `start-runner.sh` and change:
```bash
--time=2:00:00  # Instead of 12:00:00
```

### Example: Use only 1 CPU

Edit `start-runner.sh` and change:
```bash
--cpus-per-task=1  # Instead of 4
```

## Quick Reference: Daily Workflow

```bash
# Morning: SSH and start runner
ssh <netid>@tillicum.cs.washington.edu
cd ~/actions-runner
./start-runner.sh &  # Run in background, or keep terminal open

# During development: Push changes
git push origin main  # Workflow runs automatically

# After testing: Stop runner
fg  # Bring runner to foreground
Ctrl+C  # Stop it

# Verify it stopped
squeue -u $(whoami)  # Should be empty
```

## Documentation

- [GitHub Actions Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners)
- [Slurm Documentation](https://slurm.schedmd.com/)
- [CUDA Setup](https://docs.nvidia.com/cuda/)
