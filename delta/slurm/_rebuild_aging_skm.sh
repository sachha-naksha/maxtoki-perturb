#!/usr/bin/env bash
#SBATCH --job-name=aging_skm_rebuild
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/aging_skm_rebuild.%j.out
#SBATCH --error=delta/logs/aging_skm_rebuild.%j.out
set -euo pipefail
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
printf 'Slurm job %s on ' "$SLURM_JOB_ID"
hostname
apptainer exec \
  --bind /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" \
  /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif \
  python -u scripts/torch_pipeline/rebuild_aging_skm.py
