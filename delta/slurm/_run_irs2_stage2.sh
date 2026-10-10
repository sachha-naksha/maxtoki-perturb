#!/usr/bin/env bash
#SBATCH --job-name=irs2_stage2_tbc
#SBATCH --account=bhdw-delta-gpu
#SBATCH --partition=gpuA100x4-interactive,gpuA40x4-interactive,gpuA100x8-interactive,gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/irs2_stage2_tbc.%j.out
#SBATCH --error=delta/logs/irs2_stage2_tbc.%j.out
# Paired IRS2 inhibition with selected and final Stage 2 checkpoints
set -euo pipefail
# Do not inherit administrative container bind paths into compute containers.
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
: "${SLURM_JOB_ID:?Requires Slurm compute allocation}"
source delta/slurm/maxtoki_env.sh
printf 'Job %s on ' "$SLURM_JOB_ID"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
CACHE_DIR=$MAXTOKI_CACHE_DIR
mkdir -p "$CACHE_DIR"/{hf,tmp,megatron,xdg} "$APPTAINERENV_PYTHONPYCACHEPREFIX"
apptainer exec "${MAXTOKI_APPTAINER_ARGS[@]}" \
  --bind "$PERTURB_DIR":/workspaces/maxToki \
  --bind /projects/bhdw/asachan:/projects/bhdw/asachan --bind "$CACHE_DIR":/cache \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" --env PYTHONNOUSERSITE=1 \
  --env WANDB_MODE=disabled --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env HF_HOME=/cache/hf --env TMPDIR=/cache/tmp --env XDG_CACHE_HOME=/cache/xdg \
  --env MEGATRON_CACHE_DIR=/cache/megatron "$MAXTOKI_SIF" \
  bash -c 'set -euo pipefail; cd /workspaces/maxToki
    python -u scripts/torch_pipeline/run_pdk4_stage2.py run --data out/irs2_stage2_tbc_v1/data --output out/irs2_stage2_tbc_v1/runs' bash "$@"
