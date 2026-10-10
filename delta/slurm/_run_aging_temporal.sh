#!/usr/bin/env bash
#SBATCH --job-name=skm_temporal_joint
#SBATCH --account=bhdw-delta-gpu
#SBATCH --partition=gpuH200x8-interactive,gpuA100x4-interactive,gpuA100x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=delta/logs/skm_temporal_joint.%j.out
#SBATCH --error=delta/logs/skm_temporal_joint.%j.out
# sbatch ... train|predict|score <aging_temporal.py arguments>
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
    python -u scripts/torch_pipeline/aging_temporal.py "$@"' bash "$@"
