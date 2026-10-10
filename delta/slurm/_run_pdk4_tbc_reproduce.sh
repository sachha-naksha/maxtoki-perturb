#!/usr/bin/env bash
#SBATCH --job-name=pdk4_tbc_reproduce
#SBATCH --account=bhdw-delta-gpu
#SBATCH --partition=gpuH200x8-interactive,gpuA100x4-interactive,gpuA100x8-interactive,gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=delta/logs/pdk4_tbc_reproduce.%j.out
#SBATCH --error=delta/logs/pdk4_tbc_reproduce.%j.out
set -euo pipefail
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
cd "$PERTURB_DIR"
source delta/slurm/maxtoki_env.sh
CACHE_DIR=$MAXTOKI_CACHE_DIR
mkdir -p "$CACHE_DIR"/hf "$CACHE_DIR"/tmp "$CACHE_DIR"/megatron "$CACHE_DIR"/xdg "$APPTAINERENV_PYTHONPYCACHEPREFIX"
OUT_DIR="$PERTURB_DIR/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_${SLURM_JOB_ID}"
printf 'Slurm job %s on ' "$SLURM_JOB_ID"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
apptainer exec "${MAXTOKI_APPTAINER_ARGS[@]}" \
  --bind "$PERTURB_DIR":/workspaces/maxToki \
  --bind /projects/bhdw/asachan:/projects/bhdw/asachan \
  --bind "$CACHE_DIR":/cache \
  --env PYTHONNOUSERSITE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env XDG_CACHE_HOME=/cache/xdg --env HF_HOME=/cache/hf \
  --env TMPDIR=/cache/tmp --env MEGATRON_CACHE_DIR=/cache/megatron \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" \
  "$MAXTOKI_SIF" bash -c '
    set -euo pipefail
    cd /workspaces/maxToki
    unset RERAND_SEED
    python3 -u delta/slurm/_torch_pipeline_entry_rerand.py \
      --spec scripts/torch_pipeline/configs/pdk4_evenly_seq8k.yaml \
      --ckpt-dir /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo \
      --tokenizer-path delta/configs/token_dictionary_tbc_official_extended.json \
      --variant 217m --out-dir "$1" --devices 1 --tensor-parallel-size 1 \
      --pipeline-parallel-size 1 --context-parallel-size 1 \
      --precision bf16-mixed --wandb-mode disabled
    python3 -u delta/scripts/compare_tbc_reproduction.py \
      --original out/pdk4_217m_inhibit_evenly_seq8k --rerun "$1" \
      --legacy-dictionary /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.before-id-fix.json \
      --dictionary delta/configs/token_dictionary_tbc_official_extended.json
  ' bash "$OUT_DIR"
