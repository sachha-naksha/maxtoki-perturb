#!/usr/bin/env bash
#SBATCH --job-name=skm_stage2_eval
#SBATCH --account=bhdw-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=delta/logs/skm_stage2_eval.%j.out
#SBATCH --error=delta/logs/skm_stage2_eval.%j.out
set -euo pipefail
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
: "${SLURM_JOB_ID:?Requires compute allocation}"
TRAIN_OUTPUT=${1:?training output directory}
DATA_DIR=${2:?prepared data directory}
EVAL_OUTPUT=${3:?fresh evaluation output directory}
LAST_CHECKPOINT=()
for checkpoint in "$TRAIN_OUTPUT"/aging_skm_joint/dev/checkpoints/*-last; do
  if [[ -d "$checkpoint/weights" && -f "$checkpoint/aging_temporal_manifest.json" ]]; then
    LAST_CHECKPOINT+=("$checkpoint")
  fi
done
[[ ${#LAST_CHECKPOINT[@]} -eq 1 ]] || { echo "Expected exactly one completed last checkpoint"; exit 1; }
for split in val test; do
  bash delta/slurm/_run_aging_temporal.sh predict --data "$DATA_DIR"     --checkpoint "${LAST_CHECKPOINT[0]}" --task tbc --split "$split"     --output "$EVAL_OUTPUT/tbc_$split" --score
done
bash delta/slurm/_run_aging_temporal.sh predict --data "$DATA_DIR"   --checkpoint "${LAST_CHECKPOINT[0]}" --task nc --split test --limit 2   --output "$EVAL_OUTPUT/nc_test_smoke" --score
