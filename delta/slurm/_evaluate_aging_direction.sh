#!/usr/bin/env bash
#SBATCH --job-name=skm_direction_eval
#SBATCH --account=bhdw-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=delta/logs/skm_direction_eval.%j.out
#SBATCH --error=delta/logs/skm_direction_eval.%j.out
set -euo pipefail
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
: "${SLURM_JOB_ID:?Compute allocation required}"
DATA_DIR=out/aging_skm_young_to_old_v1
CHECKPOINT=out/aging_skm_stage2_training_v1/aging_skm_joint/dev/checkpoints/epoch=0-val_loss=2.18-step=499-consumed_samples=2000.0-last
OUTPUT_DIR=out/aging_skm_young_to_old_eval_v1
bash delta/slurm/_run_aging_direction_split.sh "$DATA_DIR" "$CHECKPOINT" "$OUTPUT_DIR" val 29651
bash delta/slurm/_run_aging_direction_split.sh "$DATA_DIR" "$CHECKPOINT" "$OUTPUT_DIR" test 29652
