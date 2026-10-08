#!/usr/bin/env bash
#SBATCH --job-name=nc_score
#SBATCH --account=bhdw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/nc_score.%j.out
#SBATCH --error=logs/nc_score.%j.out
#
# Re-run NextCell scoring only against an existing out dir (reuses baseline +
# perturbed predictions; skips dataset build and the ~55-min generation).
#
# Usage:
#   OUT_DIR=/.../out/nextcell_pdk4_inhibit_smoke_<JID> \
#   SPEC=scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_smoke.yaml \
#     sbatch deltaai/slurm/_run_nextcell_score_only.sh
#
# Both envs are required; the SPEC drives the task_type dispatch and the
# target-gene resolution.

set -euo pipefail

: "${OUT_DIR:?set OUT_DIR to the smoke/full output directory}"
: "${SPEC:=scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_smoke.yaml}"

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
CACHE_DIR=/work/nvme/bhdw/asachan/cache/maxtoki
CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK_PATH=$CKPT_DIR/context/token_dictionary.json

cd "$DELTAAI_ROOT"
mkdir -p logs

source slurm/maxtoki_env.sh

echo "== $(date -Is) nc_score on $(hostname) =="
echo "  OUT_DIR: $OUT_DIR"
echo "  SPEC:    $SPEC"

apptainer exec --nv \
  --bind "$MAXTOKI_ENV":/opt/env \
  --bind "$MAXTOKI_SRC":/workspace/bionemo2 \
  --bind "$PERTURB_DIR":/workspaces/maxToki \
  --bind /projects/bhdw/asachan:/projects/bhdw/asachan \
  --bind "$CACHE_DIR":/cache \
  --bind /tmp:/tmp \
  --env PYTHONNOUSERSITE=1 \
  --env HF_HOME=/cache/hf \
  --env TRANSFORMERS_CACHE=/cache/hf \
  --env TMPDIR=/cache/tmp \
  "$MAXTOKI_SIF" bash -lc "
    set -euo pipefail
    cd /workspaces/maxToki
    python3 deltaai/slurm/_torch_pipeline_entry.py \
        --spec $SPEC \
        --ckpt-dir $CKPT_DIR \
        --tokenizer-path $TOK_PATH \
        --variant 217m \
        --out-dir $OUT_DIR \
        --score-only \
        --wandb-mode disabled
  "

echo "== $(date -Is) done =="
echo "--- summary.json ---"; cat "$OUT_DIR/summary.json" 2>/dev/null || echo "no summary"
echo "--- summary_nextcell.json ---"; cat "$OUT_DIR/summary_nextcell.json" 2>/dev/null || echo "no nextcell summary"
