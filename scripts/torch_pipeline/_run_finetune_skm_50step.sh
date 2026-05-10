#!/usr/bin/env bash
# 50-step multitask validation:
#   - Path A diagnostic: ce_num drops faster than ce_gene  (off by default; --w-num 0)
#   - Path B (TimeBetweenHead MSE on z-scored pseudotime delta):
#       * mse drifts toward 1 - r^2 < 1 from the random-init baseline of ~1
#       * t_pred_std_z grows from ~0 (head untrained = constant output) toward
#         t_target_std_z (~1 for z-scored target). This is THE signal that the
#         head is reading the query cell, not collapsing to a constant.
# Trajectory shape: K=3 earlier-pseudotime cells from same sample as context;
# each cell in YM2/OM6/OM9 takes a turn as query. seq_length=4096 to fit on
# 1 H200 with batch=2.
set -euo pipefail

export PERTURB_DIR=/projects/bhdw/asachan/methods/maxtoki-perturb
export MAXTOKI_DIR=/projects/bhdw/asachan/models/MaxToki
export BUNDLE_DIR=/projects/bhdw/asachan/tmp/maxtoki_finetune_skm
export OUT_DIR=$PERTURB_DIR/out/finetune_skm_50step
export CACHE_DIR=/work/hdd/bgdb/asachan/cache/maxtoki

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

LOG="$OUT_DIR/_run_finetune_skm_50step.log"
SIF=/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif

CKPT_DIR=$MAXTOKI_DIR/MaxToki-217M-bionemo
TOK=$CKPT_DIR/context/token_dictionary.json

read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
cd /workspaces/maxToki
python scripts/torch_pipeline/stage2_finetune.py \
    --bundle-dir     "$BUNDLE_DIR" \
    --ckpt-dir       "$CKPT_DIR" \
    --tokenizer-path "$TOK" \
    --out-dir        "$OUT_DIR" \
    --max-steps      50 \
    --batch-size     2 \
    --seq-length     4096 \
    --k-context      2 \
    --lr             1e-4
INNER_EOF

echo "==== finetune-skm 50step (Path A) ====" | tee -a "$LOG"
date >> "$LOG"

apptainer exec --nv \
    --pwd /workspaces/maxToki \
    -B "$PERTURB_DIR":/workspaces/maxToki \
    -B /projects/bhdw/asachan:/projects/bhdw/asachan \
    -B /work/hdd/bgdb/asachan:/work/hdd/bgdb/asachan \
    --env BUNDLE_DIR="$BUNDLE_DIR" \
    --env CKPT_DIR="$CKPT_DIR" \
    --env TOK="$TOK" \
    --env OUT_DIR="$OUT_DIR" \
    --env MAXTOKI_TOKEN_DICT="$TOK" \
    --env HF_HOME="$CACHE_DIR/hf" \
    --env TRANSFORMERS_CACHE="$CACHE_DIR/hf" \
    --env TMPDIR="$CACHE_DIR/tmp" \
    --env MEGATRON_CACHE_DIR="$CACHE_DIR/megatron" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTHONNOUSERSITE=1 \
    "$SIF" bash -c "$INNER" 2>&1 | tee -a "$LOG"

date >> "$LOG"
echo "==== done finetune-skm 50step ====" | tee -a "$LOG"
echo
echo "=== ce_num trajectory (first/last 10) ==="
grep -E "step +[0-9]" "$LOG" | head -10
echo "  ..."
grep -E "step +[0-9]" "$LOG" | tail -10
