#!/usr/bin/env bash
# v2 — every change we believe will make the time-prediction signal real:
#   - per-param-group LR  (LM head 1e-5, time head 1e-3)
#   - rebalanced losses    (w_ce=0.1, w_num=1.0 [Path A], w_mse=1.0 [Path B])
#   - richer TimeBetweenHead (2H input: <eoq> + mean-pool of query gene tokens)
#   - 15% holdout val set, Pearson-r tracked every 100 steps
#   - 2000 steps (2x v1) since the head is now larger and needs the time
set -euo pipefail

export PERTURB_DIR=/projects/bhdw/asachan/methods/maxtoki-perturb
export MAXTOKI_DIR=/projects/bhdw/asachan/models/MaxToki
export BUNDLE_DIR=/projects/bhdw/asachan/tmp/maxtoki_finetune_skm
export OUT_DIR=${OUT_DIR:-$PERTURB_DIR/out/finetune_skm_v2}
export CACHE_DIR=/work/hdd/bgdb/asachan/cache/maxtoki

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

LOG="$OUT_DIR/_run_finetune_skm_v2.log"
SIF=/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif

CKPT_DIR=$MAXTOKI_DIR/MaxToki-217M-bionemo
TOK=$CKPT_DIR/context/token_dictionary.json

MAX_STEPS=${MAX_STEPS:-2000}
BATCH=${BATCH:-2}
SEQ=${SEQ:-4096}

read -r -d '' INNER <<INNER_EOF || true
set -euo pipefail
cd /workspaces/maxToki
python scripts/torch_pipeline/stage2_finetune.py \
    --bundle-dir     "$BUNDLE_DIR" \
    --ckpt-dir       "$CKPT_DIR" \
    --tokenizer-path "$TOK" \
    --out-dir        "$OUT_DIR" \
    --max-steps      $MAX_STEPS \
    --batch-size     $BATCH \
    --seq-length     $SEQ \
    --k-context      3 \
    --lr-lm-head     1e-5 \
    --lr-time-head   1e-3 \
    --w-ce           0.1 \
    --w-num          1.0 \
    --w-mse          1.0 \
    --val-frac       0.15 \
    --val-every      100 \
    --val-max-rows   128 \
    --save-every     500
INNER_EOF

echo "==== finetune-skm v2 (steps=$MAX_STEPS batch=$BATCH seq=$SEQ) ====" | tee -a "$LOG"
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
echo "==== done finetune-skm v2 ====" | tee -a "$LOG"
echo
echo "=== val history ==="
grep -E "^\[stage2\] VAL@" "$LOG"
