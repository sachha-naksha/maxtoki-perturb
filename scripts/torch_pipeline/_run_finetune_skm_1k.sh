#!/usr/bin/env bash
# Real Stage 2 finetune run: 1000 steps, batch=2, seq=4096 on a single H200.
# Periodic stage2_heads_step{N}.pt every 200 steps in case anything trips.
# Estimated wall-clock: ~60 min.
set -euo pipefail

export PERTURB_DIR=/projects/bhdw/asachan/methods/maxtoki-perturb
export MAXTOKI_DIR=/projects/bhdw/asachan/models/MaxToki
export BUNDLE_DIR=/projects/bhdw/asachan/tmp/maxtoki_finetune_skm
export OUT_DIR=$PERTURB_DIR/out/finetune_skm_1k
export CACHE_DIR=/work/hdd/bgdb/asachan/cache/maxtoki

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

LOG="$OUT_DIR/_run_finetune_skm_1k.log"
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
    --max-steps      1000 \
    --batch-size     2 \
    --seq-length     4096 \
    --k-context      3 \
    --lr             1e-4 \
    --w-ce           1.0 \
    --w-mse          0.1 \
    --save-every     200
INNER_EOF

echo "==== finetune-skm 1k-step ====" | tee -a "$LOG"
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
echo "==== done finetune-skm 1k-step ====" | tee -a "$LOG"
echo
echo "=== last 5 steps ==="
grep -E "step +[0-9]" "$LOG" | tail -5
