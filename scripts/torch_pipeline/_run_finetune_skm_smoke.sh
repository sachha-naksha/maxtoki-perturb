#!/usr/bin/env bash
# Driver: smoke-runs Stage 2 finetune (peak-aware sparsity attention bias on a
# frozen MaxToki 217M backbone) inside the maxtoki-dev apptainer. Single GPU,
# 5 steps, batch=1, n-traj=8, seq-length=4096. Goal: verify that
#   - the prep bundle loads and tokens remap into full-dict id space
#   - the trajectory dataset assembles and pads correctly
#   - the (B,1,1,Lk) attention bias plumbs through Megatron->TE without error
#   - one backward pass actually moves the LM head's weights
# Outputs at out/finetune_skm_smoke/summary.json + the tee'd log.
set -euo pipefail

export PERTURB_DIR=/projects/bhdw/asachan/methods/maxtoki-perturb
export MAXTOKI_DIR=/projects/bhdw/asachan/models/MaxToki
export BUNDLE_DIR=/projects/bhdw/asachan/tmp/maxtoki_finetune_skm
export OUT_DIR=$PERTURB_DIR/out/finetune_skm_smoke
export CACHE_DIR=/work/hdd/bgdb/asachan/cache/maxtoki

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

LOG="$OUT_DIR/_run_finetune_skm_smoke.log"
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
    --smoke
INNER_EOF

echo "==== finetune-skm smoke ====" | tee -a "$LOG"
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
echo "==== done finetune-skm smoke ====" | tee -a "$LOG"
echo
echo "=== summary.json ==="
cat "$OUT_DIR/summary.json" 2>/dev/null || echo "(no summary.json — check log: $LOG)"
