#!/usr/bin/env bash
# Run inference on the finetuned model: load stage2_heads.pt + restore LM head
# and TimeBetweenHead, build trajectories in eval shape (3 evenly-spaced YM2
# context cells / each 80yo cell as query) and read t_pred per query cell.
#
# With --gene PDK4 (set via INHIB_GENE), also runs a perturbed forward pass and
# writes delta_t per cell (mirrors configs/pdk4_evenly_seq8k.yaml's evaluation
# shape, but reading from the FINETUNED head instead of the analytic MSE head).
set -euo pipefail

export PERTURB_DIR=/projects/bhdw/asachan/methods/maxtoki-perturb
export MAXTOKI_DIR=/projects/bhdw/asachan/models/MaxToki
export BUNDLE_DIR=/projects/bhdw/asachan/tmp/maxtoki_finetune_skm
# The finetune output dir to load heads from. Override via FINETUNE_DIR=...
export FINETUNE_DIR=${FINETUNE_DIR:-$PERTURB_DIR/out/finetune_skm_50step}
export OUT_DIR=${OUT_DIR:-$PERTURB_DIR/out/finetune_skm_predict}
export CACHE_DIR=/work/hdd/bgdb/asachan/cache/maxtoki

# Optional: in-silico inhibition. Set INHIB_GENE=PDK4 (or unset for baseline-only).
INHIB_GENE=${INHIB_GENE:-}
INHIB_DIR=${INHIB_DIR:-inhibit}
# Cap query count. Default empty = use all 80yo cells. Override with LIMIT=32 for smoke.
LIMIT=${LIMIT-}

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

LOG="$OUT_DIR/_run_finetune_skm_predict.log"
SIF=/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif

CKPT_DIR=$MAXTOKI_DIR/MaxToki-217M-bionemo
TOK=$CKPT_DIR/context/token_dictionary.json

GENE_ARG=""
if [[ -n "$INHIB_GENE" ]]; then
    GENE_ARG="--gene $INHIB_GENE --direction $INHIB_DIR"
fi
LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

read -r -d '' INNER <<INNER_EOF || true
set -euo pipefail
cd /workspaces/maxToki
python scripts/torch_pipeline/stage2_predict.py \
    --bundle-dir     "$BUNDLE_DIR" \
    --ckpt-dir       "$CKPT_DIR" \
    --tokenizer-path "$TOK" \
    --finetune-dir   "$FINETUNE_DIR" \
    --out-dir        "$OUT_DIR" \
    --seq-length     4096 \
    --K              3 \
    --batch-size     ${BATCH:-2} \
    --context-sample YM2 \
    --query-age      80 \
    $GENE_ARG \
    $LIMIT_ARG
INNER_EOF

echo "==== finetune-skm predict (FINETUNE_DIR=$FINETUNE_DIR INHIB=$INHIB_GENE LIMIT=$LIMIT) ====" | tee -a "$LOG"
date >> "$LOG"

apptainer exec --nv \
    --pwd /workspaces/maxToki \
    -B "$PERTURB_DIR":/workspaces/maxToki \
    -B /projects/bhdw/asachan:/projects/bhdw/asachan \
    -B /work/hdd/bgdb/asachan:/work/hdd/bgdb/asachan \
    --env BUNDLE_DIR="$BUNDLE_DIR" \
    --env CKPT_DIR="$CKPT_DIR" \
    --env TOK="$TOK" \
    --env FINETUNE_DIR="$FINETUNE_DIR" \
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
echo "==== done finetune-skm predict ====" | tee -a "$LOG"
echo
echo "=== summary.json ==="
cat "$OUT_DIR/summary.json" 2>/dev/null || echo "(no summary; see $LOG)"
