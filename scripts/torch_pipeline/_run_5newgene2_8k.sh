#!/usr/bin/env bash
# Driver: runs 10 zero-shot 8k perturbation configs (SLC2A2, PAGE1, MYCT1, PRL,
# MYH8 — each inhibit + overexpress) inside the maxtoki-dev apptainer.
# All 10 share the same 3-evenly-spaced YM2 (34 y/o) context cells and 8k seq
# length; query = each 80 y/o cell (OM6 + OM9). Each config produces:
#   out/<gene>_217m_<direction>_evenly_seq8k/
#     summary.json, scores.npz, prep_summary.json, viz/*.png
# Logs land at out/_run_<gene>_<direction>_evenly_8k.log
set -euo pipefail

export H5AD=/projects/bhdw/asachan/methods/maxtoki-perturb/data/zero_shot/rna_zero_shot.preprocessed.h5ad
export MAXTOKI_DIR=/projects/bhdw/asachan/models/MaxToki
export PERTURB_DIR=/projects/bhdw/asachan/methods/maxtoki-perturb
export OUT_DIR=$PERTURB_DIR/out
export DATA_DIR=$PERTURB_DIR/data
export CACHE_DIR=/work/hdd/bgdb/asachan/cache/maxtoki
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment; never hard-code it}"

mkdir -p "$OUT_DIR" "$DATA_DIR" "$CACHE_DIR"/{hf,tmp,megatron,wandb}

SIF=/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif

read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
cd /workspaces/maxToki

CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json

run_one () {
    local gene="$1"
    local dir="$2"
    local cfg="$3"
    local out="out/${gene}_217m_${dir}_evenly_seq8k"
    local log="out/_run_${gene}_${dir}_evenly_8k.log"
    echo "==== ${gene} ${dir} evenly 8k ====" | tee -a "$log"
    date >> "$log"
    python scripts/torch_pipeline/run_inhibit_temporal_mse.py \
        --spec "$cfg" \
        --ckpt-dir "$CKPT_DIR" \
        --tokenizer-path "$TOK" \
        --variant 217m \
        --out-dir "$out" \
        --devices 1 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --context-parallel-size 1 \
        --precision bf16-mixed \
        --wandb-mode online \
        2>&1 | tee -a "$log"
    date >> "$log"
    echo "==== done ${gene} ${dir} evenly 8k ====" | tee -a "$log"
}

run_one slc2a2 inhibit     scripts/torch_pipeline/configs/slc2a2_inhibit_evenly_seq8k.yaml
run_one slc2a2 overexpress scripts/torch_pipeline/configs/slc2a2_overexpress_evenly_seq8k.yaml
run_one page1  inhibit     scripts/torch_pipeline/configs/page1_inhibit_evenly_seq8k.yaml
run_one page1  overexpress scripts/torch_pipeline/configs/page1_overexpress_evenly_seq8k.yaml
run_one myct1  inhibit     scripts/torch_pipeline/configs/myct1_inhibit_evenly_seq8k.yaml
run_one myct1  overexpress scripts/torch_pipeline/configs/myct1_overexpress_evenly_seq8k.yaml
run_one prl    inhibit     scripts/torch_pipeline/configs/prl_inhibit_evenly_seq8k.yaml
run_one prl    overexpress scripts/torch_pipeline/configs/prl_overexpress_evenly_seq8k.yaml
run_one myh8   inhibit     scripts/torch_pipeline/configs/myh8_inhibit_evenly_seq8k.yaml
run_one myh8   overexpress scripts/torch_pipeline/configs/myh8_overexpress_evenly_seq8k.yaml

echo "ALL 10 NEW GENE2 8K RUNS COMPLETE"
INNER_EOF

apptainer exec --nv \
    --pwd /workspaces/maxToki \
    -B "$PERTURB_DIR":/workspaces/maxToki \
    -B /projects/bhdw/asachan:/projects/bhdw/asachan \
    -B /work/hdd/bgdb/asachan:/work/hdd/bgdb/asachan \
    --env PYTHONNOUSERSITE=1 \
    --env H5AD="$H5AD" \
    --env MAXTOKI_DIR="$MAXTOKI_DIR" \
    --env PERTURB_DIR="$PERTURB_DIR" \
    --env OUT_DIR="$OUT_DIR" \
    --env DATA_DIR="$DATA_DIR" \
    --env MAXTOKI_TOKEN_DICT_217M="$MAXTOKI_DIR/MaxToki-217M-bionemo/context/token_dictionary.json" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env WANDB_DIR="$CACHE_DIR/wandb" \
    --env HF_HOME="$CACHE_DIR/hf" \
    --env TRANSFORMERS_CACHE="$CACHE_DIR/hf" \
    --env TMPDIR="$CACHE_DIR/tmp" \
    --env MEGATRON_CACHE_DIR="$CACHE_DIR/megatron" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$SIF" bash -c "$INNER"
