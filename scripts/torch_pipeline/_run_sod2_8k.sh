#!/usr/bin/env bash
# Driver: runs the three SOD2 8k inhibition configs in sequence inside the
# maxtoki-dev container. Each config produces:
#   out/sod2_217m_inhibit_<sampling>_seq8k/
#     summary.json, scores.npz, prep_summary.json, viz/*.png
# Logs land at out/_run_sod2_<sampling>_8k.log
set -euo pipefail
cd /workspaces/maxToki

CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json

run_one () {
    local tag="$1"      # first | evenly | oldest3
    local cfg="$2"
    local out="out/sod2_217m_inhibit_${tag}_seq8k"
    local log="out/_run_sod2_${tag}_8k.log"
    echo "==== SOD2 ${tag} 8k ====" | tee -a "$log"
    date  >> "$log"
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
        --wandb-mode disabled \
        2>&1 | tee -a "$log"
    date  >> "$log"
    echo "==== done SOD2 ${tag} 8k ====" | tee -a "$log"
}

run_one first   scripts/torch_pipeline/configs/sod2_first_seq8k.yaml
run_one evenly  scripts/torch_pipeline/configs/sod2_evenly_seq8k.yaml
run_one oldest3 scripts/torch_pipeline/configs/sod2_oldest3_seq8k.yaml
echo "ALL SOD2 8K RUNS COMPLETE"
