#!/usr/bin/env bash
#SBATCH --job-name=tbc_dict_test
#SBATCH --account=bgdb-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=delta/logs/tbc_dict_test.%j.out
#SBATCH --error=delta/logs/tbc_dict_test.%j.out
#
# Token-dictionary / untrained-row tests on the 217M BioNeMo checkpoint, PDK4 inhibit,
# TimeBetweenCells (pdk4_evenly_seq8k.yaml, 2000 OM queries, 3 YM2 context cells):
#   A  old (shifted) dict, as all previous runs            -> reproduces out/pdk4_217m_inhibit_evenly_seq8k
#   B  old dict + re-randomized untrained rows (seed 1)    -> does the Δt readout depend on random rows?
#   C  ALIGNED dict (genes/specials match the HF weights)  -> the corrected zero-shot result
#   D  aligned dict + re-randomized rows (seed 1)
#   E  NextCell 2-query smoke with the aligned dict        -> does generation become a cell?
# Launch (repo root): sbatch delta/slurm/_run_tbc_dict_test.sh

set -euo pipefail
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
DELTA_ROOT=$PERTURB_DIR/delta
CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK_OLD=$CKPT_DIR/context/token_dictionary.json
TOK_NEW=$DELTA_ROOT/configs/token_dictionary_217m_aligned.json
J=${SLURM_JOB_ID:-local}
cd "$PERTURB_DIR"
source "$DELTA_ROOT/slurm/maxtoki_env.sh"
CACHE_DIR=$MAXTOKI_CACHE_DIR
mkdir -p "$DELTA_ROOT/logs" "$CACHE_DIR"/{hf,tmp,megatron,xdg} "$APPTAINERENV_PYTHONPYCACHEPREFIX"
echo "== $(date -Is) tbc_dict_test on $(hostname) =="; nvidia-smi --query-gpu=name,memory.total --format=csv

APPTAINER_ARGS=( "${MAXTOKI_APPTAINER_ARGS[@]}"
  --bind "$PERTURB_DIR":/workspaces/maxToki --bind /projects/bhdw/asachan:/projects/bhdw/asachan --bind "$CACHE_DIR":/cache
  --env PYTHONNOUSERSITE=1 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env TMPDIR=/cache/tmp --env MEGATRON_CACHE_DIR=/cache/megatron )

run_tbc() {  # name tokdict rerand_seed(or "")
  local name=$1 tok=$2 seed=$3
  local out=$PERTURB_DIR/out/tbc_dict_test_${name}_delta_$J
  echo "=== $(date -Is) TBC $name  dict=$tok  RERAND_SEED=${seed:-none} ==="
  apptainer exec "${APPTAINER_ARGS[@]}" "$MAXTOKI_SIF" bash -lc "
    set -euo pipefail; export XDG_CACHE_HOME=/cache/xdg HF_HOME=/cache/hf RERAND_SEED='$seed'
    [[ -z \"\$RERAND_SEED\" ]] && unset RERAND_SEED
    cd /workspaces/maxToki
    python3 delta/slurm/_torch_pipeline_entry_rerand.py \
      --spec scripts/torch_pipeline/configs/pdk4_evenly_seq8k.yaml --ckpt-dir $CKPT_DIR --tokenizer-path $tok \
      --variant 217m --out-dir $out --devices 1 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
      --context-parallel-size 1 --precision bf16-mixed --wandb-mode disabled" 2>&1 | grep -avE "Predicting DataLoader|it/s\]"
  echo "--- $name summary:"; cat "$out/summary.json"; echo
}
# RUNS selects which of A B C D E to execute (default all), e.g. RUNS="B D" sbatch --export=ALL ...
RUNS=${RUNS:-"A B C D E"}
has() { [[ " $RUNS " == *" $1 "* ]]; }
has A && run_tbc A_olddict      "$TOK_OLD" ""
has B && run_tbc B_olddict_rr1  "$TOK_OLD" 1
has C && run_tbc C_aligned      "$TOK_NEW" ""
has D && run_tbc D_aligned_rr1  "$TOK_NEW" 1
has E || { echo "== $(date -Is) done (no E) =="; exit 0; }

OUT_E=$PERTURB_DIR/out/nextcell_pdk4_inhibit_smoke2_aligned_delta_$J
echo "=== $(date -Is) E NextCell 2-query, aligned dict ==="
apptainer exec "${APPTAINER_ARGS[@]}" "$MAXTOKI_SIF" bash -lc "
  set -euo pipefail; export XDG_CACHE_HOME=/cache/xdg HF_HOME=/cache/hf; cd /workspaces/maxToki
  python3 delta/slurm/_torch_pipeline_entry_rerand.py \
    --spec delta/configs/pdk4_inhibit_nextcell_smoke2_delta.yaml --ckpt-dir $CKPT_DIR --tokenizer-path $TOK_NEW \
    --variant 217m --out-dir $OUT_E --micro-batch-size 1 --devices 1 --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 --context-parallel-size 1 --precision bf16-mixed --wandb-mode disabled" 2>&1 | grep -avE "Generating tokens|Saving the dataset"
echo "--- E summary:"; cat "$OUT_E/summary.json" 2>/dev/null || echo none
echo "== $(date -Is) done =="
