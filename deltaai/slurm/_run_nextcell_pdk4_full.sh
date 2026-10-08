#!/usr/bin/env bash
#SBATCH --job-name=nc_pdk4_full
#SBATCH --account=bhdw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nc_pdk4_full.%j.out
#SBATCH --error=logs/nc_pdk4_full.%j.out
#
# NextCell FULL: 2000 OM6+OM9 queries, PDK4 inhibit, greedy decoding, 217M.
# Time cap sized from the smoke run - raise if the smoke measurement says so.
#
# Launch:  (from deltaai/)  sbatch slurm/_run_nextcell_pdk4_full.sh

set -euo pipefail

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
CACHE_DIR=/work/nvme/bhdw/asachan/cache/maxtoki
CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK_PATH=$CKPT_DIR/context/token_dictionary.json
OUT_DIR=$PERTURB_DIR/out/nextcell_pdk4_inhibit_full_${SLURM_JOB_ID:-local}

cd "$DELTAAI_ROOT"
mkdir -p logs "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

source slurm/maxtoki_env.sh

echo "== $(date -Is) nc_pdk4_full on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "  MAXTOKI_SIF: $MAXTOKI_SIF"
echo "  MAXTOKI_ENV: $MAXTOKI_ENV"
echo "  CKPT_DIR:    $CKPT_DIR"
echo "  OUT_DIR:     $OUT_DIR"

apptainer exec --nv \
  --bind "$MAXTOKI_ENV":/opt/env \
  --bind "$MAXTOKI_SRC":/workspace/bionemo2 \
  --bind "$PERTURB_DIR":/workspaces/maxToki \
  --bind /projects/bhdw/asachan:/projects/bhdw/asachan \
  --bind "$CACHE_DIR":/cache \
  --bind /tmp:/tmp \
  --env PYTHONNOUSERSITE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env HF_HOME=/cache/hf \
  --env TRANSFORMERS_CACHE=/cache/hf \
  --env TMPDIR=/cache/tmp \
  --env MEGATRON_CACHE_DIR=/cache/megatron \
  "$MAXTOKI_SIF" bash -lc "
    set -euo pipefail
    cd /workspaces/maxToki
    echo '  python=' \$(which python3) '  torch=' \$(python3 -c 'import torch; print(torch.__version__)')
    python3 -c 'import bionemo.maxtoki, nemo, megatron.core, transformer_engine; print(\"imports OK\")'
    python3 deltaai/slurm/_torch_pipeline_entry.py \
        --spec scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_full.yaml \
        --ckpt-dir $CKPT_DIR \
        --tokenizer-path $TOK_PATH \
        --variant 217m \
        --out-dir $OUT_DIR \
        --micro-batch-size 1 \
        --devices 1 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --context-parallel-size 1 \
        --precision bf16-mixed \
        --wandb-mode disabled
  "

echo "== $(date -Is) done =="
echo "outputs: $OUT_DIR"
ls -la "$OUT_DIR"
echo "--- summary ---"; cat "$OUT_DIR/summary.json" 2>/dev/null || echo "no summary"
