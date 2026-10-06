#!/usr/bin/env bash
#SBATCH --job-name=tp_pdk4_8k
#SBATCH --account=bhdw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tp_pdk4_inhibit_8k.%j.out
#SBATCH --error=logs/tp_pdk4_inhibit_8k.%j.out
#
# Smoke test: run scripts/torch_pipeline on aging-SkM data with the PDK4
# inhibit 8k zero-shot config, inside the DeltaAI ARM maxtoki env. Verifies
# that the full dataset_prep -> bionemo.predict -> score pipeline runs on
# GH200 using the same h5ad already run with maxtoki_perturb on x86.
#
# Launch:  (from deltaai/)  sbatch slurm/torch_pipeline_pdk4_8k.sbatch

set -euo pipefail

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
CACHE_DIR=/work/nvme/bhdw/asachan/cache/maxtoki
CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK_PATH=$CKPT_DIR/context/token_dictionary.json
OUT_DIR=$PERTURB_DIR/out/pdk4_217m_inhibit_evenly_seq8k_deltaai

cd "$DELTAAI_ROOT"
mkdir -p logs "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron}

source slurm/maxtoki_env.sh

echo "== $(date -Is) tp_pdk4_8k on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "  MAXTOKI_SIF: $MAXTOKI_SIF"
echo "  MAXTOKI_ENV: $MAXTOKI_ENV"
echo "  CKPT_DIR:    $CKPT_DIR"
echo "  OUT_DIR:     $OUT_DIR"

# apptainer exec with the maxtoki_env binds plus the perturb repo (so the
# config's ./data/... relative paths resolve) and /projects (so the model
# checkpoint abs path is visible from inside the container). APPTAINERENV_*
# vars set by maxtoki_env.sh flow through automatically.
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
        --spec scripts/torch_pipeline/configs/pdk4_evenly_seq8k.yaml \
        --ckpt-dir $CKPT_DIR \
        --tokenizer-path $TOK_PATH \
        --variant 217m \
        --out-dir $OUT_DIR \
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
