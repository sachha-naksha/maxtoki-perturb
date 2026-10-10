#!/usr/bin/env bash
#SBATCH --job-name=nc_pdk4_smoke_delta
#SBATCH --account=bgdb-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=delta/logs/nc_pdk4_smoke_delta.%j.out
#SBATCH --error=delta/logs/nc_pdk4_smoke_delta.%j.out
#
# NextCell SMOKE on Delta (amd64, A100 40 GB): 20 queries, PDK4 inhibit, greedy,
# 217M. Mirror of deltaai/slurm/_run_nextcell_pdk4_smoke.sh with Delta Slurm
# headers and the x86 container. See delta/AGENT_NEXTCELL.md §0 for the
# shared-filesystem rules this script follows.
#
# Launch:  (from repo root) sbatch delta/slurm/_run_nextcell_pdk4_smoke.sh

set -euo pipefail

PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
DELTA_ROOT=$PERTURB_DIR/delta
CKPT_DIR=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo
TOK_PATH=$CKPT_DIR/context/token_dictionary.json
# _delta_ prefix so grep + `ls out/` immediately shows which cluster produced it.
# Override the spec with NC_SPEC=... (e.g. the 10-query variant for 1 h interactive slots).
NC_SPEC=${NC_SPEC:-delta/configs/pdk4_inhibit_nextcell_smoke_delta.yaml}
OUT_DIR=$PERTURB_DIR/out/nextcell_pdk4_inhibit_smoke_delta_${SLURM_JOB_ID:-local}

cd "$PERTURB_DIR"
source "$DELTA_ROOT/slurm/maxtoki_env.sh"

CACHE_DIR=$MAXTOKI_CACHE_DIR
mkdir -p "$DELTA_ROOT/logs" "$OUT_DIR" "$CACHE_DIR"/{hf,tmp,megatron} \
         "$APPTAINERENV_PYTHONPYCACHEPREFIX"

echo "== $(date -Is) nc_pdk4_smoke_delta on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "  MAXTOKI_SIF: $MAXTOKI_SIF"
echo "  MAXTOKI_ENV: ${MAXTOKI_ENV:-<baked into sif>}"
echo "  MAXTOKI_SRC: $MAXTOKI_SRC (bound=$MAXTOKI_BIND_SRC)"
echo "  CKPT_DIR:    $CKPT_DIR"
echo "  OUT_DIR:     $OUT_DIR"
echo "  CACHE_DIR:   $CACHE_DIR"

# Build the apptainer binds. Share project source + checkpoints read/write (same
# as the ARM script), but keep the ARM container and ARM env prefix out of the
# bind list — they would be invisible to the amd64 interpreter anyway, but
# excluding them makes intent explicit.
APPTAINER_ARGS=(
  "${MAXTOKI_APPTAINER_ARGS[@]}"
  --bind "$PERTURB_DIR":/workspaces/maxToki
  --bind /projects/bhdw/asachan:/projects/bhdw/asachan
  --bind "$CACHE_DIR":/cache
  --env PYTHONNOUSERSITE=1
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env HF_HOME=/cache/hf
  --env TRANSFORMERS_CACHE=/cache/hf
  --env TMPDIR=/cache/tmp
  --env MEGATRON_CACHE_DIR=/cache/megatron
)

apptainer exec "${APPTAINER_ARGS[@]}" "$MAXTOKI_SIF" bash -lc "
    set -euo pipefail
    # ~/.bashrc (sourced by bash -l) points XDG_CACHE_HOME/HF_HOME at /work/nvme, which is
    # not mounted in the container; bionemo.core mkdirs its cache there at import and dies.
    export XDG_CACHE_HOME=/cache/xdg HF_HOME=/cache/hf
    cd /workspaces/maxToki
    echo '  python=' \$(which python3) '  torch=' \$(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')
    python3 -c 'import bionemo.maxtoki, nemo, megatron.core, transformer_engine; print(\"imports OK\")'
    python3 deltaai/slurm/_torch_pipeline_entry.py \
        --spec $NC_SPEC \
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
