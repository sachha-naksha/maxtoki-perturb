#!/bin/bash
#SBATCH --job-name=maxtoki-zlx1
#SBATCH --partition=gpuA100x4         # or gpuH200 etc.; check with `sinfo`
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --account=<your-allocation>
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load apptainer

# === EDIT THESE ===========================================================
BIONEMO_IMG=/sw/external/NGC/bionemo:latest.sif      # ask Delta support
WEIGHTS_DIR=/scratch/<group>/<user>/weights/maxtoki-1b-bionemo
TOKEN_DICT=$WEIGHTS_DIR/context/token_dictionary.json
SPEC=scripts/torch_pipeline/configs/example_young_context_old_query.yaml
H5AD=/scratch/<group>/<user>/data/4111_cells.h5ad
OUT_DIR=/scratch/<group>/<user>/out/${SLURM_JOB_NAME}
GENE_SYMBOL=ZLX1
DIRECTION=inhibit
VARIANT=1b
SEQ_LEN=16384
# ==========================================================================

mkdir -p logs "$OUT_DIR"

apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind /work:/work \
    "$BIONEMO_IMG" \
    bash -lc "
        pip install --user --quiet pyyaml anndata scipy && \
        python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
            --spec '$SPEC' \
            --h5ad '$H5AD' \
            --ckpt-dir '$WEIGHTS_DIR' \
            --tokenizer-path '$TOKEN_DICT' \
            --variant '$VARIANT' \
            --gene-symbol '$GENE_SYMBOL' \
            --direction '$DIRECTION' \
            --seq-length '$SEQ_LEN' \
            --out-dir '$OUT_DIR' \
            --micro-batch-size 1
    "

echo
echo '=== summary.json ==='
cat "$OUT_DIR/summary.json"
