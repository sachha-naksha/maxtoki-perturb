#!/usr/bin/env bash
#SBATCH --job-name=nc_tests
#SBATCH --account=bhdw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nc_tests.%j.out
#SBATCH --error=logs/nc_tests.%j.out
#
# CPU-only NextCell pipeline tests: spec round-trip, dataset_prep grammar,
# score_nextcell decoder + metrics + ragged-batch writer-fixture handling.
# No bionemo / torch model / anndata needed - fast.
#
# Launch:  (from deltaai/)  sbatch slurm/_run_nextcell_tests.sh

set -euo pipefail

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb

cd "$DELTAAI_ROOT"
mkdir -p logs

source slurm/maxtoki_env.sh

echo "== $(date -Is) nc_tests on $(hostname) =="

apptainer exec --nv \
  --bind "$MAXTOKI_ENV":/opt/env \
  --bind "$MAXTOKI_SRC":/workspace/bionemo2 \
  --bind "$PERTURB_DIR":/workspaces/maxToki \
  --bind /tmp:/tmp \
  --env PYTHONNOUSERSITE=1 \
  "$MAXTOKI_SIF" bash -lc "
    set -euo pipefail
    cd /workspaces/maxToki
    echo '  python=' \$(which python3) '  torch=' \$(python3 -c 'import torch; print(torch.__version__)')
    python3 -m pytest --tb=short -q tests/test_nextcell_pipeline.py
  "

echo "== $(date -Is) done =="
