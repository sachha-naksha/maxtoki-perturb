#!/usr/bin/env bash
#SBATCH --job-name=nc_tests_delta
#SBATCH --account=bgdb-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/nc_tests_delta.%j.out
#SBATCH --error=delta/logs/nc_tests_delta.%j.out
#
# CPU-only NextCell pipeline tests: spec round-trip, dataset_prep grammar,
# score_nextcell decoder + metrics + ragged-batch writer-fixture handling.
# No bionemo / torch model / anndata needed - fast.
# Mirror of deltaai/slurm/_run_nextcell_tests.sh with Delta Slurm headers and
# the x86 container.
#
# Launch:  (from repo root) sbatch delta/slurm/_run_nextcell_tests.sh

set -euo pipefail

PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
DELTA_ROOT=$PERTURB_DIR/delta

cd "$PERTURB_DIR"
source "$DELTA_ROOT/slurm/maxtoki_env.sh"

mkdir -p "$DELTA_ROOT/logs" "$APPTAINERENV_PYTHONPYCACHEPREFIX"

echo "== $(date -Is) nc_tests_delta on $(hostname) =="

# -p no:cacheprovider keeps pytest from writing .pytest_cache into the shared tree.
apptainer exec "${MAXTOKI_APPTAINER_ARGS[@]}" \
  --bind "$PERTURB_DIR":/workspaces/maxToki \
  --env PYTHONNOUSERSITE=1 \
  "$MAXTOKI_SIF" bash -lc "
    set -euo pipefail
    cd /workspaces/maxToki
    echo '  python=' \$(which python3) '  torch=' \$(python3 -c 'import torch; print(torch.__version__)')
    python3 -m pytest --tb=short -q -p no:cacheprovider tests/test_nextcell_pipeline.py
  "

echo "== $(date -Is) done =="
