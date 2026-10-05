#!/usr/bin/env bash
#SBATCH --job-name=maxtoki_pytest
#SBATCH --account=bhdw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pytest_maxtoki.%j.out
#SBATCH --error=logs/pytest_maxtoki.%j.out
#
# Smoke-run bionemo-maxtoki tests inside the apptainer prefix env.
#   cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
#   sbatch slurm/pytest_maxtoki.sbatch

set -euo pipefail

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
cd "$DELTAAI_ROOT"

source slurm/maxtoki_env.sh

echo "== $(date -Is) pytest on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "  MAXTOKI_SIF: $MAXTOKI_SIF"
echo "  MAXTOKI_ENV: $MAXTOKI_ENV"
echo "  MAXTOKI_SRC: $MAXTOKI_SRC"

# --tb=short: brief traceback; -q: quiet; no -x — run every test to get full counts.
# NB: `tests/` plural at the maxToki repo (not `test/` from the brief).
# PYTEST_EXTRA defaults to empty; override at submit time to e.g. deselect flaky
# upstream tests, like PYTEST_EXTRA="--deselect ...::test_smart_concatenate_mismatched_dtypes".
: "${PYTEST_EXTRA:=}"
maxtoki_exec bash -lc "
  cd /workspace/bionemo2
  echo 'python='\$(which python3)'  torch='\$(python3 -c 'import torch;print(torch.__version__)')
  python3 -m pytest --tb=short -q $PYTEST_EXTRA sub-packages/bionemo-maxtoki/tests
"

echo "== $(date -Is) done =="
