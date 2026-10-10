#!/usr/bin/env bash
#SBATCH --job-name=skm_temporal_prep
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/skm_temporal_prep.%j.out
#SBATCH --error=delta/logs/skm_temporal_prep.%j.out
set -euo pipefail
# Do not inherit administrative container bind paths into compute containers.
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
: "${SLURM_JOB_ID:?Requires Slurm compute allocation}"
printf 'Job %s on ' "$SLURM_JOB_ID"
hostname
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
apptainer exec --bind "$PWD" --bind /tmp:/tmp \
  --env PYTHONDONTWRITEBYTECODE=1 --env SLURM_JOB_ID="$SLURM_JOB_ID" \
  /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif \
  bash -c 'set -euo pipefail
    python -m pytest -q -p no:cacheprovider tests/test_aging_temporal.py tests/test_nextcell_pipeline.py tests/test_extract_token_dict.py
    python -u scripts/torch_pipeline/aging_temporal.py prepare "$@"' \
  bash --output "${1:-out/aging_skm_temporal_${SLURM_JOB_ID}}" "${@:2}"
