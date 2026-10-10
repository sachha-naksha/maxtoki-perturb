#!/usr/bin/env bash
#SBATCH --job-name=skm_direction_compare
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/skm_direction_compare.%j.out
#SBATCH --error=delta/logs/skm_direction_compare.%j.out
set -euo pipefail
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
hostname
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
apptainer exec --bind "$PWD" --bind /tmp:/tmp --env SLURM_JOB_ID="$SLURM_JOB_ID" --env PYTHONDONTWRITEBYTECODE=1 /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif bash -c '
set -euo pipefail
python -m pytest -q -p no:cacheprovider tests/test_score_aging_direction.py tests/test_aging_temporal.py
python -u scripts/torch_pipeline/score_aging_direction.py compare --data out/aging_skm_young_to_old_v1 --baselines out/aging_skm_young_to_old_baselines_v1 --predictions out/aging_skm_young_to_old_eval_v1 --output out/aging_skm_young_to_old_comparison_v1
'
