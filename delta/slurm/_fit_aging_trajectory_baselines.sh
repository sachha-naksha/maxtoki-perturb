#!/usr/bin/env bash
#SBATCH --job-name=skm_trajectory_baselines
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/skm_trajectory_baselines.%j.out
#SBATCH --error=delta/logs/skm_trajectory_baselines.%j.out
set -euo pipefail
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
hostname
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
apptainer exec --bind "$PWD" --bind /tmp:/tmp --env SLURM_JOB_ID="$SLURM_JOB_ID" --env PYTHONDONTWRITEBYTECODE=1 /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif bash -c '
set -euo pipefail
python -m pytest -q -p no:cacheprovider tests/test_score_aging_direction.py tests/test_aging_temporal.py tests/test_score_aging_trajectory.py
python -u scripts/torch_pipeline/score_aging_direction.py baselines --data out/aging_skm_stage2_cross_donor_v1 --output out/aging_skm_cross_donor_matched_baselines_v1
'
