#!/usr/bin/env bash
#SBATCH --job-name=irs1_stage2_prep
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=delta/logs/irs1_stage2_prep.%j.out
#SBATCH --error=delta/logs/irs1_stage2_prep.%j.out
set -euo pipefail
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
: "${SLURM_JOB_ID:?Compute allocation required}"
hostname
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
apptainer exec --bind "$PWD" --bind /projects/bhdw/asachan:/projects/bhdw/asachan --bind /tmp:/tmp --env SLURM_JOB_ID="$SLURM_JOB_ID" --env PYTHONDONTWRITEBYTECODE=1   /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif bash -c '
  set -euo pipefail
  python -m pytest -q -p no:cacheprovider tests/test_aging_temporal.py tests/test_pdk4_stage2_prepare.py tests/test_plot_pdk4_stage2.py
  python -u scripts/torch_pipeline/prepare_pdk4_stage2.py --gene IRS1 --original out/pdk4_217m_inhibit_evenly_seq8k --training-data out/aging_skm_stage2_cross_donor_v1 --output out/irs1_stage2_tbc_v1/data
  '
