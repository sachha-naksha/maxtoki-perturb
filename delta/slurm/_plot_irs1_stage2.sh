#!/usr/bin/env bash
#SBATCH --job-name=irs1_stage2_plot
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=delta/logs/irs1_stage2_plot.%j.out
#SBATCH --error=delta/logs/irs1_stage2_plot.%j.out
set -euo pipefail
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
hostname
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
apptainer exec --bind "$PWD" --bind /tmp:/tmp --env SLURM_JOB_ID="$SLURM_JOB_ID" --env PYTHONDONTWRITEBYTECODE=1 /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif bash -c '
set -euo pipefail
python -m pytest -q -p no:cacheprovider tests/test_plot_pdk4_stage2.py::test_deliverable_csv_and_figures_include_primary_and_historical tests/test_plot_pdk4_stage2.py::test_prediction_collapse_is_visible_against_variable_ground_truth
python -u scripts/torch_pipeline/plot_pdk4_stage2.py --run "Stage 2 selected (100 steps)=out/irs1_stage2_tbc_v1/runs/selected" --run "Stage 2 final (500 steps)=out/irs1_stage2_tbc_v1/runs/final" --gene-symbol IRS1 --output out/irs1_stage2_tbc_v1/plots
'
