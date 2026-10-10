#!/usr/bin/env bash
#SBATCH --job-name=skm_temporal_inspect
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=delta/logs/skm_temporal_inspect.%j.out
#SBATCH --error=delta/logs/skm_temporal_inspect.%j.out
set -euo pipefail
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
hostname
apptainer exec --bind "$PWD" --env SLURM_JOB_ID="$SLURM_JOB_ID" \
 /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif python -c '
import anndata as ad
import json
a=ad.read_h5ad("data/zero_shot/rna_zero_shot.preprocessed.h5ad",backed="r")
o=a.obs
cols=["Pseudotime","sample","Annotation"]
print("Missing:",o[cols].isna().sum().to_dict())
print("Donors:",o["sample"].value_counts().to_dict())
print("Trajectories:",o["Annotation"].value_counts(dropna=False).to_dict())
print("Example:",o.loc["CELL240_N1_2_1_11_1",cols].to_dict())
print("Missing per donor:",o[cols].isna().groupby(o["sample"],observed=True).sum().to_dict())
a.file.close()
'
