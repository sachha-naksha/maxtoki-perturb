#!/usr/bin/env bash
#SBATCH --job-name=maxtoki_env_build
#SBATCH --account=bhdw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/build_env.%j.out
#SBATCH --error=logs/build_env.%j.out
#
# Submit from deltaai/ so relative paths (logs/, containers/, ...) resolve.
#   cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
#   sbatch slurm/build_env.sbatch
#
# ~1h TE compile + ~30 min for the rest. Idempotent via stamp files; re-run to resume.

set -euo pipefail

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
cd "$DELTAAI_ROOT"

BASE_SIF=$DELTAAI_ROOT/containers/pytorch2506-arm64.sif
# ENV_DIR on NVMe: /projects/bhdw ran out of inodes mid-install on attempt 10
# (project-wide 935k hard limit, ~770k already used by other members). The prefix
# install writes ~250k small files — must live on a filesystem with inode headroom.
# /work/nvme/bhdw gives 2.55M soft and is faster for small-file IO.
ENV_DIR=${ENV_DIR:-/work/nvme/bhdw/asachan/maxtoki_env}
MAXTOKI_SRC=$DELTAAI_ROOT/src/maxToki
SLURM_DIR=$DELTAAI_ROOT/slurm
INSTALL_SCRIPT=$SLURM_DIR/install_prefix.sh

test -e "$BASE_SIF"       || { echo "!! base sif missing: $BASE_SIF"; exit 1; }
test -e "$MAXTOKI_SRC"    || { echo "!! maxToki src missing: $MAXTOKI_SRC"; exit 1; }
test -e "$INSTALL_SCRIPT" || { echo "!! install script missing: $INSTALL_SCRIPT"; exit 1; }

mkdir -p "$ENV_DIR" logs

export APPTAINER_CACHEDIR=$DELTAAI_ROOT/containers/.cache
export APPTAINER_TMPDIR=/tmp

echo "== $(date -Is) starting build on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "  BASE_SIF:    $BASE_SIF"
echo "  ENV_DIR:     $ENV_DIR"
echo "  MAXTOKI_SRC: $MAXTOKI_SRC"
echo "  jobid=$SLURM_JOB_ID cpus=$SLURM_CPUS_ON_NODE mem=${SLURM_MEM_PER_NODE}M"

apptainer exec --nv --cleanenv \
  --bind "$ENV_DIR":/opt/env \
  --bind "$MAXTOKI_SRC":/workspace/bionemo2 \
  --bind "$SLURM_DIR":/workspace/slurm \
  --bind "$DELTAAI_ROOT/containers":/workspace/containers \
  --bind /tmp:/tmp \
  "$BASE_SIF" \
  bash /workspace/slurm/install_prefix.sh

echo "== $(date -Is) done =="
du -sh "$ENV_DIR"
