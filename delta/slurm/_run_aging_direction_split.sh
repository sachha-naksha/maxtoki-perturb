#!/usr/bin/env bash
set -euo pipefail
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
DATA_DIR=${1:?data}
CHECKPOINT=${2:?checkpoint}
OUTPUT_DIR=${3:?output}
SPLIT=${4:?split}
export MASTER_PORT=${5:?master port}
export APPTAINERENV_MASTER_PORT="$MASTER_PORT"
bash delta/slurm/_run_aging_temporal.sh predict --data "$DATA_DIR" --checkpoint "$CHECKPOINT" --task tbc --split "$SPLIT" --output "$OUTPUT_DIR/tbc_$SPLIT"
bash delta/slurm/_run_aging_temporal.sh predict --data "$DATA_DIR" --checkpoint "$CHECKPOINT" --task nc --split "$SPLIT" --limit 10 --output "$OUTPUT_DIR/nc_$SPLIT"
