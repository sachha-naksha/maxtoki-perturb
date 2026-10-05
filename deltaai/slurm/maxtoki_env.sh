# Source this before running maxToki commands on DeltaAI.
#   source /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai/slurm/maxtoki_env.sh
#
# Defines two helpers:
#   $MAXTOKI_APPTAINER_ARGS  - args to prepend to `apptainer exec` (binds, PYTHONPATH env)
#   maxtoki_exec ...         - wrapper that calls `apptainer exec --nv <binds> <sif> <args>`

DELTAAI_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
export DELTAAI_ROOT

export MAXTOKI_SIF=${MAXTOKI_SIF:-$DELTAAI_ROOT/containers/pytorch2506-arm64.sif}
# Prefix lives on NVMe (not /projects) to stay under the project's inode quota —
# see SETUP_LOG attempt 10. Override via $MAXTOKI_ENV if moved elsewhere.
export MAXTOKI_ENV=${MAXTOKI_ENV:-/work/nvme/bhdw/asachan/maxtoki_env}
export MAXTOKI_SRC=${MAXTOKI_SRC:-$DELTAAI_ROOT/src/maxToki}

# The pytorch:25.06-py3 image uses python3.12; matches the Dockerfile's dist-packages path.
_MAXTOKI_PY=${_MAXTOKI_PY:-python3.12}

# pip on Debian/Ubuntu writes --prefix installs to $PREFIX/local/lib/<py>/dist-packages,
# not $PREFIX/lib/<py>/site-packages. Point PYTHONPATH at the actual install location.
export APPTAINERENV_PYTHONPATH="/opt/env/local/lib/${_MAXTOKI_PY}/dist-packages"
export APPTAINERENV_PATH="/opt/env/local/bin:/opt/env/bin:/usr/local/bin:/usr/bin:/bin"
# Triton's libcuda.so lookup via `ldconfig -p` misses --nv's /.singularity.d/libs/;
# TRITON_LIBCUDA_PATH tells Triton to look there directly.
export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

MAXTOKI_APPTAINER_ARGS=(
  --nv
  --bind "$MAXTOKI_ENV":/opt/env
  --bind "$MAXTOKI_SRC":/workspace/bionemo2
  --bind /tmp:/tmp
)
export MAXTOKI_APPTAINER_ARGS

maxtoki_exec() {
  apptainer exec "${MAXTOKI_APPTAINER_ARGS[@]}" "$MAXTOKI_SIF" "$@"
}
