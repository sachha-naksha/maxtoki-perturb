# Source this before running maxToki commands on Delta (x86_64, A100/A40).
#   source /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/delta/slurm/maxtoki_env.sh
#
# Mirror of deltaai/slurm/maxtoki_env.sh for the amd64 side. Reuses the shared
# project source + checkpoints under /projects, but points at the amd64 .sif and
# a separate /work/nvme cache so ARM and x86 runs cannot cross-contaminate.

DELTA_ROOT=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/delta
PERTURB_DIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb
export DELTA_ROOT PERTURB_DIR

# Container is amd64; refusing to run on aarch64 is the intended safety net.
export MAXTOKI_SIF=${MAXTOKI_SIF:-/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif}

# Bind the ARM-side bionemo-maxtoki fork read-only. If the x86 .sif already has
# the fork baked in this just shadows it with the same content; if not, this is
# how the fork gets found. See AGENT_NEXTCELL.md §2 for the dry-run check.
export MAXTOKI_SRC=${MAXTOKI_SRC:-$PERTURB_DIR/deltaai/src/maxToki}
export MAXTOKI_BIND_SRC=${MAXTOKI_BIND_SRC:-1}

# Only set MAXTOKI_ENV on Delta if you discover the .sif was built as a thin
# container and needs an external pip prefix (unlikely at 14 GB). Keep empty by
# default; the smoke script skips the /opt/env bind when this is unset.
export MAXTOKI_ENV=${MAXTOKI_ENV:-}

# x86-specific cache. Do NOT point this at the ARM cache dir — Triton and
# Megatron will drop compiled kernels into it.
export MAXTOKI_CACHE_DIR=${MAXTOKI_CACHE_DIR:-/work/nvme/bhdw/asachan/cache/maxtoki_x86}

# Keep .pyc out of the bind-mounted shared source so ARM and x86 interpreters
# don't fight over __pycache__ entries in /projects.
export APPTAINERENV_PYTHONPYCACHEPREFIX=${APPTAINERENV_PYTHONPYCACHEPREFIX:-/work/nvme/bhdw/asachan/.pycache_x86}

# pytorch:25.06-py3 ships python3.12 on both arches.
_MAXTOKI_PY=${_MAXTOKI_PY:-python3.12}

if [[ -n "$MAXTOKI_ENV" ]]; then
  export APPTAINERENV_PYTHONPATH="/opt/env/local/lib/${_MAXTOKI_PY}/dist-packages"
  export APPTAINERENV_PATH="/opt/env/local/bin:/opt/env/bin:/usr/local/bin:/usr/bin:/bin"
fi

# Triton's libcuda.so lookup via `ldconfig -p` misses --nv's /.singularity.d/libs/;
# TRITON_LIBCUDA_PATH tells Triton to look there directly.
export APPTAINERENV_TRITON_LIBCUDA_PATH="/.singularity.d/libs"

# The image's /usr/local/cuda/compat/lib (libcuda 575.57.08) sits ahead of --nv's
# /.singularity.d/libs on LD_LIBRARY_PATH and shadows the host driver's libcuda
# -> "Error 803: unsupported display driver / cuda driver combination". The NGC
# entrypoint normally rm's it, but the .sif is read-only; mask it with an empty dir.
_MAXTOKI_EMPTY_DIR=$MAXTOKI_CACHE_DIR/empty
mkdir -p "$_MAXTOKI_EMPTY_DIR"

MAXTOKI_APPTAINER_ARGS=(--nv --bind /tmp:/tmp --bind "$_MAXTOKI_EMPTY_DIR":/usr/local/cuda/compat/lib
                        --bind /work/nvme/bhdw/asachan:/work/nvme/bhdw/asachan)
if [[ -n "$MAXTOKI_ENV" ]]; then
  MAXTOKI_APPTAINER_ARGS+=(--bind "$MAXTOKI_ENV":/opt/env)
fi
if [[ "$MAXTOKI_BIND_SRC" == "1" ]]; then
  MAXTOKI_APPTAINER_ARGS+=(--bind "$MAXTOKI_SRC":/workspace/bionemo2)
fi
export MAXTOKI_APPTAINER_ARGS

maxtoki_exec() {
  apptainer exec "${MAXTOKI_APPTAINER_ARGS[@]}" "$MAXTOKI_SIF" "$@"
}
