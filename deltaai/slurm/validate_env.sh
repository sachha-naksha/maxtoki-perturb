#!/usr/bin/env bash
# Sanity checks for the /opt/env prefix install. Run on ghx4-interactive after build_env.sbatch:
#
#   srun -A bhdw-dtai-gh -p ghx4-interactive --gpus-per-node=1 --cpus-per-task=16 --mem=100G -t 1:00:00 --pty bash
#   cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
#   source slurm/maxtoki_env.sh
#   maxtoki_exec bash /workspace/slurm/validate_env.sh    # (bind slurm/ separately if needed)
#
# Or, more directly (mirroring the maxtoki_env.sh bind list):
#
#   apptainer exec --nv \
#     --bind $PWD/env:/opt/env --bind $PWD/src/maxToki:/workspace/bionemo2 \
#     --bind $PWD/slurm:/workspace/slurm \
#     containers/pytorch2506-arm64.sif \
#     bash /workspace/slurm/validate_env.sh

set -euo pipefail

PREFIX=${PREFIX:-/opt/env}
PY=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
# Debian layout: pip --prefix installs to $PREFIX/local/lib/<py>/dist-packages.
export PYTHONPATH="$PREFIX/local/lib/$PY/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PREFIX/local/bin:$PREFIX/bin:$PATH"
export TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-/.singularity.d/libs}

echo "== validate_env.sh =="
echo "  host:   $(hostname)   arch: $(uname -m)"
echo "  python: $(python3 --version)"
echo "  PYTHONPATH: $PYTHONPATH"

echo ""
echo "-- GPU visibility --"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "-- torch + cuda --"
python3 -c '
import torch
print(f"torch    {torch.__version__}")
print(f"cuda     avail={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device   {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
'

echo ""
echo "-- key imports --"
python3 - <<'PY'
import importlib, sys
mods = [
    "torch",
    "transformer_engine.pytorch",
    "megatron.core",
    "nemo",
    "nemo.collections.llm",
    "bionemo.core",
    "bionemo.llm",
    "bionemo.maxtoki",
    "bionemo.maxtoki.predict",
    "bionemo.maxtoki.train",
    "scanpy",
]
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        f = getattr(mod, "__file__", "<pkg>")
        print(f"  OK   {m:<40} {f}")
    except Exception as e:
        print(f"  FAIL {m:<40} {type(e).__name__}: {e}")
        failed.append(m)
if failed:
    print("\nFAILED modules:", failed)
    sys.exit(1)
print("\nall imports OK")
PY

echo ""
echo "-- quick TE round-trip on GPU --"
python3 - <<'PY'
import torch, transformer_engine.pytorch as te
x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)
lin = te.Linear(128, 128).cuda().to(torch.bfloat16)
y = lin(x)
print(f"TE linear ok, out shape={tuple(y.shape)}, dtype={y.dtype}")
PY

echo ""
echo "== all checks passed =="
