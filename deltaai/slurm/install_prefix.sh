#!/usr/bin/env bash
# Runs inside `apptainer exec --nv --bind ENV:/opt/env --bind MAXTOKI:/workspace/bionemo2 pytorch2506-arm64.sif`.
# Installs the maxToki Python stack into /opt/env (user-owned, bind-mounted).
#
# Read this alongside src/maxToki/Dockerfile at commit 0372263 — this script mirrors the
# `bionemo2-base` stage of that Dockerfile, but redirects every install with `--prefix`
# because we don't have root/fakeroot on DeltaAI.
#
# Idempotency: each stage checks a marker file under $STAMPS. Delete the stamp to force redo.

set -euo pipefail

PREFIX=/opt/env
MAXTOKI=/workspace/bionemo2
STAMPS=$PREFIX/.stamps
mkdir -p "$STAMPS"

PY=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
# Debian/Ubuntu Python uses dist-packages under `local/lib/`, not site-packages
# under `lib/`. `pip install --prefix=$PREFIX` writes to $PREFIX/local/lib/<py>/dist-packages/,
# so PYTHONPATH must point there for our installs to be picked up.
SITE=$PREFIX/local/lib/$PY/dist-packages
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PREFIX/local/bin:$PREFIX/bin:$PATH"
mkdir -p "$SITE" "$PREFIX/local/bin"

# The pytorch:25.06-py3 base image sets these for its own uv-based flows; we mirror them.
export UV_LINK_MODE=copy \
       UV_COMPILE_BYTECODE=1 \
       UV_PYTHON_DOWNLOADS=never \
       UV_SYSTEM_PYTHON=true \
       UV_BREAK_SYSTEM_PACKAGES=1 \
       PIP_CONSTRAINT= \
       PIP_ROOT_USER_ACTION=ignore

# The pytorch:25.06-py3 image ships CUDA 12.9 with nvcc defaulting to `-ccbin g++-14`,
# but only g++-13 is installed. Force nvcc to use the g++ that actually exists.
# Also set CC/CXX explicitly in case anything downstream tries to use them (default
# host env may have leaked HPC-style wrappers like CC=cc, CXX=CC).
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
# nvcc's -ccbin flag also works and is more surgical:
export NVCC_PREPEND_FLAGS="-ccbin ${CUDAHOSTCXX}"

# Triton looks for libcuda.so.1 via `ldconfig -p`, but apptainer --nv drops the host
# driver at /.singularity.d/libs/ which isn't in ldconfig's cache. Triton's driver.py
# honors TRITON_LIBCUDA_PATH as an override — set it so Triton skips the ldconfig probe.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs

echo "== install_prefix.sh =="
echo "  arch:   $(uname -m)"
echo "  python: $PY  ($(which python3))"
echo "  PREFIX: $PREFIX"
echo "  SITE:   $SITE"
echo "  torch:  $(python3 -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"
echo "  time:   $(date -Is)"

# ---------------------------------------------------------------------------
# Stage 0: sanity — base image has git, curl, python, nvcc.
# ---------------------------------------------------------------------------
for cmd in git curl python3 pip nvcc; do
  command -v "$cmd" >/dev/null || { echo "!! missing $cmd in base image"; exit 1; }
done

# ---------------------------------------------------------------------------
# Stage 1: patched TransformerEngine, from source, compiled for Hopper only.
#   NVTE_CUDA_ARCHS=90     - GH200 is sm_90a; skips other archs (saves ~45 min).
#   MAX_JOBS=16            - cap parallel g++ jobs; TE build is memory-hungry.
# ---------------------------------------------------------------------------
WHEELS=$PREFIX/wheels
mkdir -p "$WHEELS"

# TE build stage 1: build the wheel to $WHEELS (persistent — survives a failed install).
if [[ ! -f "$STAMPS/te.wheel.done" ]]; then
  echo "-- [te] cloning + patching TransformerEngine --"
  TE_TAG=9d4e11eaa508383e35b510dc338e58b09c30be73
  TE_SRC=/tmp/TransformerEngine
  rm -rf "$TE_SRC"
  # Do NOT use `git clone --recurse-submodules` + `checkout --recurse-submodules`:
  # git chokes on nested submodules added after the default branch (cudnn-frontend).
  git clone https://github.com/NVIDIA/TransformerEngine.git "$TE_SRC"
  git -C "$TE_SRC" checkout "$TE_TAG"
  git -C "$TE_SRC" submodule sync --recursive
  git -C "$TE_SRC" submodule update --init --recursive
  patch -p1 -d "$TE_SRC" < "$MAXTOKI/patches/te.patch"

  echo "-- [te] building wheel (this is the slow step, ~45-60 min) --"
  export NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 NVTE_CUDA_ARCHS=90 MAX_JOBS=16
  export MPI_HOME=${MPI_HOME:-/usr/local/mpi}
  ( cd "$TE_SRC" && \
    pip --disable-pip-version-check --no-cache-dir wheel \
        --wheel-dir="$WHEELS" --no-deps --no-build-isolation . )

  rm -rf "$TE_SRC"
  ls -lh "$WHEELS"/transformer_engine-*.whl
  touch "$STAMPS/te.wheel.done"
fi

# TE build stage 2: install the wheel into $PREFIX.
# --ignore-installed: don't try to uninstall the older TE that lives in the container's
#   read-only system site-packages (2.4.0+3cd6870 in pytorch:25.06-py3).
# --no-deps: TE's deps (torch etc.) are already satisfied by the base image.
if [[ ! -f "$STAMPS/te.done" ]]; then
  echo "-- [te] installing wheel into $PREFIX --"
  pip install --prefix="$PREFIX" --ignore-installed --no-deps --no-cache-dir \
      "$WHEELS"/transformer_engine-*.whl
  python3 -c "import transformer_engine.pytorch as te; print('TE ok at', te.__file__)"
  touch "$STAMPS/te.done"
fi

# ---------------------------------------------------------------------------
# Stage 2: NeMo-Run (needed by NeMo pipeline configs, small).
# ---------------------------------------------------------------------------
if [[ ! -f "$STAMPS/nemo_run.done" ]]; then
  echo "-- [nemo-run] installing --"
  # --ignore-installed: don't try to uninstall pre-existing versions from the base image's
  # read-only /usr/local/lib/python3.12/dist-packages.
  pip install --prefix="$PREFIX" --ignore-installed hatchling urllib3
  pip install --prefix="$PREFIX" --ignore-installed \
    "nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@v0.3.0" \
    --use-deprecated=legacy-resolver
  touch "$STAMPS/nemo_run.done"
fi

# ---------------------------------------------------------------------------
# Stage 3: nvidia-resiliency-ext (Dockerfile installs this "because it doesn't yet
# have ARM wheels"). Build from source into $PREFIX.
# ---------------------------------------------------------------------------
if [[ ! -f "$STAMPS/resiliency.done" ]]; then
  echo "-- [resiliency-ext] installing from source --"
  # nvidia-resiliency-ext uses poetry-dynamic-versioning as its PEP-517 build backend.
  # Pre-installing to $PREFIX didn't help: pip's PEP-517 subprocess with --no-build-isolation
  # can't see the prefix's dist-packages. Simpler: let pip use build isolation for this
  # one install. resiliency-ext doesn't need torch at build time, so the temp venv is small.
  NRE_SRC=/tmp/nvidia-resiliency-ext
  rm -rf "$NRE_SRC"
  git clone https://github.com/NVIDIA/nvidia-resiliency-ext "$NRE_SRC"
  pip install --prefix="$PREFIX" --ignore-installed "$NRE_SRC"
  rm -rf "$NRE_SRC"
  touch "$STAMPS/resiliency.done"
fi

# ---------------------------------------------------------------------------
# Stage 4: the big install — NeMo, Megatron-LM, bionemo sub-packages, scanpy, +CVE/test pins.
# Mirrors the Dockerfile's `uv pip install` block. We use plain pip because uv isn't preinstalled;
# resolver is slower but this only runs once.
# ---------------------------------------------------------------------------
if [[ ! -f "$STAMPS/bionemo.done" ]]; then
  echo "-- [bionemo] patching bionemo-core to drop ngcsdk (protobuf conflict WAR) --"
  sed -i "/ngcsdk/d" "$MAXTOKI/sub-packages/bionemo-core/pyproject.toml"

  echo "-- [bionemo] pre-installing setup.py-time build deps --"
  # numcodecs (transitive dep via zarr → nemo_toolkit[llm]) imports `cpuinfo` at
  # setup.py time. With --no-build-isolation, pip uses container Python which doesn't
  # have it. Cython is also commonly needed. PYTHONPATH already covers $PREFIX.
  #
  # setuptools_scm pin: numcodecs sdists declare setup_requires=['setuptools_scm'].
  # Without a version pin, setuptools' fetch_build_eggs pulls the latest from PyPI,
  # which at the moment is a broken 10.3.4 that tries to `from vcs_versioning import
  # Configuration` and dies. Pre-installing <10 into $PREFIX makes pkg_resources
  # satisfy the setup_requires from working_set and skip the egg fetch entirely.
  pip install --prefix="$PREFIX" --ignore-installed \
      "py-cpuinfo" "Cython>=3" "setuptools_scm<10"

  echo "-- [bionemo] staging libsqlite3-dev headers into \$PREFIX --"
  # pyfastx (from requirements-cve.txt) compiles a C extension that includes sqlite3.h.
  # The pytorch:25.06-py3 image ships only libsqlite3-0 (runtime .so, no dev headers).
  # Extract the matching noble arm64 libsqlite3-dev deb (pre-staged in containers/
  # from the login node — ghx4 compute nodes can't reach ports.ubuntu.com directly;
  # see attempt 11 failure) into $PREFIX/sqlite3-dev. ABI is 3.45.1 either way.
  SQLITE_DEV_DIR=$PREFIX/sqlite3-dev
  # build_env.sbatch binds $DELTAAI_ROOT/containers at /workspace/containers so the
  # pre-staged .deb is visible to the container without needing compute-node network.
  SQLITE_DEV_DEB=/workspace/containers/libsqlite3-dev_3.45.1-1ubuntu2.9_arm64.deb
  if [[ ! -f "$SQLITE_DEV_DIR/usr/include/sqlite3.h" ]]; then
    test -f "$SQLITE_DEV_DEB" || { echo "!! pre-staged $SQLITE_DEV_DEB not found"; exit 1; }
    mkdir -p "$SQLITE_DEV_DIR"
    dpkg-deb -x "$SQLITE_DEV_DEB" "$SQLITE_DEV_DIR"
  fi
  export CPATH="$SQLITE_DEV_DIR/usr/include${CPATH:+:$CPATH}"
  # The .deb puts libsqlite3.so into usr/lib/aarch64-linux-gnu/ as a symlink to .so.0.
  # Add that dir to LIBRARY_PATH so `-lsqlite3` resolves at link time.
  export LIBRARY_PATH="$SQLITE_DEV_DIR/usr/lib/aarch64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"

  echo "-- [bionemo] installing NeMo 2.7.2 + Megatron + bionemo sub-packages --"
  # --ignore-installed avoids the read-only-fs error on uninstall attempts against the
  # base image's /usr/local/lib/python3.12/dist-packages (many pre-installed packages).
  # --prefer-binary: for any transitive dep that ships an aarch64 wheel, prefer it
  # over an sdist so we skip source builds (and avoid more setup.py-time surprises).
  pip install --prefix="$PREFIX" --ignore-installed --no-build-isolation --prefer-binary \
    "nemo_toolkit[llm]==2.7.2" \
    "$MAXTOKI/3rdparty/Megatron-LM" \
    "$MAXTOKI/sub-packages/bionemo-core" \
    "$MAXTOKI/sub-packages/bionemo-llm" \
    "$MAXTOKI/sub-packages/bionemo-maxtoki" \
    "$MAXTOKI/sub-packages/bionemo-testing" \
    scanpy \
    -r "$MAXTOKI/requirements-cve.txt" \
    -r "$MAXTOKI/requirements-test.txt"

  echo "-- [bionemo] CVE cleanup + ngcsdk reinstall --"
  # pip uninstall doesn't take --prefix; it uses sys.path, which already includes $SITE via
  # PYTHONPATH — so it only touches prefix-installed copies, not the read-only system ones.
  pip uninstall -y llama-index llama-index-core llama-index-legacy 2>/dev/null || true
  pip install --prefix="$PREFIX" --ignore-installed ngcsdk==3.64.3
  pip uninstall -y bitsandbytes 2>/dev/null || true
  # bitsandbytes 0.46.1 likely has no aarch64 wheel — try, fall back to skipping.
  pip install --prefix="$PREFIX" --ignore-installed bitsandbytes==0.46.1 \
    || echo "!! bitsandbytes install failed on aarch64 - skipping (inference doesn't need it)"
  pip uninstall -y sqlitedict zstandard 2>/dev/null || true

  echo "-- [bionemo] removing prefix torch/triton/nvidia_* to defer to base image --"
  # The big install drags in torch 2.14.1 + cuda 13 wheels as transitive deps, which
  # overshadow the base image's torch 2.8.0a0+5228986c39.nv25.06 via PYTHONPATH.
  # TE/megatron-core/bionemo were all compiled against torch 2.8's C++ ABI (because at
  # wheel-build time, $PREFIX was still empty of torch), so loading torch 2.14 at
  # runtime yields: `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`
  # on `import transformer_engine.pytorch`.
  # Delete the prefix copies so Python falls through to /usr/local/.../torch (base).
  rm -rf "$SITE/torch" "$SITE/torch-"*".dist-info" \
         "$SITE/torchvision" "$SITE/torchvision-"*".dist-info" \
         "$SITE/torchaudio" "$SITE/torchaudio-"*".dist-info" \
         "$SITE/triton" "$SITE/triton-"*".dist-info" \
         "$SITE"/nvidia_* "$SITE/nvidia"

  touch "$STAMPS/bionemo.done"
fi

echo "== install complete =="
echo "== validate imports =="
python3 - <<'PY'
import importlib, sys
for m in ["torch", "transformer_engine.pytorch", "megatron.core", "nemo",
         "bionemo.core", "bionemo.llm", "bionemo.maxtoki"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:<40} {getattr(mod, '__file__', '?')}")
    except Exception as e:
        print(f"  FAIL {m:<40} {type(e).__name__}: {e}")
        sys.exit(1)
print("all imports OK")
PY
echo "== done at $(date -Is) =="
