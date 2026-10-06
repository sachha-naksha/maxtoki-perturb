# DeltaAI setup log

Chronological record of what was actually done to get MaxToki running on DeltaAI
(aarch64 / GH200). Any deviation from `docs/AGENT_MAXTOKI_DELTAAI.md` (the brief)
must land here, with commands/outputs so the next agent doesn't repeat mistakes.

## 2026-09-28

### Environment verification (login node `gh-login02`)

```
uname -m                            aarch64
apptainer --version                 apptainer version 1.4.2-111.1
apptainer build --help | grep -i fakeroot   -f, --fakeroot   build with the appearance of running as root
accounts                            bhdw-dtai-gh  balance 993 GPU-hr (of 1000 deposited)
sinfo -o "%P %l %D %G"              full/test/ghx4/ghx4-interactive, gpu:nvidia_gh200_120gb:4
quota /projects/bhdw                148.6G / 500G soft, 350+ GB free
quota /u/asachan (home)             2.6G / 100 GB soft
```

- `/projects/bhdw/asachan/maxtoki/` was not present. Chose to keep everything
  inside the repo under `deltaai/` (path: `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai/`)
  so that scripts and the setup log are version-controlled via the maxtoki-perturb
  GitHub remote. Large artifacts (`.sif`, weights, data, sandbox) are `.gitignore`d.

### Directory layout created

```
deltaai/
  .gitignore
  README.md
  SETUP_LOG.md
  containers/            # .def tracked, .sif / .cache ignored
    .cache/              # APPTAINER_CACHEDIR
  slurm/                 # sbatch scripts (tracked); slurm-*.out ignored
  src/                   # upstream clones, all ignored
  weights/  data/  out/  logs/    # ignored
```

## Deviations from the brief

### Fakeroot / subuid is NOT available on DeltaAI

Route A in the brief (`apptainer build --fakeroot maxtoki-arm64.sif maxtoki-arm64.def`) does
not work on this cluster:

```
$ grep "^$USER:" /etc/subuid   # empty; only "ansible" is listed
$ apptainer build --fakeroot _fakeroot_smoke.sif _fakeroot_smoke.def
INFO:    User not listed in /etc/subuid, trying root-mapped namespace
/.singularity.d/libs/fakeroot: eval: line 140: /.singularity.d/libs/faked: not found
fakeroot: error while starting the `faked' daemon.
FATAL:   While performing build: while running engine: while running %post section: exit status 1
```

This affects **every** `.def` with a `%post` section, on both login and compute nodes
(subuid is system-wide). `apptainer build` **without** `%post` (docker → SIF only) works
fine, and confirms the base image resolves to aarch64. `apptainer exec --writable` on a
sandbox also needs fakeroot.

Discussed options with user; chose the **user-space prefix install** (route A′ in the brief).

### Chosen strategy — user-space prefix install

1. Pull `nvcr.io/nvidia/pytorch:25.06-py3` as read-only `containers/pytorch2506-arm64.sif`
   (no `%post`, no fakeroot).
2. Create a user-owned prefix directory `deltaai/env/`.
3. Under `apptainer exec --nv --bind deltaai/env:/opt/env --bind src/maxToki:/workspace/bionemo2 pytorch2506-arm64.sif`,
   run `slurm/install_prefix.sh` — mirrors the Dockerfile's `bionemo2-base` stage but every
   `pip install` uses `--prefix=/opt/env` so writes go to the bound directory rather than
   the read-only image.
4. At runtime, source `slurm/maxtoki_env.sh`, which sets
   `APPTAINERENV_PYTHONPATH=/opt/env/lib/python3.12/site-packages` and
   `APPTAINERENV_PATH=/opt/env/bin:...`, and always adds the same `--bind` args.

TE, NeMo, Megatron, bionemo-* still install normally; they just land at a
user-writable location. Native performance is identical to a root-installed container.

Trade-off vs. a proper `.sif`: every runtime invocation must include the `--bind
env:/opt/env` and PYTHONPATH env (wrapped in `maxtoki_env.sh`). Not portable to
other clusters without carrying `env/` too. Fine for DeltaAI-only.

### Skipped from the Dockerfile

- `apt-get install libsndfile1 ffmpeg pre-commit sudo gnupg` — requires root inside the
  container. `git curl unzip libsqlite3-dev` are already present in `pytorch:25.06-py3`.
  MaxToki inference does not need audio/video libs; will revisit if imports fail.
- `aws-cli` build from source — MaxToki inference does not need it.
- The Dockerfile's `rm -rf /opt/pytorch/pytorch/third_party/onnx` security scan cleanup —
  can't modify the read-only base image; note as an accepted vulnerability for now.
- `uv` is used by the Dockerfile for install speed; we use plain pip (one-time install,
  speed doesn't matter). Fewer moving parts.

## Artifacts

| path | tracked? | purpose |
|---|---|---|
| `containers/pytorch2506-arm64.sif` | no (git-ignored) | Base image, read-only |
| `env/` | no | User-owned Python install prefix |
| `slurm/install_prefix.sh` | yes | Runs inside apptainer exec — installs TE/NeMo/Megatron/bionemo to /opt/env |
| `slurm/build_env.sbatch` | yes | Slurm submitter for the install |
| `slurm/maxtoki_env.sh` | yes | Sourceable runtime env: sets binds + PYTHONPATH |

### Base image pulled

```
$ ls -lh containers/pytorch2506-arm64.sif
-rwxr-xr-x  12G Sep 28 23:35 containers/pytorch2506-arm64.sif

$ apptainer exec containers/pytorch2506-arm64.sif bash -c \
    'uname -m && python3 -c "import sys,torch; print(sys.version.split()[0], \"torch\", torch.__version__)"'
aarch64
3.12.3 torch 2.8.0a0+5228986c39.nv25.06
```

`sha256(pytorch2506-arm64.sif) = 3621c66e4a19703478453cc1e788d960589fce724d5065e26284442d84be1892`
(also written to `containers/pytorch2506-arm64.sif.sha256`).

### Build attempt 1 — FAILED after 26 s on TE submodule init

Job 3260736 (ghx4, gh082). Full log at `logs/build_env.3260736.out`.

```
-- [te] cloning + patching TransformerEngine --
Cloning into '/tmp/TransformerEngine'...
... (cutlass, googletest, nccl-extensions submodule fetches from default branch) ...
fatal: not a git repository: ../../.git/modules/3rdparty/cudnn-frontend
fatal: could not reset submodule index
```

**Root cause.** The Dockerfile does
`git clone --recurse-submodules ... && git checkout --recurse-submodules ${TE_TAG}`
in one flow. On our git (2.43 in the base image + our host git for the initial clone),
this trips on a nested submodule (`cudnn-frontend`) that the pinned TE commit adds but
the default branch doesn't have. `--recurse-submodules` tries to touch a gitdir for it
before the submodule directory exists.

**Fix (applied to `install_prefix.sh`).** Split the checkout from submodule init:

```bash
git clone https://github.com/NVIDIA/TransformerEngine.git "$TE_SRC"
git -C "$TE_SRC" checkout "$TE_TAG"
git -C "$TE_SRC" submodule sync --recursive
git -C "$TE_SRC" submodule update --init --recursive
```

Verified on the login node — clones cleanly with just the 2 submodules TE actually
needs at this commit (`cudnn-frontend`, `googletest`; not `cutlass` or `nccl-extensions`
which live on later default-branch state).

### Build attempt 2 — FAILED at TE CMake (nvcc host compiler)

Job 3260858 (ghx4, gh067). Log at `logs/build_env.3260858.out`.

Two symptoms in the CMake CUDA-compiler-ID probe:

```
Build flags: -Wl,--version-script=/tmp/TransformerEngine/.../libtransformer_engine.version
nvcc fatal : Unknown option '-Wl,--version-script=...'
[CMake retries with empty flags]
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' ...
g++-14: No such file or directory
nvcc fatal : Failed to preprocess host compiler properties.
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

**Root causes.**

1. **g++-14 hardcoded in nvcc default.** `pytorch:25.06-py3` on aarch64 ships CUDA
   12.9.86, and nvcc's default `-ccbin` is `g++-14`, but the image only has `/usr/bin/g++`
   → `g++-13`. Verified: `apptainer exec pytorch2506-arm64.sif ls /usr/bin/g++*` returns
   only `g++` and `g++-13`.
2. **Host env leaking in.** My login shell exports `CC=cc`, `CXX=CC` (Cray-style HPC
   wrappers). These leak into the container via `apptainer exec` (which forwards host
   env by default). Inside the container, `CC` (uppercase binary) doesn't exist, so CMake's
   C++ compiler probe fails.
3. `-Wl,--version-script=...` is TE's linker script bleeding into the first CMake
   CUDA-compiler-ID probe. CMake retries with empty flags on the next probe, so this is
   self-recoverable once (1) and (2) are fixed.

**Fixes (applied).**

- `build_env.sbatch`: add `--cleanenv` to `apptainer exec` so host env is not forwarded.
  Only `APPTAINERENV_*` prefixed vars pass through (we set none for build; we do for
  runtime in `maxtoki_env.sh`).
- `install_prefix.sh`: after cleanenv, set `CC=/usr/bin/gcc CXX=/usr/bin/g++
  CUDAHOSTCXX=/usr/bin/g++ NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++'` at the top so all
  downstream compiles use the g++ that actually exists.

Verified on login node: `apptainer exec --nv --cleanenv <sif> nvcc -x cu /tmp/t.cu -o /tmp/t`
now compiles cleanly with those env vars set.

### Build attempt 3 — FAILED at TE install (read-only fs on uninstall)

Job 3260874 (ghx4), 3:38 elapsed. Log at `logs/build_env.3260874.out`.

**Surprise:** TE compiled in ~2 min on 32 cores with `NVTE_CUDA_ARCHS=90 MAX_JOBS=16`
— much faster than the "~45-60 min" folklore. Good to know for the go/no-go experiment.

```
Building wheel for transformer_engine (setup.py): finished with status 'done'
Created wheel for transformer_engine: transformer_engine-2.3.0.dev0+9d4e11ea-cp312-cp312-linux_aarch64.whl  (66 MB)
Installing collected packages: transformer_engine
  Attempting uninstall: transformer_engine
    Found existing installation: transformer_engine 2.4.0+3cd6870
    Uninstalling transformer_engine-2.4.0+3cd6870:
ERROR: Could not install packages due to an OSError: [Errno 30] Read-only file system: 'INSTALLER'
```

**Root cause.** pip found an older TE (2.4.0+3cd6870) in the container's system
`/usr/local/lib/python3.12/dist-packages/transformer_engine/`, tried to uninstall it to
avoid a version conflict, and failed because the base image is read-only. Same failure
mode will hit any package with a pre-existing version in the base image.

**Fixes (applied to install_prefix.sh).**

- **Persistent wheel builds.** TE stage split into two sub-stages:
  `te.wheel.done` = wheel built into `$PREFIX/wheels/` (survives install failures) →
  `te.done` = wheel installed. If install fails, next run skips the compile.
- **`--ignore-installed` on every `pip install --prefix` call.** pip no longer tries to
  uninstall pre-existing versions from the read-only base image; it just writes fresh
  copies into `$PREFIX`. Applied to all four stages (TE, nemo-run, resiliency-ext, bionemo).
  Costs extra disk (~10-20 GB total in the prefix) but avoids the whole failure class.
- TE install also uses `--no-deps` since its deps (torch etc.) are already in the base
  image and don't need re-resolution.

### Build attempt 4 — FAILED at TE import test (two bugs)

Job 3260935 (ghx4, gh014). Log at `logs/build_env.3260935.out`.

TE wheel built successfully (~2 min), installed successfully. Post-install import test
failed with:

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/transformer_engine/__init__.py" ...
  ...
  File ".../triton/backends/nvidia/driver.py", line 41, in libcuda_dirs
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
AssertionError: libcuda.so cannot found!
Possible files are located at ['/usr/local/cuda/compat/lib/libcuda.so.1'].
```

**Two overlapping bugs:**

1. **Debian pip layout.** `pip install --prefix=/opt/env` on Ubuntu 24.04 Python 3.12
   installs to `/opt/env/**local/lib/python3.12/dist-packages/**`, NOT
   `/opt/env/lib/python3.12/site-packages/`. My `PYTHONPATH` was pointing at the (empty)
   `site-packages/` dir, so Python fell through to the container's system TE (v2.4.0
   without the cudnn-frontend patch) at `/usr/local/lib/python3.12/dist-packages/`.
   Verified by finding TE files under `env/local/lib/python3.12/dist-packages/transformer_engine/`.

2. **Triton libcuda lookup.** Triton uses `ldconfig -p` (which reflects the ld.so cache,
   not `LD_LIBRARY_PATH`) to find `libcuda.so.1`. Inside apptainer with `--nv`, the host
   driver is bound at `/.singularity.d/libs/libcuda.so.1` — not visible to ldconfig.
   Triton's `driver.py` honors `TRITON_LIBCUDA_PATH` as an override.

**Fixes (applied):**

- `install_prefix.sh`, `maxtoki_env.sh`, `validate_env.sh`: change all
  `PYTHONPATH`/`PATH` references from `$PREFIX/lib/<py>/site-packages` and `$PREFIX/bin`
  to `$PREFIX/local/lib/<py>/dist-packages` and `$PREFIX/local/bin:$PREFIX/bin`.
- Export `TRITON_LIBCUDA_PATH=/.singularity.d/libs` in both `install_prefix.sh` (build-
  time) and `maxtoki_env.sh` (runtime, via `APPTAINERENV_TRITON_LIBCUDA_PATH`).

### Build attempt 5 — FAILED at resiliency-ext (PEP 517 backend missing)

Job 3260972 (ghx4, gh064). Log at `logs/build_env.3260972.out`.

TE install ok (path fix + `TRITON_LIBCUDA_PATH` worked). nemo-run ok. Resiliency-ext failed:

```
Cloning into '/tmp/nvidia-resiliency-ext'...
Processing /tmp/nvidia-resiliency-ext
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
ERROR: Exception:
  ...
  pip._vendor.pyproject_hooks._impl.BackendUnavailable:
    Cannot import 'poetry_dynamic_versioning.backend'
```

**Root cause.** `nvidia-resiliency-ext` declares `poetry-dynamic-versioning` as its
PEP-517 `build-backend`. With `--no-build-isolation`, pip uses the container's Python
which doesn't have that backend installed.

**Fix (applied).** Before the resiliency-ext install, `pip install --prefix $PREFIX
--ignore-installed 'poetry-dynamic-versioning>=1.0.0'`. Keeps `--no-build-isolation` so
we reuse the container's torch during build.

### Build attempt 6 — FAILED same resiliency-ext PEP-517 error

Job 3260991. Same error. `poetry-dynamic-versioning-1.10.0` DID install to
`/opt/env/local/lib/python3.12/dist-packages/`, but pip's PEP-517 hook subprocess
(spawned with `--no-build-isolation`) couldn't import it — pip's build_env has
subtle handling that hides the prefix's dist-packages from the subprocess.

**Fix (applied).** Drop `--no-build-isolation` for the resiliency install. Pip fetches
poetry-dynamic-versioning into a temp build venv, builds resiliency-ext, installs the
wheel into `$PREFIX`. resiliency-ext doesn't need torch at build time, so the isolation
overhead is small.

### Build attempt 7

Job 3261000. Log at `logs/build_env.3261000.out`.

### Build attempts 8–?: FAILED on numcodecs sdist (setuptools_scm 10.3.4 bug)

Job 3261062 (Sep 29 00:30). Log at `logs/build_env.3261062.out`.

TE/nemo_run/resiliency installs all stamped done. Pip resolver walked numcodecs
sdists 0.15.1 → 0.15.0 → 0.13.1 → 0.13.0 → 0.12.1 → 0.12.0 → 0.11.0 → 0.10.2
(no aarch64 wheels published for any numcodecs version at py3.12), and each one
died in `setup.py` with:

```
.eggs/setuptools_scm-10.3.4-py3.12.egg/setuptools_scm/__init__.py", line 8, in <module>
    from vcs_versioning import Configuration
ModuleNotFoundError: No module named 'vcs_versioning'
```

**Root cause.** numcodecs declares `setup_requires=['setuptools_scm']` with no
upper bound. Setuptools' `fetch_build_eggs` pulled the latest from PyPI —
`setuptools_scm 10.3.4` — which (apparently a broken release) tries to
`from vcs_versioning import Configuration` at import time and dies because the
`vcs_versioning` package isn't declared as a dependency.

**Fix (applied to install_prefix.sh).** Pre-install `setuptools_scm<10` into
`$PREFIX` as part of the bionemo-stage pre-install step. With PYTHONPATH
covering `$SITE`, setuptools' `fetch_build_eggs` resolves the setup_requires
against `pkg_resources.working_set` from the pre-installed 8.x copy and skips
the egg fetch. Also added `--prefer-binary` to the main bionemo install so any
transitive dep with an aarch64 wheel skips its sdist.

### Build attempt 9 — FAILED on pyfastx (sqlite3.h missing)

Job 3315663, 2026-10-05, ~4:27 elapsed. Log at `logs/build_env.3315663.out`.

Confirmed the setuptools_scm fix worked: numcodecs 0.15.1 built to a wheel
(`numcodecs-0.15.1-cp312-cp312-linux_aarch64.whl`). Resolver + wheel builds
proceeded through megatron-core, bionemo-core/llm/maxtoki/testing, pytest-dependency,
jieba, asciitree, wget, numcodecs. Then pyfastx died:

```
src/fakeys.h:5:10: fatal error: sqlite3.h: No such file or directory
    5 | #include "sqlite3.h"
```

**Root cause.** The pytorch:25.06-py3 base image installs `libsqlite3-0`
(runtime `.so.0` only) but not `libsqlite3-dev`. The SETUP_LOG's earlier claim
that `libsqlite3-dev` was already present in the image was wrong — only the
runtime lib is there. pyfastx 1.1.0 (pinned in requirements-cve.txt) builds a
C extension that `#include "sqlite3.h"` which doesn't exist in the image.

**Fix (applied to install_prefix.sh).** Before the bionemo install, fetch the
Ubuntu 24.04 noble arm64 `libsqlite3-dev_3.45.1-1ubuntu2.9` .deb from
`ports.ubuntu.com`, extract with `dpkg-deb -x` into `$PREFIX/sqlite3-dev/`, and
add its include + lib dirs to `CPATH`/`LIBRARY_PATH`. Headers match the ABI of
the already-installed `libsqlite3.so.0.8.6` (both are 3.45.1 upstream).

### Build attempt 10 — FAILED on disk quota (inodes, not blocks)

Job 3315699, 2026-10-05, 10:31 elapsed. Log at `logs/build_env.3315699.out`.

Both earlier fixes worked: `numcodecs-0.15.1-cp312-cp312-linux_aarch64.whl` and
`pyfastx-1.1.0-cp312-cp312-linux_aarch64.whl` both built. Pip resolved every dep
and started installing the final ~250 wheels into `$PREFIX`. Partway through:

```
ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded:
   '/opt/env/local/lib/python3.12/dist-packages/awscli/examples/memorydb/create-acl.rst'
```

**Root cause.** `/projects/bhdw` has an **inode (file-count) quota**, not just a
block quota. State at failure:
- block: 180G used / 500G soft / 550G hard (fine)
- **files: 932,952 used / 850,000 soft / 935,000 hard (over soft, hitting hard)**

Of the ~933k used inodes, only ~140k were asachan's — the other ~790k belong
to other `delta_bhdw` members, so no amount of our-side cleanup gets us enough
headroom for a 250k-file Python prefix install.

**Fix (applied).** Move `env/` from `/projects/bhdw/.../deltaai/env` to
`/work/nvme/bhdw/asachan/maxtoki_env`. `/work/nvme/bhdw` has a 2.55M file soft
limit (currently 332k used) and is NVMe, which is also faster for small-file IO.
`env/wheels/transformer_engine-*.whl` is preserved on the new location and
`.stamps/te.wheel.done` is pre-touched so the next build skips the TE compile
(2 min saved); all other stages rebuild from scratch.

Edits:
- `slurm/build_env.sbatch`: `ENV_DIR=/work/nvme/bhdw/asachan/maxtoki_env`.
- `slurm/maxtoki_env.sh`: `MAXTOKI_ENV` default points to the NVMe path.
- Deleted `/projects/.../deltaai/env` entirely (freed ~93k inodes to 840k used,
  back under the 850k soft limit).

### Build attempt 11 — FAILED on sqlite3-dev curl (compute-node network)

Job 3315952, 2026-10-05, 10:38 elapsed. Log at `logs/build_env.3315952.out`.

NVMe move successful: TE install (from preserved wheel, no compile) +
nemo_run + resiliency all stamped in ~5 min. Then bionemo stage reached its
new sqlite3-dev staging step and failed on the curl:

```
-- [bionemo] staging libsqlite3-dev headers into $PREFIX --
curl: (56) Recv failure: Connection reset by peer
```

**Root cause.** `ghx4` compute nodes evidently can't reach `ports.ubuntu.com`
(or the connection was torn down mid-transfer). PyPI/HuggingFace etc. are
reachable via the cluster's egress, but random Ubuntu-archive mirrors aren't
guaranteed to be. The login node (which we tested manually) has broader
egress.

**Fix (applied).** Pre-stage the .deb from the login node into
`deltaai/containers/libsqlite3-dev_3.45.1-1ubuntu2.9_arm64.deb` (896KB,
sha256 `a6865483003fb59a7beae16dad9436b8e37cd36d397457eae3b714a59f54eff3`).
`install_prefix.sh` now `dpkg-deb -x`'s from that path (reached via the
`/workspace/slurm/..` bind mount — slurm dir is bound at `/workspace/slurm`,
so `/workspace/slurm/../containers` resolves to `$DELTAAI_ROOT/containers`).
No network dependency on the compute node.

### Build attempt 12 — FAILED on bind-path mismatch

Job 3316360, 21s. Log at `logs/build_env.3316360.out`.

```
-- [bionemo] staging libsqlite3-dev headers into $PREFIX --
!! pre-staged /workspace/slurm/../containers/libsqlite3-dev_3.45.1-1ubuntu2.9_arm64.deb not found
```

**Root cause.** `/workspace/slurm/..` inside the container is `/workspace`, not
`$DELTAAI_ROOT` — only `slurm/` was bind-mounted at `/workspace/slurm`, so `..`
goes up the *container's* path, not the host's. Pre-staged .deb lives in
`$DELTAAI_ROOT/containers/` which wasn't visible to the container.

**Fix (applied).**
- `build_env.sbatch`: add `--bind "$DELTAAI_ROOT/containers":/workspace/containers`.
- `install_prefix.sh`: reference the .deb as `/workspace/containers/libsqlite3-dev_*.deb`.

### Build attempt 13 — install SUCCESS, import failed on torch ABI shadow

Job 3316410, 2026-10-05 16:19 → 16:28, 9:26 elapsed. Log at `logs/build_env.3316410.out`.

Install completed cleanly for the first time:
- Preserved stamps skipped TE compile + nemo_run + resiliency (saved ~5 min)
- sqlite3-dev staged from pre-fetched .deb via `/workspace/containers/` bind
- Big bionemo install succeeded on NVMe (line 1255: `Successfully installed ... bionemo-core-2.4.4 bionemo-llm-2.4.5 bionemo-maxtoki-2.4 bionemo-testing-2.4.1 megatron-core-0.15.0rc8 nemo_toolkit-2.7.2 ...`)
- ngcsdk-3.64.3 reinstall OK
- bitsandbytes-0.46.1 install OK
- `== install complete ==`, bionemo.done stamp written

Import validation then failed:

```
OK   torch   /opt/env/local/lib/python3.12/dist-packages/torch/__init__.py
FAIL transformer_engine.pytorch   ImportError: /opt/env/.../transformer_engine_torch.cpython-312-aarch64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

**Root cause.** The big install's dep resolver dragged in `torch 2.14.1` (bitsandbytes
requires `torch>=2.2`; several bionemo deps also allow any `torch>=2`) and the entire
`nvidia_* cu13` stack as transitive deps. These got written to `$PREFIX/local/lib/.../`
and, via `PYTHONPATH=/opt/env/local/lib/python3.12/dist-packages:...`, shadowed the
base image's `torch 2.8.0a0+5228986c39.nv25.06` at `/usr/local/lib/python3.12/dist-packages/torch`.

But TE/megatron-core/bionemo wheels were all **built** against `torch 2.8`: at
wheel-build time, pip had only resolved + collected the sdists; `$PREFIX` was still
free of torch, so `--no-build-isolation` built everything using the container's
torch 2.8 ABI (symbol `c10::cuda::c10_cuda_check_implementation(...)` from 2.8).
Loading those `.so`s against torch 2.14 at runtime → the undefined symbol.

**Fix (applied — install_prefix.sh + manual cleanup).** After the big install (and
after the ngcsdk reinstall), delete the prefix copies of `torch*`, `torchvision*`,
`torchaudio*`, `triton*`, and `nvidia_*`. Python's import then falls through to
`/usr/local/lib/python3.12/dist-packages/torch` from the base image — matching the
ABI the wheels were built against. Codified before `touch $STAMPS/bionemo.done`.

Manual cleanup also applied to the current `$PREFIX` directly.

**Validation (ghx4-interactive, gh092 ad hoc srun):**

```
OK   torch                        /usr/local/lib/python3.12/dist-packages/torch/__init__.py
OK   transformer_engine.pytorch   /opt/env/local/lib/python3.12/dist-packages/transformer_engine/pytorch/__init__.py
OK   megatron.core                /opt/env/local/lib/python3.12/dist-packages/megatron/core/__init__.py
OK   nemo                         /opt/env/local/lib/python3.12/dist-packages/nemo/__init__.py
OK   bionemo.core                 /opt/env/local/lib/python3.12/dist-packages/bionemo/core/__init__.py
OK   bionemo.llm                  /opt/env/local/lib/python3.12/dist-packages/bionemo/llm/__init__.py
OK   bionemo.maxtoki              None   # namespace package
torch.__version__ = 2.8.0a0+5228986c39.nv25.06
torch.cuda.is_available() = True
```

ARM maxtoki environment on DeltaAI: **OPERATIONAL**.

### Smoke test — upstream pytest

Jobs 3316530 (`-x`, hit first failure) + 3316576 (full run).
Script: `slurm/pytest_maxtoki.sbatch` (resubmit via `sbatch slurm/pytest_maxtoki.sbatch`).

Final result on GH200 (gh062), 46.6 s: **88 passed, 9 skipped, 3 failed**.

All 3 failures are in `test_data_prep.py` with the same root cause — a
`_pickle.PicklingError: Can't pickle <class 'MonthDayNano'>` chain through
`datasets.utils._dill` → `dill` → `pyarrow`'s `MonthDayNano` type (recursive
self-reference). Not an ARM/build issue — would reproduce on x86 with the same
`datasets`/`pyarrow`/`dill` versions.

```
FAILED test_data_prep.py::TestDatasetUtils::test_smart_concatenate_mismatched_dtypes
FAILED test_data_prep.py::TestDatasetUtils::test_smart_concatenate_matching_dtypes
FAILED test_data_prep.py::TestE2EPipeline::test_tokenize_and_assemble
```

Everything else passing — notably `test_sdpa_attention.py` forward/backward
(TE/Megatron attention equivalence under CUDA on GH200), `test_collate.py`,
`test_conversion.py`, `test_generate_utils.py`, `test_train.py`.

Deferred: pin pyarrow or `--deselect` the 3 flakes once they block something.

### torch_pipeline wiring — PDK4 inhibit 8k smoke test (2026-10-06)

First end-to-end run of `scripts/torch_pipeline/run_inhibit_temporal_mse.py`
on DeltaAI, mirroring the x86 `_run_5newgene_8k.sh` pattern against the
aging-SkM `rna_zero_shot.preprocessed.h5ad` with the PDK4 inhibit 8k
config. Verifies dataset_prep → `bionemo.predict` × 2 → score → viz across
the whole prefix env.

Artifacts added:
- `slurm/torch_pipeline_pdk4_8k.sbatch` — ghx4 sbatch (bhdw-dtai-gh, 1×GH200,
  16 cpus, 96 GB, 2 h). Sources `maxtoki_env.sh`, adds `--bind
  $PERTURB_DIR:/workspaces/maxToki` + `--bind /projects/bhdw/asachan` so the
  config's `./data/...` relative paths and the model checkpoint absolute
  path both resolve inside the container.
- `slurm/_torch_pipeline_entry.py` — thin wrapper. Replaces
  `datasets.generate_fingerprint` in BOTH `datasets.fingerprint` and
  `datasets.arrow_dataset` namespaces with a schema-hash function, then
  `runpy`-launches the real driver. Needed because the deferred
  `MonthDayNano` pickling flake (above) blocks `Dataset.from_list` on
  pyarrow 25 + datasets 5 + dill 0.4 — the fingerprint is only used as a
  cache key, so a deterministic schema hash is a safe substitute. Shared
  pipeline code untouched.

First attempt (job 3323023) died at 1:33 on the fingerprint pickle, with
the same `builtins.MonthDayNano` chain as the pytest flakes. Retry
(job 3323051) with the entry wrapper completed cleanly in 4:01, peak RSS
16.4 GB, 0 cuda OOM.

Result summary (`out/pdk4_217m_inhibit_evenly_seq8k_deltaai/summary.json`):

```
n_rows = 2000  (OM6 + OM9, 80 y/o queries, 3-cell YM2 context)
n_rows_with_gene_in_query = 983
mean_mse = 11458 ; mean_mse_present = 23312
mean_delta_t = -37.4 ; mean_delta_t_present = -76.1
seq_length = 8192 ; variant = 217m
```

Negative Δt on the gene-present subset matches the direction expected from
the x86 PDK4-inhibit runs (inhibit → earlier predicted temporal position →
rejuvenation signal).

## TODO — remaining setup
- [x] Validate imports on `ghx4-interactive`.
- [x] Run upstream `pytest sub-packages/bionemo-maxtoki/tests` (88 / 9 / 3; 3 fails are an upstream pyarrow/dill compat issue, not ARM).
- [x] HF weights download + conversion. Checkpoints live at
      `/projects/bhdw/asachan/models/MaxToki/` with both `*-HF/` (raw safetensors)
      and `*-bionemo/` (pre-converted distcp) layouts for 217M and 1B.
- [x] TimeBetweenCells smoke prediction — verified via torch_pipeline PDK4
      inhibit 8k run (job 3323051, 4:01 on GH200). NextCell variant still pending.
- [x] Wire up `scripts/torch_pipeline/` for DeltaAI (`ghx4`, account, prefix env)
      — `slurm/torch_pipeline_pdk4_8k.sbatch` + `_torch_pipeline_entry.py`.
- [ ] Add NextCell mode to `predict_runner.py`.
- [ ] Go/no-go NextCell sensitivity experiment.

## Pinned versions (re-verified against upstream Dockerfile at commit 0372263)

- `maxToki` @ `03722639a675f0faa42257b43a0bed9e01d98ae0`  ✓ matches brief
- `TransformerEngine` @ `9d4e11eaa508383e35b510dc338e58b09c30be73` + `patches/te.patch`  ✓
- Base image `nvcr.io/nvidia/pytorch:25.06-py3` (multi-arch)  ✓
- `nemo_toolkit[llm]==2.7.2` (Dockerfile ARG `NEMO_VERSION`)  ✓
- `nemo_run @ v0.3.0` (Dockerfile ARG `NEMU_RUN_TAG` — sic)  ✓
- `ngcsdk==3.64.3`, `bitsandbytes==0.46.1` (bnb: likely no aarch64 wheel; may drop)
- `Megatron-LM` submodule @ `bf1a5035f1f776b0bded8bffa0a36eeb573a7a8e`
