# Handoff — MaxToki on DeltaAI (ARM)

Prior session got the ARM-native MaxToki stack running on DeltaAI (GH200 /
aarch64). Base image + NVMe user-space prefix. Imports pass, upstream pytest
passes (modulo 3 upstream flakes). Both 217M and 1B checkpoints are already
local, in both HF and bionemo distcp formats. Everything is ready to drive
inference.

Full context: `SETUP_LOG.md` (13-attempt build journey, root causes & fixes).

## What's in place

### Environment
- Base .sif: `containers/pytorch2506-arm64.sif` (NVIDIA pytorch:25.06-py3 aarch64, read-only, 12 GB). sha256 recorded in `containers/pytorch2506-arm64.sif.sha256`.
- Install prefix: `/work/nvme/bhdw/asachan/maxtoki_env/` — bind-mounted to `/opt/env` at runtime. On NVMe because `/projects/bhdw` inode quota is too tight (~935k hard, 770k used by other project members).
- maxToki source: `src/maxToki/` (git clone, @ `03722639a675f0faa42257b43a0bed9e01d98ae0`).
- Pre-staged apt deb (needed by install_prefix.sh): `containers/libsqlite3-dev_3.45.1-1ubuntu2.9_arm64.deb` (896KB; compute nodes can't reach ports.ubuntu.com).

### Checkpoints — all local, no download needed
```
/projects/bhdw/asachan/models/MaxToki/
├── MaxToki-217M-HF/         raw HF (model.safetensors, config.json, generation_config.json)
├── MaxToki-217M-bionemo/    distcp: weights/*.distcp + common.pt + .metadata, context/model.yaml + token_dictionary.json + io.json
├── MaxToki-1B-HF/           raw HF, 1B params
└── MaxToki-1B-bionemo/      distcp, 32 shards (__0_0 through __31_1)
```
The `-bionemo/` dirs are what `predict_runner.py` and bionemo-maxtoki inference scripts expect. No further conversion needed.

### Scripts
- `slurm/build_env.sbatch` — rebuilds the prefix from scratch (idempotent via stamps in `$MAXTOKI_ENV/.stamps/`). Takes ~15 min with the TE wheel preserved; ~45 min fully cold.
- `slurm/install_prefix.sh` — the actual install logic, run inside `apptainer exec`. Encodes every workaround (see SETUP_LOG).
- `slurm/maxtoki_env.sh` — **source this before running anything.** Defines `maxtoki_exec`.
- `slurm/pytest_maxtoki.sbatch` — smoke test template. Pattern worth copying for new inference jobs.
- `slurm/validate_env.sh` — ad-hoc import check.

## How to run things

```bash
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
source slurm/maxtoki_env.sh
maxtoki_exec python -c "import bionemo.maxtoki; print('ok')"
```

Inside a slurm batch: see `pytest_maxtoki.sbatch` — the `source maxtoki_env.sh`
+ `maxtoki_exec bash -lc '...'` pattern.

At runtime:
- `torch 2.8.0a0+5228986c39.nv25.06` resolves from the **base image** (NOT the prefix). This matters — the prefix's torch was intentionally rm'd so TE's `.so` loads against the ABI it was compiled against. If you ever see `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib` → someone put torch back into `$PREFIX/local/lib/python3.12/dist-packages/`. Delete it.
- Everything else (TE/megatron/nemo/bionemo) resolves from the prefix via `PYTHONPATH=/opt/env/local/lib/python3.12/dist-packages`.
- `TRITON_LIBCUDA_PATH=/.singularity.d/libs` is required (apptainer `--nv` drops libcuda there, not in ldconfig's cache).

## What's done (TODO ✓)

- [x] Validate imports on `ghx4-interactive` → torch + TE + megatron.core + nemo + bionemo.core/llm/maxtoki all import cleanly on GH200, CUDA detected.
- [x] Upstream pytest — 88 passed / 9 skipped / 3 upstream flakes (`test_data_prep.py` `MonthDayNano`/dill pickling; not ARM-related, would repro on x86 with same versions).
- [x] HF weights + conversion — both model sizes, both formats, pre-staged.

## What's next (TODO ✗)

In order of natural flow:

1. **TimeBetweenCells smoke prediction on 217M.** Smallest useful end-to-end check. Needs:
   - A tiny input dataset (ask user for a canonical one, or use anything in `src/maxToki/sub-packages/bionemo-maxtoki/examples/`).
   - Config pointing to `/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/`.
   - Launch via `maxtoki_exec python src/maxToki/<predict_runner_path>` under slurm.
   - Record tokens/sec and peak GPU mem.
2. **Wire `scripts/torch_pipeline/` for DeltaAI.** The repo's `scripts/torch_pipeline/` has shell scripts assuming a different cluster (account, partition, env setup). Need DeltaAI versions:
   - Account: `bhdw-dtai-gh`
   - Partition: `ghx4`
   - Env: `source deltaai/slurm/maxtoki_env.sh` + `maxtoki_exec ...`
   - Many existing scripts in that dir — `_run_5newgene_8k.sh`, `_resume_ak2_overexpress_8k.sh`, etc.
3. **Add NextCell mode to `predict_runner.py`.** Currently only TimeBetweenCells exists. User's eventual experiment is a NextCell sensitivity study.
4. **Go/no-go NextCell sensitivity experiment.** Depends on #3.

Deferred fix (not blocking): the 3 pytest flakes. Either pin `pyarrow<14` or add `--deselect` for `test_data_prep.py::TestDatasetUtils::test_smart_concatenate_*` + `TestE2EPipeline::test_tokenize_and_assemble`.

## Non-obvious gotchas to carry forward

- **Never put torch into the prefix.** Any `pip install --prefix=/opt/env` that doesn't pass `--no-deps` risks pulling torch as a transitive dep. If a new install breaks TE, check `ls /opt/env/local/lib/python3.12/dist-packages/ | grep -i torch` and rm anything shadowing the base.
- **Compute nodes have narrow egress.** PyPI works; `ports.ubuntu.com` and random mirrors don't. Stage artifacts from the login node into `containers/` (which is bound at `/workspace/containers`).
- **`/projects/bhdw` is inode-tight.** Keep large-file-count things (python envs, pytest caches, HF snapshots) on `/work/nvme/bhdw` or `/work/hdd/bhdw`. Project-wide limit is 935k inodes, with ~770k already used by other members — the installed env alone is ~250k files.
- **Apptainer binds don't preserve host parent dirs.** `/workspace/slurm/..` inside the container is `/workspace`, not `$DELTAAI_ROOT`. Bind each needed dir explicitly.
- **The ngcsdk sed patch to bionemo-core touches `src/maxToki/sub-packages/bionemo-core/pyproject.toml`.** It drops the `ngcsdk` dep to avoid a protobuf conflict (ngcsdk reinstalled separately later). If you ever reset the maxToki submodule, re-run from a fresh checkout — the stamp approach assumes idempotent re-runs so this is fine, but worth noting.

## Where to look if things break

- `logs/build_env.*.out` — slurm log for each build attempt (13 of them so far). Each failure is annotated in SETUP_LOG with root cause + fix.
- `$MAXTOKI_ENV/.stamps/` — stage markers. Delete to force re-run of a stage. Order: `te.wheel.done` → `te.done` → `nemo_run.done` → `resiliency.done` → `bionemo.done`.
- `$MAXTOKI_ENV/wheels/` — persistent built wheels (currently just TE). Survives install failures.
