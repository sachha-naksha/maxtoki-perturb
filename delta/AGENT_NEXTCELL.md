# Agent brief: NextCell on Delta (x86_64, A100/A40), sibling of `deltaai/`

You are a Claude agent on **NCSA Delta** (x86_64, `dt-login*.delta.ncsa.illinois.edu`),
working inside `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb`. This repo's
`deltaai/` tree is the ARM/GH200 counterpart; everything you touch belongs under
`delta/`. The pipeline, checkpoints, data, and bionemo-maxtoki source are already in
place — you are porting the ARM NextCell scripts to Delta hardware so smoke/full
runs can bypass ghx4 fairshare contention.

Keep a running `delta/PROGRESS_NEXTCELL.md`: what you changed, what you ran (job
ids), tokens/sec, peak GPU mem, open questions. Deliverable at the end of that file:
the exact `sbatch` commands the parent agent will run.

---

## 0. The shared-filesystem contract (read this before anything else)

Delta and DeltaAI mount the **same** `/projects`, `/u`, and `/work/nvme`. That means
you can — and must — reuse the data/checkpoint/Python source already in `deltaai/`,
but a Delta run can silently corrupt the ARM run if you share arch-specific state.
Rules:

| Path | Share? | Why |
|---|---|---|
| `/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/` | **share** | checkpoints are torch tensors, arch-agnostic |
| `./data/zero_shot/rna_zero_shot.preprocessed.h5ad` | **share** | h5ad, read-only |
| `./scripts/torch_pipeline/` (Python drivers, configs) | **share** | pure Python, arch-independent |
| `./deltaai/src/maxToki/` (bionemo-maxtoki fork) | **share, but bind-mount read-only** | editable source. See §2 on whether the x86 .sif has it baked in |
| `./deltaai/slurm/_torch_pipeline_entry.py` | **share** | pure Python shim (datasets fingerprint patch) |
| `/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif` | **Delta only** | amd64 — will refuse to run on gh-login* |
| `./deltaai/containers/pytorch2506-arm64.sif` | **DeltaAI only** | aarch64 — don't bind-mount from Delta |
| `/work/nvme/bhdw/asachan/maxtoki_env/` | **DeltaAI only** | ARM pip prefix (wheels have aarch64 .so). Never source from Delta |
| `/work/nvme/bhdw/asachan/cache/maxtoki/` | **split** | HF/Megatron caches. Use `cache/maxtoki_x86` on Delta so compiled kernels/triton caches don't cross-contaminate |
| `./out/nextcell_*` | **split by suffix** | the ARM scripts suffix outputs with `${SLURM_JOB_ID}`. Keep doing that; also prefix new runs with `_delta_` or `_x86_` for grep-ability |
| `__pycache__/` inside bind-mounted source | **set `PYTHONPYCACHEPREFIX`** | .pyc from Python 3.12-x86 vs 3.12-aarch64 are byte-compatible for pure modules but Python strips arch-specific tags only on CPython itself — don't trust it. Point `PYTHONPYCACHEPREFIX=/work/nvme/bhdw/asachan/.pycache_x86` in the sbatch |
| `./deltaai/logs/` | **don't write here** | use `./delta/logs/` |

If in doubt: **new directories under `delta/`, new cache dirs with `_x86` suffix, new
output subdirs with `_delta_` prefix.** Never write back into `deltaai/`.

---

## 1. Hardware + Slurm (Delta)

- Login: `ssh asachan@dt-login01.delta.ncsa.illinois.edu` (or dt-login02/03). Two-
  factor Duo is required.
- Account: **`bgdb-delta-gpu`** (confirmed from `/projects/bhdw/asachan/bash_scripts/jupyter_gpu.sbatch`).
- Partitions (GPU, visible to this account):
  - `gpuA100x4` — 4× A100 40 GB per node. Default target for NextCell 217M.
  - `gpuA100x8` — 8× A100 80 GB (one node, high contention). Only if 40 GB OOMs.
  - `gpuA40x4` — 4× A40 48 GB per node. Fallback if `gpuA100x4` is contested; A40 is slower but has more HBM.
  - `gpuA100x4-interactive`, `gpuA40x4-interactive` — 1 h cap, for debug.
- GPU request grammar on Delta is **`--gpus-per-node=1`** (same as DeltaAI). Memory per
  GPU scales with CPU allocation on these shared nodes — ask for `--cpus-per-task=16`
  and `--mem=64G` for a single-GPU 217M smoke, matching the `jupyter_gpu.sbatch`
  profile the user already runs.

Verify partitions and your access on first login:

```bash
sinfo --format="%P %a %l %D %T" | grep -E "gpu|cpu"
sacctmgr show user $USER withassoc format=User,Account,Cluster,Partition,QOS
```

---

## 2. Container

Use `/projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif` (amd64, 14 GB).

**Unknown that you must verify on first run** (I did not check from the ARM side):
whether this .sif has the bionemo-maxtoki fork from `deltaai/src/maxToki/` baked in,
or expects it bind-mounted the same way the ARM container does (`/workspace/bionemo2`).

Test both from a 1-GPU interactive session:

```bash
# (a) baked-in path — bind nothing extra:
apptainer exec --nv /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif \
    python -c "import bionemo.maxtoki; print(bionemo.maxtoki.__file__)"

# (b) if (a) fails with ModuleNotFoundError, bind the fork like the ARM scripts:
apptainer exec --nv \
    --bind /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai/src/maxToki:/workspace/bionemo2 \
    /projects/bhdw/asachan/app_envs/containers/maxtoki-dev.sif \
    python -c "import bionemo.maxtoki; print(bionemo.maxtoki.__file__)"
```

Record which path works in `delta/slurm/maxtoki_env.sh` (toggle `MAXTOKI_BIND_SRC=1`
in the sourcer). The provided sourcer defaults to **(b)** since it's a strict
superset: binding the fork on top of a baked-in copy only shadows it.

The ARM scripts also bind an external pip prefix at `/opt/env`. If `maxtoki-dev.sif`
on Delta was built with the deps baked in (likely, given the +1.5 GB vs ARM), that
bind is unnecessary. The provided sourcer leaves `MAXTOKI_ENV` empty on Delta by
default — set `MAXTOKI_ENV=/some/x86/prefix` only if (a) and (b) both fail with
`ModuleNotFoundError` on bionemo's deps.

---

## 3. What to port, in order

1. **`delta/slurm/maxtoki_env.sh`** — provided. Env sourcer, binds, apptainer args.
   Mirrors `deltaai/slurm/maxtoki_env.sh` with x86 paths and Delta-specific cache.
2. **`delta/slurm/_run_nextcell_pdk4_smoke.sh`** — provided. 1-GPU A100 smoke of
   `pdk4_inhibit_nextcell_smoke.yaml`. Writes to `out/nextcell_pdk4_inhibit_smoke_delta_${SLURM_JOB_ID}`.
3. `delta/slurm/_run_nextcell_tests.sh` — TODO. Mirror `deltaai/slurm/_run_nextcell_tests.sh` on a Delta interactive-partition reservation. Use `gpuA100x4-interactive`, 1 h cap.
4. `delta/slurm/_run_nextcell_pdk4_full.sh` — TODO. Multi-donor evenly-sampled run; 3-6 h cap on `gpuA100x4` depending on tokens/sec you measure in the smoke.
5. `delta/slurm/_run_nextcell_score_only.sh` — TODO. CPU-only scoring (`cpu` partition, `--cpus-per-task=8`, no GPU) that reads pre-generated `.npz`/`.pt` from either cluster's `out/`.

For (3)-(5), copy the ARM sibling, flip the two SBATCH headers (`--account`, `--partition`), swap `source slurm/maxtoki_env.sh` for `source delta/slurm/maxtoki_env.sh`, and change the output suffix.

---

## 4. Validate on first smoke, before posting the full-run numbers

- `nvidia-smi --query-gpu=name,memory.total --format=csv` → should report `NVIDIA A100-SXM4-40GB` (or 80 GB on x8).
- Inside the container: `python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda, torch.__version__)"` — flag if torch.version.cuda is <12.4.
- Verify checkpoints load (`bionemo.maxtoki.predict` import path lives at
  `scripts/torch_pipeline/predict_runner.py`'s `run_headless_predict`).
- Peak GPU memory for the 20-query smoke on 217M should be well under 40 GB. If it
  pushes 35+ GB, flag — the GH200 version had plenty of headroom and this is a
  behavioral regression, not a hardware limit.

---

## 5. If something goes wrong

- `the image's architecture (amd64) could not run on the host's (arm64)` → you're
  logged into a gh-login node. SSH to `dt-login01` instead.
- `ModuleNotFoundError: bionemo.maxtoki` → see §2(b), bind the fork.
- `pyarrow.MonthDayNano` pickle error → `_torch_pipeline_entry.py` already patches
  this. If it reappears, the x86 container may ship different `datasets`/`pyarrow`
  versions than the ARM one; adjust the patch's import guards, don't bypass it.
- `Permission denied` writing `./deltaai/logs/*` → you're writing to the ARM log
  dir. Use `./delta/logs/`.

End of brief. Fill in `delta/PROGRESS_NEXTCELL.md` as you go.
