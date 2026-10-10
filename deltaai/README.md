# DeltaAI setup for MaxToki + maxtoki-perturb

Workspace for building and running NVIDIA MaxToki (BioNeMo) on NCSA DeltaAI
(aarch64 Grace CPU + GH200). Companion to `docs/source/delta_recipes.rst`
(written for x86 Delta; a DeltaAI section is added once the ARM image works).

## Layout

- `containers/` — Apptainer `.def` files (tracked) and `.sif` images (git-ignored)
- `slurm/` — sbatch scripts for the image build and pipeline runs
- `src/` — clones of upstream `maxToki`, `TransformerEngine`, etc. (git-ignored)
- `weights/`, `data/`, `out/` — checkpoints, tokenized data, prediction outputs (git-ignored)
- `logs/` — captured build/run logs referenced from SETUP_LOG.md (git-ignored except summaries)
- `SETUP_LOG.md` — running log: exact image path, sha256, pinned commits, deviations, failures

## Cluster facts (verified 2026-09-28)

- Login: `gh-login02.delta.ncsa.illinois.edu` — aarch64
- Apptainer: 1.4.2-111.1, `--fakeroot` supported
- Slurm account: `bhdw-dtai-gh` (993 GPU-hr remaining of 1000)
- Partitions: `ghx4` (default, 2 d), `ghx4-interactive` (2 h), `full` (1 d), `test` (2 h)
- GPUs: 4 × `nvidia_gh200_120gb` per node
- Shared project dir: `/projects/bhdw/asachan/` (not `/work/`)

## Env for apptainer builds

```bash
export APPTAINER_CACHEDIR=/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai/containers/.cache
export APPTAINER_TMPDIR=/tmp
```

Home quota is only 100 GB — never let apptainer cache into `$HOME`.
