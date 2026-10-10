# NextCell on Delta — progress

## 2026-10-08

### Changed (all under `delta/`)
- `slurm/maxtoki_env.sh`
  - mask the image's `/usr/local/cuda/compat/lib` (libcuda 575.57.08) with an empty bind — it
    shadowed the host driver → `CUDA Error 803: unsupported display driver / cuda driver combination`.
  - bind `/work/nvme/bhdw/asachan` at the same path (user request); `~/.bashrc` points
    `XDG_CACHE_HOME`/`HF_HOME` there and `bash -lc` sources it inside the container.
- `slurm/_run_nextcell_tests.sh` — new, port of the ARM tests script (`bgdb-delta-gpu`, `gpuA100x4-interactive`).
- `slurm/_run_nextcell_pdk4_smoke.sh` — export `XDG_CACHE_HOME=/cache/xdg HF_HOME=/cache/hf` inside the
  container shell (after `.bashrc`); spec selectable with `NC_SPEC=...`; default spec is the Delta copy.
- `slurm/_run_nextcell_pdk4_full.sh` — new, port of the ARM full script (same fixes). **Not run; see findings.**
- `configs/pdk4_inhibit_nextcell_smoke_delta.yaml` — copy of the shared smoke spec with
  `generation.buffer_size_gb: 16.0` (40.0 preallocates the whole 40/48 GB card → OOM in flash-attn).
- `configs/pdk4_inhibit_nextcell_smoke10_delta.yaml` — same with `query.limit_n: 10` to fit 1 h interactive caps.
- `deltaai/NEXTCELL.md` §9 — what NextCell/the smoke give you vs TBC (user-requested write into `deltaai/`).

### Container facts
- `maxtoki-dev.sif` has bionemo-core/llm/maxtoki, NeMo 2.7.2, megatron-core 0.15.0rc8, TE 2.3.0 **baked in**
  (`/usr/local/lib/python3.12/dist-packages`). Binding the fork at `/workspace/bionemo2` does NOT shadow it.
  Baked copy (Apr 2026) differs from the fork (Sep 2026) only in the regression-head predict path and
  data_prep tweaks — not used by NextCell. `MAXTOKI_ENV` stays empty.
- torch 2.8.0a0+nv25.06, CUDA 12.9. Works on A100, A40, H200 once the compat lib is masked.

### Runs
| job | partition / GPU | result |
|---|---|---|
| 22764077 | gpuA100x4-interactive | `tests/test_nextcell_pipeline.py`: **25 passed** (57 s) |
| 22764104 | gpuA100x4, A100-40GB | FAILED: CUDA Error 803 (compat libcuda) |
| 22764153 | gpuA100x4, A100-40GB | FAILED: `Read-only file system: /work` (XDG cache from .bashrc) |
| 22764219 | gpuA100x4-interactive | FAILED: same (env override lost to `bash -l`) |
| 22764320 | gpuA40x4-interactive, A40 | FAILED: CUDA OOM — `buffer_size_gb=40` on 44 GiB |
| 22764362 | gpuA40x4-interactive | cancelled (user: move to H200) |
| 22764367 | gpuH200x8-interactive | 20-query smoke; cancelled at 11/20 baseline rows — would exceed 1 h cap. Gave timing. |
| **22764696** | gpuH200x8-interactive, `--cpus-per-task=2 --mem=32G` | **10-query smoke COMPLETED** 44 min. `out/nextcell_pdk4_inhibit_smoke_delta_22764696/` |

Smoke 22764696 numbers (first completed correct-grammar NextCell run anywhere):
- ~33 tok/s steady, **~2.1 min/row**, every row capped at 4096 tokens (`n_*_finished = 0`), `n_invalid = 0`.
- Peak GPU mem **20.9 GB** (30 s samples). Host RSS ~5.5 GB.
- mean Jaccard@50 0.77, @100 0.68, @500 0.57; mean Spearman(shared) 0.75 on ~3210 shared genes.
  Per-row Jaccard@50 ranges 0.67–1.00. Top ~20 genes identical baseline vs perturbed in every row.
- `gene_present` 10/10 (ctx_3 is shared → split is degenerate).

### Sanity gate (CPU, `delta/scripts/sanity_nextcell.py`) — FAILED
Generated rankings vs real cells: top-100 overlap **0.01** with the conditioning context cell,
**0.01** with the real OM target, 0.00 with random real OM/YM2 cells. Real-vs-real reference is
0.19–0.25. Real cells are muscle (NEB, TTN, MYBPC1…); generated are not (S100A12, FAM90A20P, ZNF534…).
Conclusion: zero-shot 217M NextCell output is not a cell; no PDK4 inference possible from it.
Full write-up: `delta/NEXTCELL_WHAT_WE_LEARNED.md` §3.1.

### Checkpoint provenance — Stage-1 only, NextCell/TBC tokens untrained
HF card: "pretrained on 175M single-cell transcriptomes … can be further trained … second stage".
HF `vocab_size: 20275` (no `<boq>/<eoq>`/Δt tokens); `import_hf.py` leaves the extra 3002 BioNeMo rows
at random init. Zero-shot NextCell (and zero-shot TBC) with this checkpoint reads untrained tokens.
Details + options: `delta/NEXTCELL_WHAT_WE_LEARNED.md` §3.2.

### Dictionary shift + re-randomization test (jobs 22766096, 22766317; H200)
`MaxToki-217M-bionemo/context/token_dictionary.json` has every gene id = HF id − 1 and `<eos>`/`<bos>`
swapped vs the weights (positional HF copy, verified max|diff|=0 on rows <20275; rows ≥20275 random
init). New aligned dict: `delta/configs/token_dictionary_217m_aligned.json`. PDK4-inhibit TBC
(`pdk4_evenly_seq8k.yaml`): A old dict −37.4 (reproduces May −37.7); B old dict + rows≥20275 re-drawn
−62.2 (per-cell baseline r = −0.14 vs A); C aligned −176; D aligned + re-drawn −682. NextCell with the
aligned dict emits `<eos>` immediately (0-gene cells). Scripts: `delta/slurm/_run_tbc_dict_test.sh`,
`delta/slurm/_torch_pipeline_entry_rerand.py` (RERAND_SEED), `delta/scripts/plot_tbc_dict_test.py`.
Figure: `delta/figs/tbc_dict_test_22766317.png`. Write-up: `delta/NEXTCELL_WHAT_WE_LEARNED.md` §3.3.

### Findings that change the plan
1. **Prompt depends only on Δt_q.** Rows 0 and 6 (different cells/donors, pseudotime 85.0) are
   byte-identical. Over 2000 OM cells `round(ptime-100)` takes **68 values** (−67…0). The 2000-query
   "full run" = 68 unique prompts × 2 ≈ 136 generations ≈ **5 GPU-h** on one H200 — but per-cell /
   per-donor statistics on it are copies. Details: `deltaai/NEXTCELL.md` §9.4.
2. Nothing ever emits `<eos>`; the ARM gate (≥18/20 natural completions) is not met. Half of each
   generation (tokens ~2000–4096) is past any real cell.
3. H200 interactive slots: free GPUs sit on nodes with ≤2 idle CPUs → request `--cpus-per-task=2`
   or the job queues for hours.

### Open questions for the parent
- Run the 68-distinct-Δt_q version instead of 2000 cells? (A dedup pass in `dataset_prep`, or a spec
  listing the Δt_q grid, would make it explicit.) Or redesign so the target cell enters the prompt?
- Lower `generation.max_tokens` to ~2200 (halves cost; only matters if `<eos>` is never reached)?
- A NextCell negative control is needed (gene absent from ctx_3).

### sbatch commands (from repo root)
```bash
sbatch delta/slurm/_run_nextcell_tests.sh                                        # 25 tests pass
# 10-query smoke, fits 1 h interactive H200 slot:
NC_SPEC=delta/configs/pdk4_inhibit_nextcell_smoke10_delta.yaml \
  sbatch --export=ALL --partition=gpuH200x8-interactive --time=01:00:00 \
         --cpus-per-task=2 --mem=32G delta/slurm/_run_nextcell_pdk4_smoke.sh
# 20-query smoke (needs ~90 min → batch partition):
sbatch --partition=gpuH200x8 --time=02:00:00 --cpus-per-task=2 --mem=32G delta/slurm/_run_nextcell_pdk4_smoke.sh
# full: delta/slurm/_run_nextcell_pdk4_full.sh — HOLD until finding 1 is decided
```
