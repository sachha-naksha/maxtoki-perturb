# Progress — NextCell on Aging SKM (DeltaAI)

Companion to `AGENT_NEXTCELL.md`. The DeltaAI agent appends to this file as it works
through §7.1 – §7.8 of the brief. Keep entries chronological. Do not commit heavy
artifacts (`.sif`, `.safetensors`, `.distcp`, generated-token dumps, h5ad); logs under
`deltaai/logs/` stay untracked.

---

## §7.1 DeltaAI baseline

Collected on login node (filesystem + git only; no bionemo/torch imports — those run in
sbatch on compute).

| | value |
|---|---|
| host | `gh-login03.delta.ncsa.illinois.edu` (DeltaAI) |
| repo | `maxtoki-perturb` @ `055837d` (branch `main`, before this implementation commit) |
| upstream `deltaai/src/maxToki` | `03722639a675f0faa42257b43a0bed9e01d98ae0` |
| container | `deltaai/containers/pytorch2506-arm64.sif` (12 GB) sha256 `3621c66e4a19703478453cc1e788d960589fce724d5065e26284442d84be1892` |
| env prefix | `/work/nvme/bhdw/asachan/maxtoki_env/` (bound to `/opt/env`; see `deltaai/slurm/maxtoki_env.sh`) |
| 217M checkpoint | `/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/` |
| &nbsp;&nbsp;`weights/common.pt` sha256 | `9d88b67b4cd9c2b73bf471d18ba94e1d9d833acce9dd4f645bde4cc1dd64d587` |
| tokenizer | `.../MaxToki-217M-bionemo/context/token_dictionary.json` sha256 `93c26fbba204afa8cafb2e1b2270aa45702dd97eb1c0e7654a693dd731416853` |
| vocab size | 23277 |
| special tokens | `<pad>=0`, `<eos>=2`, `<boq>=23274`, `<eoq>=23275`, `<bos>=23276` |
| PDK4 | ENSG00000004799 → token id 62 (resolved against the full dict above) |
| H5AD | `data/zero_shot/rna_zero_shot.preprocessed.h5ad` (283 MB, dated 2024-05-01) |

Donor counts, pseudotime range, and PDK4 presence per cell will be re-verified by the CPU
smoke-test job (deltaai/slurm/_run_nextcell_tests.sh) rather than from the login node.

## §7.2 Spec + generation config

Done in `scripts/torch_pipeline/spec.py`:
- `TaskType = Literal["time_between_cells", "next_cell"]`
- `GenerationSpec` dataclass (max_tokens, top_k, top_p, temperature, buffer_size_gb,
  buffer_guaranteed_fraction, chunk_size_tokens, buffer_overflow_factor)
- `ExperimentSpec.task_type` (default `"time_between_cells"`) and
  `ExperimentSpec.generation` fields
- `QuerySpec.limit_n` + `.seed` so the smoke config can deterministically sub-sample
- `validate()` floors `(seq_length - max_tokens - 1) // (K+1)` at 512; checks sampling bounds
- `to_dict` + `spec_from_dict` round-trip through `task_type` + `generation` + `query.limit_n`

Round-trip coverage lives in `tests/test_nextcell_pipeline.py::test_spec_roundtrip_*` and
`::test_spec_default_task_is_time_between_cells_when_omitted`. Will run in the test sbatch.

## §7.3 Prompt construction

Done in `scripts/torch_pipeline/dataset_prep.py`:
- `_build_input_ids(..., task_type=...)` branches:
  - `time_between_cells` → `... <boq> q_genes <eoq> dummy_numeric` (unchanged default)
  - `next_cell` → `... <boq> q_genes <eoq>` (no trailing dummy)
- `_per_cell_max_len(..., task_type, max_tokens_to_generate)` reserves generation budget
  under NextCell: `(seq_length - max_tokens - 1) // (K+1)`. Returns 3583 for the pinned
  config (seq=16384, max_tokens=2048, K=3).
- `build_paired_dataset` emits a `row_index` field and writes `row_manifest.json` next
  to the two HF datasets so the scorer can fail hard on any pairing drift.
- `_select_queries` applies `spec.query.limit_n` + `.seed` after `filter_obs`.

Grammar + arithmetic coverage in `tests/test_nextcell_pipeline.py::test_build_input_ids_*`
and `::test_per_cell_max_len_*`. CPU dataloader smoke against the real BioNeMo collator
will run on compute inside `_run_nextcell_tests.sh` (login node is pytest-only; the
collator check is implicitly exercised by the GPU smoke).

## §7.4 Predict wiring

Done in `scripts/torch_pipeline/predict_runner.py::run_headless_predict`:
- new kwargs `generate_next_cell`, `max_tokens_to_generate`, `top_k`, `top_p`,
  `temperature`, `buffer_size_gb`, `buffer_guaranteed_fraction`, `chunk_size_tokens`,
  `buffer_overflow_factor`
- all passed through to `bionemo.maxtoki.predict.predict`
- `generate_next_cell=False` remains the default (TimeBetweenCells path untouched)
- log line prints the generation knobs for debugging

1-GPU inspection (ragged-prompt check, greedy reproducibility) will happen under
`_run_nextcell_pdk4_smoke.sh` on compute.

## §7.5 NextCell scorer

New file `scripts/torch_pipeline/score_nextcell.py`:
- `_extract_per_row_tokens(predictions_dir)` normalizes the writer's nested per-batch
  structure into a flat list of `{tokens, length, finished}` dicts, iterating rank files
  in sorted order (handles ragged batches, degenerate single-row dicts).
- `decode_generation(tokens, id_to_ensg, specials, numeric_ids)` enforces the
  `<bos>, g1, …, <eos>` grammar: dedupes genes, skips special/numeric/invalid tokens,
  counts them, flags `<eos>` presence.
- `jaccard_at_k`, `spearman_on_shared` (hand-written Pearson on dense shared-rank
  positions — no scipy dep), `rank_shift` (perturbed − baseline, NaN for one-sided).
- `score_nextcell(baseline_dir, perturbed_dir, tokenizer_path, baseline_dataset_dir,
  out_path, gene_present_flags, target_ensg)` loads both prediction trees, loads the
  row manifest written by dataset_prep, fails hard on count mismatch, writes
  `scores_nextcell.npz` + `decoded_nextcell.json` + `summary_nextcell.json` with the
  §6 metrics split by (present vs absent) and (completed vs capped).

Driver `scripts/torch_pipeline/run_inhibit_temporal_mse.py` now branches on
`spec.task_type` for both predict kwargs and scoring (keeps Δt path byte-identical when
`task_type == "time_between_cells"`).

Metric sanity on hand-crafted paired lists in `tests/test_nextcell_pipeline.py` (identical
Jaccard, reversed Spearman, mixed rank_shift with NaN padding).

## §7.6 CPU tests + 20-query GPU smoke

CPU tests written (`tests/test_nextcell_pipeline.py`); will be exercised by:

```
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
sbatch slurm/_run_nextcell_tests.sh
```

20-query GPU smoke (deterministic subset, full-run prompt settings, 2048 generation cap):

```
cd /projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/deltaai
sbatch slurm/_run_nextcell_pdk4_smoke.sh
```

Expected outputs: `out/nextcell_pdk4_inhibit_smoke_<JOBID>/`:
- `baseline.dataset/`, `perturbed.dataset/`, `row_manifest.json`
- `baseline_predictions/`, `perturbed_predictions/` (one `predictions__rank_0.pt` each)
- `scores_nextcell.npz`, `decoded_nextcell.json`, `summary_nextcell.json`, `summary.json`

Smoke gate (§7.6 of brief): exactly 20 rows per condition, valid decoding, greedy
reproducibility across two reruns, ≥18/20 natural completions (`baseline_finished`/
`perturbed_finished` from `scores_nextcell.npz`).

## §7.7 Launch scripts + full-run sizing

Scripts committed:
- `scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_smoke.yaml` (20 queries, seed=42)
- `scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_full.yaml` (2000 queries)
- `deltaai/slurm/_run_nextcell_pdk4_smoke.sh` (ghx4, 1 GPU, 90 min, PDK4 inhibit)
- `deltaai/slurm/_run_nextcell_pdk4_full.sh` (ghx4, 1 GPU, 2 h — raise if smoke says so)
- `deltaai/slurm/_run_nextcell_tests.sh` (ghx4-interactive, 1 GPU, 30 min, pytest only)

Full-run wall-clock estimate pending the smoke measurement. **The Δt figure from
HANDOFF.md (~4 min / 2000 rows) is not a NextCell estimate** — Δt is one forward pass,
NextCell is up to 2048 forward passes with a growing KV cache. Size the full-run time
cap after seeing the smoke's tokens/sec.

## §7.8 Handback

See `deltaai/AGENT_NEXTCELL.md` §8 for the shape of the final handback block. The sbatch
commands above are the "ready to run" trio; append the smoke's job id + measured
tokens/sec + the first 3 paired top-20 decoded lists here after `_run_nextcell_tests.sh`
and `_run_nextcell_pdk4_smoke.sh` finish on compute. **Do not launch the 2000-query full
run** until the parent reviews the smoke results.
