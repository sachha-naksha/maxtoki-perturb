# torch_pipeline — zero-shot gene-perturbation → temporal-MSE on the BioNeMo MaxToki

CUDA / x86 sibling of `maxtoki_mlx`. Runs **inside the BioNeMo container** on
A100 / H200, loading the upstream
[NVIDIA-Digital-Bio/maxToki](https://github.com/NVIDIA-Digital-Bio/maxToki)
distcp checkpoint directly. Temporal head = the headless-MSE regression of
`bionemo.maxtoki.model.MaxTokiFineTuneModel`.

## What you specify

Everything that matters per experiment lives in a YAML / JSON spec
(`spec.py:ExperimentSpec`):

| section | controls |
|---|---|
| `data` | h5ad path, pseudotime column, group column (donor / sample), cell-id column, count column |
| `context` | which past cells form the trajectory the model conditions on (per query) |
| `query` | which cells to score (every cell, or one per group, optionally `filter_obs`) |
| `perturbation` | the gene to KO (Ensembl ID or symbol), direction (inhibit / delete / overexpress), and whether the perturbation applies to the query only or to every cell in the row |
| `seq_length` | model context length (16384 default for trajectory tasks) |

Three working examples in `configs/`:

- **`example_self_baseline.yaml`** — cell-vs-itself, no pseudotime needed. Each query is its own context. Equivalent to the previous default.
- **`example_pseudotime_trajectory.yaml`** — every query sees up to 4 prior cells from the same donor, sorted by ascending pseudotime. The realistic temporal-prediction setup.
- **`example_fixed_reference.yaml`** — every query sees the SAME hand-picked reference cells as context. Useful for "compare 4111 cells against a canonical young-donor trajectory" experiments.
- **`example_young_context_old_query.yaml`** — `pool` strategy: pick 3 cells from the 34y donor as context, score every 80y cell as query. The aging-readout setup.

## Context strategies

| `context.strategy` | what it does | requires |
|---|---|---|
| `self` | context = `[query]` (cell vs. itself) | nothing |
| `prefix` | cells in same group with `pseudotime < query.pseudotime`, sorted asc, capped to `max_cells` (latest kept) | `data.group_col`, `data.pseudotime_col` |
| `all_in_group` | every cell in same group except the query, sorted by pseudotime, capped | `data.group_col` |
| `explicit` | the same `context.explicit_indices` for every query | `context.explicit_indices` |
| `pool` | filter cells by `pool_filter` (e.g. `donor_age: [34]`), sort, and pick `n` (first / last / evenly_spaced / random). Applied identically to every query — useful for "young-donor reference vs. old-donor query" | `pool_filter`, `pool_select` |

`include_self: true` adds the query to its own context (latest). Set
`apply_to: query_and_context` if the gene KO should be applied to every
cell in the row, not just the query.

## Pipeline

```
spec + h5ad
    │
    ▼
for each query cell q:
    ctx_cells = select_context(q, spec.context)
    base_row  = [<bos>, ctx_1, <eos>, …, <bos>, ctx_K, <eos>,
                 <boq>, q,            <eoq>, dummy_numeric]
    pert_row  = same, but with perturb(gene, direction) applied to q
                (and optionally to ctx_*)
    │
    ▼
save baseline.dataset / perturbed.dataset (HF, identical row order)
    │
    ▼
bionemo.maxtoki.predict.predict(...)  ×2   (headless TimeBetweenCells)
    │
    ▼
delta_t(q) = pert_t[q] - base_t[q]
per-row MSE = delta_t²
dataset MSE = mean(delta_t²)
```

The per-cell rank-value cap is computed dynamically per row from
`seq_length / (K+1)` so the worst-case row fits in context.

## Files

| file | role |
|---|---|
| `spec.py` | `ExperimentSpec` dataclass + YAML/JSON loader |
| `tokenizer.py` | rank-value tokenizer; reads full BioNeMo dict (boq/eoq/numeric tokens) via `MAXTOKI_TOKEN_DICT` env or `--tokenizer-path` |
| `perturbation.py` | `delete` / `inhibit` / `overexpress` token rewrites |
| `dataset_prep.py` | spec → paired HF datasets (baseline + perturbed) saved to disk |
| `predict_runner.py` | thin wrapper around `bionemo.maxtoki.predict.predict` (variant defaults below) |
| `score.py` | reads `predictions__rank_*.pt`, joins by row order, computes MSE |
| `run_inhibit_temporal_mse.py` | end-to-end CLI driver |
| `configs/*.yaml` | example specs |

## Variants — 217M and 1B

The Megatron architecture is read out of the distcp via
`load_settings_from_checkpoint`, so the same `MaxTokiMultitaskFineTuneConfig`
covers both — only `--ckpt-dir` changes. Defaults in `predict_runner.py`:

| variant | seq_length | micro_batch_size |
|---|---|---|
| 217m | 16384 | 4 |
| 1b   | 16384 | 1 (bump on H200 if VRAM allows) |

Both `seq_length` and `scale_factor` are in the upstream config's
`override_parent_fields`, so the values you pass actually override the
checkpoint defaults (necessary for 16k context with the right RoPE scaling).

## Run

```bash
# inside the BioNeMo container (bionemo + nemo + megatron available)
python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
    --spec scripts/torch_pipeline/configs/example_pseudotime_trajectory.yaml \
    --ckpt-dir /weights/maxtoki-1b-bionemo \
    --tokenizer-path /weights/maxtoki-1b-bionemo/context/token_dictionary.json \
    --variant 1b \
    --out-dir ./out/zlx1
```

CLI overrides for one-off changes (the spec YAML stays as your source of
truth): `--gene-symbol`, `--gene`, `--direction`, `--seq-length`, `--h5ad`.

`--prep-only` stops after dataset build (run on a CPU box to inspect token
sequences before burning GPU time). `--score-only` re-scores existing
prediction dirs without rebuilding.

## Outputs (`--out-dir`)

- `spec.resolved.json` — fully resolved spec + Ensembl ID + token id
- `baseline.dataset/`, `perturbed.dataset/` — HF datasets fed to BioNeMo. Each row has `input_ids`, `cell_id`, `group`, `query_pseudotime`, `context_pseudotimes`, `context_cell_ids`, `gene_present_in_query`, `condition`.
- `baseline_predictions/`, `perturbed_predictions/` — `predictions__rank_*.pt` from upstream `PredictionWriter`
- `scores.npz` — per-row `baseline`, `perturbed`, `delta_t`, `per_cell_mse`, `gene_present`
- `prep_summary.json`, `summary.json` — dataset-level summaries

## h5ad expectations

- raw counts in `adata.X` (or `adata.raw.X` if available)
- Ensembl IDs: `var["ensembl_id"]` / `var["feature_id"]` / ENSG-prefixed `var_names`
- pseudotime column (e.g. `obs["pseudotime"]`, `obs["age"]`, `obs["days_post_treatment"]`) — only required when `context.strategy` is `prefix` or `all_in_group`
- group column (e.g. `obs["donor_id"]`) — required by `prefix` and `all_in_group`
- optional `obs["n_counts"]` (else computed as `X.sum()` per row)

## Token dictionary

The `maxtoki_mlx` package ships a stripped dict (gene + special tokens, no
`<boq>` / `<eoq>` / numeric tokens). For the temporal head, **always** pass
`--tokenizer-path` pointing at the full `token_dictionary_v1.json` shipped
with the BioNeMo distcp checkpoint.
