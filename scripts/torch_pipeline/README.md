# torch_pipeline — zero-shot gene inhibition → temporal MSE on the BioNeMo MaxToki model

CUDA / x86 sibling of `maxtoki_mlx`. Runs **inside the BioNeMo container**
on A100 / H200 (e.g. NCSA Delta), loading the upstream
[NVIDIA-Digital-Bio/maxToki](https://github.com/NVIDIA-Digital-Bio/maxToki)
distcp checkpoint directly. No HF backbone — the temporal regression head
is the headless-MSE head from `bionemo.maxtoki.model.MaxTokiFineTuneModel`.

## Pipeline

```
h5ad ─► rank-value tokenize each cell ─► for each cell c:
          baseline  = [<bos>, c, <eos>, <boq>, c,             <eoq>, dummy_num]
          perturbed = [<bos>, c, <eos>, <boq>, perturb(c, g), <eoq>, dummy_num]
        save as two HuggingFace datasets in identical row order
                 │
                 ▼
        bionemo.maxtoki.predict.predict(...) twice
        (headless TimeBetweenCells task, regression at <eoq>)
                 │
                 ▼
        delta_t(c) = pert_t[c] - base_t[c]
        per-cell MSE = delta_t²
        dataset MSE = mean(delta_t²)
```

The model is the multitask-finetuned MaxToki, so the `<eoq>` regression
output is the headless time-between-cells prediction. Asked baseline
"how much time elapsed between c and c?" → ≈0. Asked perturbed
"how much time elapsed between c and c-with-gene-g-inhibited?" → the model's
zero-shot estimate of the perturbation's aging effect.

## Files

| file | role |
|---|---|
| `tokenizer.py` | rank-value tokenizer; reads full BioNeMo dict (boq/eoq/numeric tokens) when `MAXTOKI_TOKEN_DICT` is set or `--tokenizer-path` is passed |
| `perturbation.py` | `delete` / `overexpress` / `inhibit` token rewrites |
| `dataset_prep.py` | h5ad → paired HF datasets (baseline + perturbed) saved to disk |
| `predict_runner.py` | thin wrapper around `bionemo.maxtoki.predict.predict` (variant defaults below) |
| `score.py` | reads `predictions__rank_*.pt`, joins by row order, computes MSE |
| `run_inhibit_temporal_mse.py` | end-to-end CLI driver |

## Variants — 217M and 1B

The Megatron architecture (layers, hidden, heads, RoPE) is read out of the
distcp via `load_settings_from_checkpoint`, so the same config wrapper works
for both variants — only `--ckpt-dir` differs. Defaults in `predict_runner.py`:

| variant | seq_length | micro_batch_size |
|---|---|---|
| 217m | 16384 | 4 |
| 1b   | 16384 | 1 (bump if you have headroom) |

`scale_factor` defaults to 8.0 (matches the upstream `MaxTokiConfig`). For
1B (Llama3-style scaled RoPE) this is what was used in training; for 217M
(standard RoPE) it's a no-op.

## Run

```bash
# inside the BioNeMo container (bionemo + nemo + megatron available)
python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
    --h5ad ./data/4111_cells.h5ad \
    --ckpt-dir /weights/maxtoki-1b-bionemo \
    --tokenizer-path /weights/maxtoki-1b-bionemo/context/token_dictionary.json \
    --variant 1b \
    --gene-symbol ZLX1 \
    --direction inhibit \
    --seq-length 16384 \
    --out-dir ./out/zlx1
```

Outputs under `--out-dir`:
- `baseline.dataset/`, `perturbed.dataset/` — HF datasets passed to BioNeMo
- `baseline_predictions/`, `perturbed_predictions/` — `predictions__rank_*.pt`
- `scores.npz` — `baseline`, `perturbed`, `delta_t`, `per_cell_mse`, `gene_present`
- `summary.json` — mean MSE / mean Δt across the dataset

`--prep-only` stops after dataset build (handy for inspecting tokenization
on a CPU box). `--score-only` skips prep + predict and re-scores existing
`*_predictions/` dirs.

## h5ad expectations

- raw counts in `adata.X` (or `adata.raw.X` if available)
- Ensembl IDs: `var["ensembl_id"]` / `var["feature_id"]` / ENSG-prefixed `var_names`
- optional `obs["n_counts"]` (else computed as `X.sum()` per row)

## Token dictionary

The `maxtoki_mlx` package ships a stripped dict (gene + special tokens, no
`<boq>` / `<eoq>` / numeric tokens) — fine for backbone-only work, **not**
for the temporal head. Always pass `--tokenizer-path` pointing at the full
`token_dictionary_v1.json` shipped with the BioNeMo distcp (or upstream
`maxToki/resources/token_dictionary_v1.json`).
