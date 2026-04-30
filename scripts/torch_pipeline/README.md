# torch_pipeline — zero-shot gene-inhibition scoring on the BioNeMo MaxToki temporal head

This is the CUDA / x86 sibling of `maxtoki_mlx`. It runs on A100 / H200 boxes
(e.g. NCSA Delta) where the full BioNeMo distcp checkpoint — the **only**
checkpoint that ships the temporal head — can be loaded directly. The MLX
package leaves the temporal head out (its load path is blocked on distcp →
MLX conversion); here it stays in.

## Layout

| file | purpose |
|---|---|
| `tokenizer.py` | rank-value tokenizer, no MLX deps (reads JSONs from `src/maxtoki_mlx/resources/`) |
| `perturbation.py` | `delete` / `overexpress` / `inhibit` — bit-identical to the MLX version |
| `data.py` | h5ad → tokenized cells (the 4111-cell dataset) |
| `model.py` | `MaxTokiTemporal` adapter: backbone + temporal head |
| `scoring.py` | per-cell baseline vs. perturbed temporal-MSE scoring |
| `run_inhibit_temporal_mse.py` | CLI driver |

## Pipeline

```
h5ad ─► tokenizer ─► token_ids[c]
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        baseline forward       perturbed forward (inhibit g)
              │                       │
        backbone hidden          backbone hidden
              │                       │
       temporal_head           temporal_head
              │                       │
              └──────► Δt(c,g) ◄──────┘
                          │
                  MSE = mean(Δt²)
```

The signed `Δt` (perturbed − baseline) is also kept so direction is recoverable.

## What's deterministic right now

Everything except the weight loader. `tokenizer`, `perturbation`, `data`,
`scoring`, and the CLI are complete and runnable. The `MaxTokiTemporal`
adapter has two load paths:

* `from_hf_with_temporal(hf_repo, head_state_dict_path, variant)` — works
  today, **if** you've extracted the temporal-head weights to a `.pt` file
  from the distcp checkpoint.

* `from_distcp(distcp_dir, variant)` — **stub**, raises `NotImplementedError`.
  Wiring this up needs the upstream `NVIDIA-Digital-Bio/maxToki` repo so I can
  match the exact `MegatronBaseModel` subclass, recipe yaml, and temporal-head
  module definition. Drop it next to this checkout (or share the path) and
  I'll fill in the `TODO` block in `model.py:from_distcp`.

The placeholder `TemporalHead` (Linear → GELU → Linear, EOS pooling) is a
sensible default but **may not match the upstream architecture**. Don't trust
the absolute MSE numbers until that's verified against the published code.

## Run

```bash
# 1) install (CUDA / x86)
pip install torch transformers anndata scipy numpy

# 2) point env vars at your weights (or pass --distcp / --hf flags)
python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
    --h5ad ./data/4111_cells.h5ad \
    --hf theodoris-lab/MaxToki-1B-HF \
    --temporal-head ./weights/temporal_head_1b.pt \
    --variant 1b \
    --gene-symbol ZLX1 \
    --direction inhibit \
    --batch-size 16 \
    --out ./out/zlx1_inhibit.npz
```

Output `.npz` contains `baseline`, `perturbed`, `delta`, `per_cell_mse`,
`gene_present`, `cell_ids`, plus a JSON `summary` with dataset-level mean MSE.
