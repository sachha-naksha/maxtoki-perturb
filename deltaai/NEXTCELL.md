# NextCell: what it is and how it generates states over all query cells

Companion reference for `AGENT_NEXTCELL.md` / `PROGRESS_NEXTCELL.md`. Focused on the
conceptual flow and the exact code hooks used by this pipeline. Written against the
BioNeMo-maxtoki code vendored in `deltaai/src/maxToki/sub-packages/bionemo-maxtoki/` and
the glue in `scripts/torch_pipeline/`.

---

## 1. NextCell as a task head

MaxToki is a decoder-only transformer trained with a rank-value cell tokenizer and two
task heads:

| task | prompt grammar | what the model emits |
|---|---|---|
| **TimeBetweenCells** | `[..., <boq>, q_genes, <eoq>, numeric]` | one scalar `delta_t` (regression at `<eoq>`) |
| **NextCell**         | `[..., <boq>, q_genes, <eoq>]`           | an autoregressive token sequence `<bos>, g_1, ..., g_K, <eos>` |

BioNeMo's `MaxTokiTokenizer.determine_task_type` classifies rows by the token
immediately after `<eoq>` (numeric → TBC, `<bos>` → NextCell). Our zero-shot pipeline
historically only emitted TBC rows (`dataset_prep` appended `dummy_numeric` after
`<eoq>`); the NextCell path was built upstream but dormant until this change.

The pipeline flips a single bit — `spec.task_type` — to switch the dataset grammar and
the predict step. Everything else (context/query selection, perturbation, paired
baseline/perturbed row order) is unchanged.

## 2. Why generate the next cell at all

TBC collapses the model's response to one scalar per query: "the perturbation moved
this cell by Δt units along pseudotime." Useful but low-bandwidth.

NextCell gives the full predicted expression ranking directly. Because the
rank-value tokenizer sorts non-zero genes in **descending** expression order, the
model's generated gene list IS the predicted ranking — the first emitted gene is the
predicted highest-expressed, the second is the next-highest, and so on. You can
compare baseline-vs-perturbed as two ordered lists: which genes gained rank, which
lost rank, which appeared, which disappeared.

## 3. The prompt → generation flow for one target

For each target future cell `t` (e.g. an OM 80y cell), `dataset_prep.build_paired_dataset`
emits two rows (baseline and perturbed) sharing a `row_index`. The NextCell row grammar
follows NVIDIA's `data_prep/dataset_utils.py::compose_example` exactly — **cells are
interleaved with Δt tokens, and the query slot is a Δt scalar, not a cell**:

```
[<bos>, ctx_1_genes, <eos>, Δt_12,
 <bos>, ctx_2_genes, <eos>, Δt_23,
 <bos>, ctx_3_genes, <eos>,
 <boq>, Δt_q, <eoq>,
 <bos>]                             <- sentinel for the multitask collator
```

The three context cells and their inter-cell Δt tokens carry the model's "trajectory
sense" (young-donor cells at pseudotime t_1 < t_2 < t_3). The query block
`<boq> Δt_q <eoq>` tells the model: "project from the last context cell forward by
Δt_q time units; emit the next cell's rank-value expression." The OM target cell
itself is NOT fed into the prompt — only its pseudotime, used to compute
`Δt_q = round(ptime(target) - ptime(ctx_K))`.

The trailing `<bos>` is a protocol sentinel: BioNeMo's `collate_batch_multitask`
indexes `token_ids[eoq_index + 1]` to classify the row (numeric → TBC, `<bos>` →
NextCell). We satisfy that requirement with a sentinel and tell the generator to
truncate it off before the autoregressive loop (`using_pretrain_dataset=True` in
`bionemo.maxtoki.predict.predict`).

**Perturbation site under NextCell.** The query (Δt_q) is a scalar — nothing to edit.
We instead perturb the **last context cell** `ctx_K` when `apply_to=query`, or **every
context cell** when `apply_to=query_and_context`. Baseline vs. perturbed differ ONLY
in whether `ctx_K` carries the gene edit; Δt_12 / Δt_23 / Δt_q are byte-identical
between the two rows.

**Δt token encoding.** The BioNeMo full dictionary includes 3000 numeric tokens
covering integers in `[-1500, 1499]` (sample from the pinned 217M dict). We discretize
pseudotime differences with `round(…)` and clamp to that range before lookup via
`dataset_prep._dt_token`. For aging-SKM pseudotime in `[0, 10]` the clamp is a no-op.

The generator (`maxtoki_generate_predict_step` in `generate_utils.py`) then runs:

```
truncate prompt to [..., <eoq>]        (drop the <bos> sentinel)

inference_context = DynamicInferenceContext(
    max_sequence_length = prompt_len + max_tokens_to_generate,
    ...
)

while not finished:
    input_ids, pos_ids = inference_context.current_input_and_position_ids()
    out = model.forward(input_ids, pos_ids, inference_context=inference_context,
                        next_token_only=True)
    logits = out["lm_outputs"][-1]
    # mask: forbid every special token except <bos>/<eos>; forbid all numeric;
    #       forbid any token already generated in this request
    logits[sampling_mask] = -inf
    next_tok = sample(logits, top_k=1, top_p=0.0, temperature=1.0)   # greedy
    if next_tok == <eos>: finished = True
```

Pinned decoding in the first run is greedy (`top_k=1`), which makes the whole process
deterministic: two reruns on identical prompts produce byte-identical output. That
determinism is what makes paired baseline-vs-perturbed differences causally
attributable to the gene-rank edit, not to sampling noise.

The KV cache is sized up front by `DynamicInferenceContext`:

```
max_sequence_length = prompt_len + max_tokens_to_generate
```

so Megatron never reallocates mid-generation. This is also why we reserve the generation
budget in `_per_cell_max_len`:

```python
rve_cap_next_cell = (seq_length - max_tokens_to_generate - K - 3 - 1) // K
```

(K cells + K−1 inter-Δt + 3 query-block + 1 sentinel + 1 off-by-one safety.) For the
pinned run (`seq_length=16384`, `max_tokens_to_generate=4096`, `K=3`) that's **4093
tokens/cell** — nearly identical to the TBC cap of 4095 because the NextCell query
block is only 3 tokens (vs. a full cell under TBC).

## 4. How it scales across all query cells

We don't generate one global next-cell state — we generate N **conditionally
independent** next-cell states, one per query row. For the pinned 2000-OM-query run:

```
query 1  (OM6 cell 0547) ─┐                              ┌─ next_cell_pred_1
query 2  (OM6 cell 0548) ─┤  shared 3 YM2 context cells  ├─ next_cell_pred_2
query 3  (OM6 cell 0549) ─┤  (same prompt prefix for all)├─ next_cell_pred_3
   ...                   │                               │     ...
query 2000 (OM9 cell …)  ─┘                              └─ next_cell_pred_2000
```

The three YM2 context cells (picked `evenly_spaced` across the 34y pseudotime
trajectory in `_resolve_pool`) are **the same for every row** — the context block of
the prompt is byte-identical across all 2000 queries. Only the query block changes.
This is important for two reasons:

1. **Shared trajectory signal.** Every query sees the same "young-donor arc," so any
   difference in the generated next cell can't come from differently-selected context
   cells per row. Fair paired comparison.
2. **KV cache amortization opportunity.** In principle you could compute the context
   block's KV once and reuse it. In practice BioNeMo's `DynamicInferenceContext`
   doesn't expose a cache-reuse hook across requests, so each row recomputes it. This
   is why NextCell is ~1000× slower than TBC per cell — the context block costs the
   same forward pass whether you reuse it or not, but there's an additional 2048 steps
   of autoregression per row with growing KV.

Within a run, each generation is independent of the others — DynamicInferenceContext
packs multiple requests into one batch and advances them in lockstep per generation
step, finishing a request as soon as it emits `<eos>`. Megatron handles the
cross-request packing; the user-level view is "2000 independent next-cell
generations with a shared prompt prefix."

## 5. Baseline vs. perturbed: the paired structure

Both runs use the same seed, same context cells, same decoding, same `row_index`, and
the SAME `Δt_12`, `Δt_23`, `Δt_q`. Under NextCell the perturbation lives in `ctx_K`,
not the query block (the query block is a Δt scalar — nothing to edit):

```
row i baseline:   [... <bos> ctx_K_genes <eos>  <boq> Δt_q <eoq>  <bos>]
                            ^^^^^^^^^^^^^
                            PDK4 at some rank
                                     |
                                     ▼ apply inhibit:
row i perturbed:  [... <bos> perturb(ctx_K_genes) <eos>  <boq> Δt_q <eoq>  <bos>]
                            ^^^^^^^^^^^^^^^^^^^^^
                            PDK4 moved to lowest rank
```

`perturbation.py` offers three edits:
- `inhibit`: move the gene token to the LAST position (lowest rank).
- `delete`: remove the gene token entirely.
- `overexpress`: move the gene token to the FIRST position (highest rank).

Any difference between the baseline-generated list `A_i` and the perturbed-generated
list `B_i` is the model's downstream response to the gene-rank edit in `ctx_K`.

The paired row-order invariant — `baseline[i].row_index == perturbed[i].row_index ==
i` — is enforced by `build_paired_dataset` and double-checked by the scorer against
`row_manifest.json` (any mismatch is a hard failure, not silently repaired).

## 6. From generated tokens back to interpretable metrics

`score_nextcell.py` reads both prediction trees, decodes each `generated_tokens`
sequence back to an ENSG rank list via the full BioNeMo token dictionary, and emits
per-row:

| metric | what it tells you | formula |
|---|---|---|
| **Jaccard@50** / @100 / @500 | how many of the top-k genes survived the edit | &#124;A_i[:k] ∩ B_i[:k]&#124; / &#124;A_i[:k] ∪ B_i[:k]&#124; |
| **Spearman ρ on shared** | how much the ranking got reshuffled | Pearson of ranks on the genes present in both lists |
| **Per-gene rank shift** | which specific genes moved | pos(gene, B_i) − pos(gene, A_i), NaN for one-sided presence |

The decoder (`decode_generation`) enforces grammar: strips the leading `<bos>`, stops
at `<eos>`, drops duplicates (the model sometimes repeats a gene), skips numeric /
`<boq>` / `<eoq>` / pad tokens (none should appear during NextCell generation — if
they do, it's a sampling-mask bug), and counts invalid tokens for reporting.

The summary splits each metric by:
- target-gene **present** in the prompt vs. **absent** (PDK4 isn't expressed in every
  cell; the perturbation can only affect generations where the gene was in the query)
- both generations **finished naturally** (`<eos>` emitted) vs. at least one **capped**
  at `max_tokens` (capped generations only partially reveal the model's ranking →
  noisier metrics)

Raw decoded ENSG lists are persisted to `decoded_nextcell.json` so downstream notebooks
can re-score however they want (e.g. restrict to transcription factors, swap in a
different k for Jaccard@k, compute set-overlap enrichment).

## 7. Where to find each piece in the code

| step | code |
|---|---|
| spec `task_type` + `generation` block | `scripts/torch_pipeline/spec.py::ExperimentSpec`, `GenerationSpec` |
| context / query selection (unchanged by NextCell) | `scripts/torch_pipeline/dataset_prep.py::_select_context`, `_select_queries` |
| row grammar branch | `scripts/torch_pipeline/dataset_prep.py::_build_input_ids` |
| per-cell cap reserving generation budget | `scripts/torch_pipeline/dataset_prep.py::_per_cell_max_len` |
| paired row + manifest | `scripts/torch_pipeline/dataset_prep.py::build_paired_dataset` (writes `row_manifest.json`) |
| generation kwargs plumbed through | `scripts/torch_pipeline/predict_runner.py::run_headless_predict` |
| upstream task routing | `deltaai/src/maxToki/.../bionemo/maxtoki/predict.py::predict` (branch on `generate_next_cell`) |
| autoregressive loop | `deltaai/src/maxToki/.../bionemo/maxtoki/generate_utils.py::maxtoki_generate_predict_step` |
| sampling mask (disables all specials except `<bos>`/`<eos>`; disables numeric) | `deltaai/src/maxToki/.../bionemo/maxtoki/generate_utils.py::setup_default_sampling_mask` |
| decoder + metrics | `scripts/torch_pipeline/score_nextcell.py` |
| CLI dispatch | `scripts/torch_pipeline/run_inhibit_temporal_mse.py::main` (branch on `spec.task_type`) |
| pinned configs | `scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_{smoke,full}.yaml` |
| sbatch wrappers | `deltaai/slurm/_run_nextcell_pdk4_{smoke,full}.sh`, `_run_nextcell_tests.sh` |

## 8. Known gotchas

- **The trailing `<bos>` is a protocol requirement, not a semantic token.** Our dataset
  rows end with `[..., <eoq>, <bos>]` purely to satisfy
  `collate_batch_multitask → determine_task_type`. The generator truncates it off
  before the autoregressive loop.
- **Upstream `determine_task_type` uses `is` for the `<bos>` check** (`token_ids[eoq_index + 1]
  is self.special_tokens["<bos>"]`). CPython caches ints only in `[-5, 256]`; `<bos>`
  id is well outside, so `is` fails and the classifier raises "Invalid grammar."
  `predict_runner._patch_determine_task_type_bos_check()` monkey-patches the comparison
  to `==` at runtime.
- **`using_pretrain_dataset=True` is forced** by `predict_runner` whenever
  `generate_next_cell=True` so the generator truncates each row at `eoq_index + 1`
  (dropping the `<bos>` sentinel) before starting the autoregressive loop.
- **Runtime is NOT comparable to TBC.** Δt is one forward pass per row (~ms).
  NextCell is up to `max_tokens_to_generate` forward passes with a growing KV cache
  — roughly 2048× the per-row cost at the pinned cap, offset partially by batch
  packing in DynamicInferenceContext. Size the full-run wall limit from the smoke
  measurement, not from the Δt timings in `deltaai/HANDOFF.md`.
- **Greedy ≠ no stochasticity in multi-run comparisons.** With `top_k=1` the generator
  itself is deterministic, but kernel-level numerics (reduction order, bf16 rounding)
  can shift the argmax on near-ties. If two reruns of the smoke disagree on any row,
  investigate — don't chase the discrepancy via sampling noise, it isn't.
- **`buffer_overflow_factor` must be ~50, not 1.** Upstream's CLI default is 50.0.
  The dormant brief suggested 1.0, which causes Megatron's `DynamicInferenceContext`
  to refuse the 2nd request in a batch with `TokenOverflowError`, even when the
  nominal buffer (`buffer_size_gb=20`) has orders of magnitude of room. The comment
  in `bionemo.maxtoki.predict` says: "smaller values leave larger unused space in
  the KV cache. 50.0 uses the max tokens allocated for the token limit check." Pin
  50.0 as the default in `GenerationSpec`.
- **Prompt grammar is NOT optional.** An earlier smoke (job 3343372) built rows as
  `[c1, c2, c3, <boq>, query_cell_genes, <eoq>, <bos>]` — all cells adjacent, query
  is a cell, no inter-cell `Δt`. Mechanics ran (grammar classifier passed), but the
  model never emitted `<eos>` on any of 40 generations, baseline and perturbed were
  near-identical, and top-ranked outputs were pseudogene-dominated. Root cause: the
  pretraining data pipeline (`compose_example`) always interleaves cells with `Δt`
  tokens and uses a `Δt` as the question slot in `predict_time_or_cell="cell"` mode.
  Feeding the model something it has never seen during training produces
  low-entropy default-mode garbage. Match the NVIDIA grammar exactly.

## 9. What NextCell gives you vs. TimeBetweenCells, and what the smoke does / doesn't tell you

Written 2026-10-08 against the TBC outputs in `out/<gene>_217m_<dir>_evenly_seq8k/` and
the NextCell smoke outputs in `out/nextcell_pdk4_inhibit_smoke_*`.

### 9.1 The question each task answers

Both tasks use the same prompt prefix — 3 young (YM2, 34y) context cells picked
evenly across pseudotime — and the same 2000 old (OM6+OM9, 80y) target cells. They
differ in what the model is asked at the end, and therefore in what a perturbation
can tell you.

| | TimeBetweenCells (what you have in `out/*_seq8k`) | NextCell |
|---|---|---|
| **Question posed** | "Here is the old target cell's expression. How far in time is it from the young trajectory?" | "Starting from the young trajectory, what does a cell look like `Δt_q` time units later?" |
| **Target cell's role** | fed into the prompt as the query (`<boq> target_genes <eoq>`) | **not fed in** — only its pseudotime is used, to set `Δt_q` |
| **Where the perturbation lands** | on the target (query) cell | on `ctx_3`, the last young context cell (the query is a scalar) |
| **Model output per row** | one number, `Δt` | an ordered gene list `g_1 … g_K` (= predicted rank-value expression of the future cell) |
| **Perturbation effect per row** | `Δt_perturbed − Δt_baseline`: did silencing the gene make the old cell read as younger / older | two ranked gene lists to compare: which genes moved, appeared, disappeared |
| **Summary you get** | `summary.json`: `mean_delta_t`, `mean_mse`, `n_rows_with_gene_in_query`, `*_present` variants; `scores.npz`; `viz/` | `summary_nextcell.json`: `mean_jaccard_at_{50,100,500}`, `mean_spearman_shared`, `n_*_finished`; `scores_nextcell.npz`; `decoded_nextcell.json` |

So TBC collapses the model's reaction to a perturbation onto a single axis (a shift
along pseudotime, in the units of the `Pseudotime` column). It tells you *how much*
the model thinks the gene edit moves a cell along aging, averaged over 2000 real old
cells — that is what `mean_delta_t_8k_evenly*.csv` and the `by_donor_*` figures show.

NextCell gives you the model's **counterfactual transcriptome**: the predicted
rank-ordered gene list of a cell `Δt_q` ahead of the young trajectory, once with
PDK4 at its native rank in `ctx_3` and once with PDK4 pushed to the bottom. Comparing
the two lists tells you *what changes*, not just how much — which genes rise or fall
in predicted rank when PDK4 is silenced upstream. That is the only one of the two
tasks whose output you can hand to gene-set / pathway analysis, or compare gene-by-
gene against the real old target cell's ranking (not implemented yet; the decoded
lists in `decoded_nextcell.json` are kept for exactly this kind of re-scoring).

### 9.2 What is in the NextCell output directory

| file | contents |
|---|---|
| `row_manifest.json` | one record per query: `row_index`, `cell_id`, `group` (donor), `query_pseudotime`, `context_cell_ids` (same 3 for every row), `gene_present_in_query` |
| `baseline_predictions/`, `perturbed_predictions/` | raw `generated_tokens`, `lengths`, `finished_naturally` per row (`predictions__rank_0.pt`) |
| `decoded_nextcell.json` | per row: `baseline.ensg_order`, `perturbed.ensg_order` — the two predicted rankings as ENSG lists, deduped, `<bos>`/`<eos>` stripped |
| `scores_nextcell.npz` | per-row arrays: `jaccard_at_50/100/500`, `spearman`, `spearman_overlap`, `baseline_finished`, `perturbed_finished`, `both_finished`, `gene_present`, `baseline_length`, `perturbed_length`, `*_n_invalid` |
| `summary_nextcell.json` | means of the above, split by `gene_present` and by finished-vs-capped; plus `sample[]` with the first rows' paired top-20 lists |
| `summary.json` | the same summary with the spec metadata appended (gene, direction, variant, generation settings) |

How to read the per-row metrics:
- **Jaccard@k = 1.0** → the top-k predicted genes are the same set with and without
  the edit (order may differ). **Lower** → the edit changed *which* genes the model
  puts at the top. @50 is the "marker gene" view; @500 the broad-program view.
- **Spearman on shared** → how much the *order* of genes present in both lists was
  reshuffled (1.0 = same ranking). `spearman_overlap` is the number of shared genes
  it was computed on (≈1800 in the smokes).
- **`*_finished`** → whether the model emitted `<eos>` before `max_tokens`. If false,
  the list is a truncated prefix of the model's ranking and k-dependent metrics are
  only trustworthy for k ≪ length.
- **`gene_present`** → whether PDK4 was actually in the cell that got edited (see 9.4).

### 9.3 What the smoke test is for

The smoke is the full-run pipeline on a deterministic 10- or 20-query subset
(`query.limit_n`, `query.seed`) with the full-run prompt settings. It is a *gate*, not
an experiment. It establishes:

1. **Mechanics end to end** — the NVIDIA grammar rows pass the collator, the generator
   emits gene tokens (no numeric/special leakage: `*_n_invalid == 0`), the decoder
   produces ENSG lists, and the baseline/perturbed pairing survives (`row_manifest`
   cross-check).
2. **Cost for sizing the real run** — tokens/sec and peak GPU memory. On Delta
   (job 22764367/22764696, H200): ~29–31 tok/s, 20.5 GB peak, **~2.1 min per row**
   because every row runs to the 4096-token cap; 2 rows per query → ~4.2 min/query.
   That makes the 2000-query spec ≈ 140 GPU-hours on one H200 — it needs sharding
   across GPUs or a lower `max_tokens` before it is launchable.
3. **Determinism** — greedy decoding means a rerun of the same spec must reproduce
   every row byte-for-byte; any drift is a numerics problem, not noise (§8).
4. **A first look at the effect** — are the perturbed lists different from the
   baseline at all, and do they differ only where the edit could matter.

What the smoke does **not** give you: statistics. 10–20 cells cannot stratify by donor
or pseudotime, cannot say which gene rank shifts are reproducible across cells, and a
mean Jaccard over 10 rows has no error bar worth quoting. That is what the full
(or an intermediate 200–500 query) run is for.

### 9.4 Two things the current smokes already show

- **No generation has ever finished naturally.** Every row in every smoke so far
  (ARM 2048-cap runs and Delta 4096-cap runs) hits the cap: `n_both_finished = 0`.
  The ARM smoke gate in `PROGRESS_NEXTCELL.md` asked for ≥18/20 natural completions,
  so by that criterion the gate has not been passed. A real cell has ~2000 non-zero
  genes, so lists of 4096 are the model ranking genes well past where expression
  would be zero. Metrics at k ≤ 500 are still meaningful (they only look at the head
  of the list), but "predicted cell length" is not a usable readout, and half of every
  generation's cost buys nothing. Whether `<eos>` is reachable at all with greedy
  decoding, or only with sampling, is open.
- **The present/absent split is degenerate under the NVIDIA grammar.** Because the
  perturbation sits in `ctx_3` and `ctx_3` is the same cell for all rows,
  `gene_present` is identical for every row — in the Delta smoke prep summary it is
  20/20 present. The built-in negative control that TBC had ("rows where the gene is
  absent must show zero effect") no longer exists; a NextCell control has to be
  constructed explicitly (e.g. a `delete`/`inhibit` of a gene absent from `ctx_3`, or
  a different context pool).
- **The prompt depends only on `Δt_q`, so the 2000-query run has only 68 distinct rows.**
  The target cell is not in the prompt; with a shared context pool the whole row is
  determined by `round(ptime(target)) − ptime(ctx_3)`. The Delta smoke confirmed it:
  rows 0 and 6 (CELL2335 / OM6 and CELL4348 / OM9, both pseudotime 85.0) produced
  byte-identical generations and scores. Over all 2000 OM cells, `round(ptime − 100)`
  takes 68 values (−67…0; OM6 covers 68, OM9 67). Consequences: (a) the "full run"
  is really 68 unique queries × 2 rows ≈ 136 generations ≈ 5 GPU-h at the 4096 cap,
  not 4000 generations; the per-cell version just duplicates them. (b) Per-cell or
  per-donor statistics over 2000 rows would be counting exact copies — the effective
  sample size is the number of distinct Δt_q, and donors differ only in which Δt_q
  values they populate. (c) If per-cell variation is wanted, the target cell has to
  enter the prompt (e.g. `apply_to`/context built from the target's own trajectory),
  which is a different experiment from the current spec.
- **Caution on the numbers already in `out/`.** `out/nextcell_pdk4_inhibit_smoke_3343372/summary_nextcell.json`
  (Jaccard@50 0.90 overall / 0.79 present) comes from the **old, wrong prompt grammar**
  (§8, last bullet) and should not be quoted as a NextCell result. The first completed
  correct-grammar result is the Delta 10-query smoke (`out/nextcell_pdk4_inhibit_smoke_delta_22764696/`).
