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

## 3. The prompt → generation flow for one query

For each query cell `q`, `dataset_prep.build_paired_dataset` emits two rows (baseline
and perturbed) sharing a `row_index` so the paired comparison is unambiguous. The
NextCell row grammar in both:

```
[<bos>, ctx_1_genes, <eos>,
 <bos>, ctx_2_genes, <eos>,
 <bos>, ctx_3_genes, <eos>,
 <boq>, q_genes,     <eoq>,
 <bos>]                      <- sentinel for the multitask collator
```

The three context cells and the query carry the model's "trajectory sense" — young
donor cells at pseudotime t_1 < t_2 < t_3, then an old-donor query. The trailing
`<bos>` is a protocol detail: BioNeMo's `collate_batch_multitask` indexes
`token_ids[eoq_index + 1]` to classify the row, so the prompt MUST have something
after `<eoq>`. We put the training-grammar continuation (`<bos>`) there and tell the
generator to truncate it off before the autoregressive loop (`using_pretrain_dataset=True`
in `bionemo.maxtoki.predict.predict`).

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
rve_cap_next_cell = (seq_length - max_tokens_to_generate - 1) // (K + 1)
```

For the pinned run (`seq_length=16384`, `max_tokens_to_generate=2048`, `K=3`) that's
**3583 tokens/cell**. Compare with the TBC cap of 4095 — NextCell runs with slightly
truncated context cells to leave room for 2048 generated tokens.

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

Both runs use the same seed, same context, same decoding, same `row_index`. The only
difference is one gene's token position inside the query block:

```
row i baseline:  [... <boq>  q_i_genes_sorted_by_expression  <eoq> ...]
                                       ↑
                                       |
                                  PDK4 at some rank in q_i
                                       |
                                       ▼ apply inhibit:
row i perturbed: [... <boq>  <perturb(q_i)>                  <eoq> ...]
                                       ↑
                                       |
                                  PDK4 moved to the lowest rank
```

`perturbation.py` offers three edits:
- `inhibit`: move the gene token to the LAST position (lowest rank).
- `delete`: remove the gene token entirely.
- `overexpress`: move the gene token to the FIRST position (highest rank).

Any difference between the baseline-generated list `A_i` and the perturbed-generated
list `B_i` is the model's downstream response to that one-gene rank edit.

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
