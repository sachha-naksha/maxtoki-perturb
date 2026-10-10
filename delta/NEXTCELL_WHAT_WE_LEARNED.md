# NextCell: what the current experiment does, what you want, and what the smoke shows

Written 2026-10-08 after the first completed correct-grammar NextCell smoke on Delta
(job 22764696, H200, 10 queries). Companion to `deltaai/NEXTCELL.md` (mechanics) and
`delta/PROGRESS_NEXTCELL.md` (jobs, fixes, commands).

---

## 1. What the current NextCell experiment actually does

The spec (`scripts/torch_pipeline/configs/pdk4_inhibit_nextcell_*.yaml`) copies the
TimeBetweenCells design and then follows NVIDIA's NextCell prompt grammar. The grammar
changes the meaning of the experiment:

```
prompt = [ young cell 1 ] Δt [ young cell 2 ] Δt [ young cell 3  <- PDK4 edited HERE ]  <boq> Δt_q <eoq>
model  → generates "the cell Δt_q time units after young cell 3"
```

- The 3 context cells are **young** (YM2, 34y, pseudotimes 1 / 57 / 100). They are the
  same 3 cells for every row (`context.strategy: pool`, `pool_filter: age [34]`).
- The perturbation (PDK4 moved to lowest rank) is applied to **young cell 3**, not to any
  old cell (`perturbation.apply_to: query` means "last context cell" under NextCell).
- The **old cell is never in the prompt**. It contributes one number:
  `Δt_q = round(pseudotime(old cell)) − 100`. In NVIDIA's grammar the question slot is a
  *time*, not a cell.

So the question asked is: *"Given a young trajectory in which PDK4 was silenced in the
last cell, what does the cell look like Δt_q later?"* It is a legitimate experiment, but
it is **not** "inhibit PDK4 in old cells and see whether they move toward a younger
state."

Consequence: because the old cell enters only as a rounded number, the 2000 OM cells
collapse to **68 distinct prompts** (Δt_q = −67 … 0). The smoke proved it — rows 0 and 6
(an OM6 cell and an OM9 cell, both pseudotime 85) produced byte-identical generations.
Running 2000 cells repeats 68 generations ~30 times each.

## 2. The experiment you want, and how NextCell would be set up for it

Hypothesis: *old cell + PDK4 inhibited → its next state looks younger than the same old
cell un-edited.*

```
baseline :  [ old cell i ]                    <boq> Δt_q <eoq>   → generated next cell A_i
perturbed:  [ old cell i with PDK4 → bottom ]  <boq> Δt_q <eoq>   → generated next cell B_i
score    :  is B_i "younger" than A_i ?
```

Already supported by the pipeline (`scripts/torch_pipeline/spec.py`):
- `context.strategy: self` → the context is the query (old) cell itself, 1 cell.
- or `context.strategy: prefix`, `include_self: true`, `data.group_col: sample` → the old
  cell plus earlier cells from the same donor, i.e. a real old-donor trajectory.
- `perturbation.apply_to: query` → edits the last context cell, which in both cases *is*
  the old cell.
- `generation.*` unchanged (greedy, `max_tokens`, buffer).

Not yet implemented (both small):
1. **A Δt_q knob.** Today `Δt_q = pseudotime(target) − pseudotime(last context cell)`,
   which is 0 when the context is the cell itself. Needs e.g. `query.delta_t: 5` in the
   spec, plumbed through `dataset_prep.build_paired_dataset`.
2. **A "younger" scorer.** NextCell returns a ranked gene list, not an age. "Younger" must
   be defined — e.g. Spearman of the generated ranking against the mean YM2 ranking vs
   the mean OM ranking, or feed each generated cell back through TimeBetweenCells to get
   a Δt. Nothing in `score_nextcell.py` does this yet.

## 3. How to see what NextCell generates — and the problem it shows

Generated cells are in
`out/nextcell_pdk4_inhibit_smoke_delta_22764696/decoded_nextcell.json`:
`rows[i].baseline.ensg_order` and `rows[i].perturbed.ensg_order`, highest predicted
expression first. Symbols via `src/maxtoki_mlx/resources/gene_name_id.json` (symbol → ENSG;
invert it).

Row 0 (OM6 cell CELL2335, pseudotime 85 → Δt_q = −15), mapped to symbols:

- **baseline top 25:** S100A12, FAM90A20P, ZNF534, LYZL1, TSPAN2, WNT10B, DHX57, ANKRD44,
  KRTAP10-11, SLC33A1, ADORA2B, NBPF3, ZC3HC1, LACTB2, ZNF565, NBPF26, VSIG4, FAM200B,
  GDF6, TMED4, WDR1, RAD51D, CSRNP3, CISD2, FLRT2
- **perturbed:** identical for the first 19, then RIOX1, CACNB3, OR4C11, PARPBP, … diverging
- PDK4 itself: rank 976 in the baseline list, absent from the perturbed list (the edit
  does propagate).
- Scores over the 10 rows: Jaccard@50 0.77, @100 0.68, @500 0.57; Spearman on shared
  genes 0.75. Every generation ran to the 4096-token cap; none emitted `<eos>`.

**This is not a skeletal-muscle cell.** A real one has ACTA1 / MYH* / TTN / mitochondrial
genes at the top; this list is led by a pseudogene (FAM90A20P), a keratin-associated
protein (KRTAP10-11), an olfactory receptor (OR4C11), and zinc fingers. The zero-shot
217M model, given this prompt, produces a low-plausibility default ranking, not a
muscle transcriptome. Any "younger vs older" measurement on these lists would be
measuring noise.

### 3.1 Sanity gate result (run 2026-10-08, `delta/scripts/sanity_nextcell.py`)

Top-k set overlap between generated rankings and real rank-value tokenizations
(real cells re-tokenized with the checkpoint's token dictionary; context cells recovered
from `baseline.dataset` exactly as the model saw them):

| comparison | top-50 | top-100 | top-500 |
|---|---|---|---|
| real OM vs real OM (30 random cells, all pairs) | 0.25 | 0.21 | 0.20 |
| real OM vs real YM2 | 0.22 | 0.19 | 0.18 |
| ctx_3 vs ctx_1 (the two young context cells) | 0.22 | 0.21 | 0.17 |
| **generated vs ctx_3** (the cell it was conditioned on) | **0.02** | **0.01** | **0.03** |
| **generated vs its real OM target cell** | **0.00** | **0.01** | **0.02** |
| generated vs 30 random real OM / YM2 cells | — | 0.00 / 0.00 | — |

Identical for baseline and perturbed, all 10 rows. 46 of the generated top-50 genes do
not appear anywhere in ctx_3's 1065 expressed genes; the 4 that do sit at median rank
~560 there. Only 27% of the generated top-500 appear in the top-500 of *any* of 30 real
OM cells.

For orientation, the real cells are unmistakably muscle: ctx_3 top-15 = NEB, TTN,
FILIP1L, MYBPC1, GBE1, ABLIM1, SVIL, …; a real OM cell = TTN, NEB, ASB5, MYBPC1, …
The generated row-0 top-15 = S100A12, FAM90A20P, ZNF534, LYZL1, TSPAN2, WNT10B, …

**Verdict: the zero-shot 217M NextCell generation does not resemble a cell at all** —
it is ~10× below the overlap two unrelated real cells have with each other, and shares
essentially nothing with the cell it was conditioned on. The Jaccard/Spearman
differences between baseline and perturbed (§3) are differences between two
non-cells. No PDK4 conclusion can be drawn from this output, and the experiment
redesign in §2 is moot until generation itself works.

Reproduce (CPU, inside the container, from the repo root, with
`MAXTOKI_TOKEN_DICT=/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json`):
`python3 delta/scripts/sanity_nextcell.py out/nextcell_pdk4_inhibit_smoke_delta_22764696`

### 3.2 Was the checkpoint trained on NextCell? — No. It is a Stage-1 model.

Evidence (checked 2026-10-08):

| source | what it says |
|---|---|
| `models/MaxToki/README.md` (HF model card) | "MaxToki-217M and MaxToki-1B … **pretrained on 175 million human single-cell transcriptomes**. These models **can be further trained** with context-specific cell state trajectories during the **second stage** training, such as across human aging." |
| `MaxToki-217M-HF/config.json` | `vocab_size: 20275`, `max_position_embeddings: 4096`, `bos=2`, `eos=3`. The vocabulary has **no `<boq>`, `<eoq>`, or time tokens** — those exist only in the full BioNeMo dictionary (23277 tokens). |
| bionemo-maxtoki `README.md` | Stage 1: "autoregressive … next-token prediction over rank value encoded gene expression sequences" (single cells). Stage 2 ("Second-stage training") is what "adds the TimeBetweenCells regression task" and the cell-paragraph / NextCell task (`--task-ratio 0.5` timelapse vs next-cell). |
| `import_hf.py` (HF→BioNeMo conversion) | copies the 20275 HF embedding/LM-head rows and leaves the other 3002 rows (`<boq>`, `<eoq>`, 3000 Δt tokens) at their fresh NeMo initialisation — **random, never trained**. |
| `MaxToki-217M-bionemo/context/model.yaml` | plain `MaxTokiConfig`, `seq_length: 4096` — the converted Stage-1 model, no fine-tuning recorded. |

So the released 217M/1B weights know how to **generate one cell** (that is literally their
training objective) but have **never seen** a multi-cell paragraph, a Δt token, or the
`<boq> … <eoq>` question block. In our NextCell prompt, every Δt token and the whole
question block are random embeddings. That explains §3.1 exactly: the model is pushed
off its training distribution and falls into a generic default ranking.

Implications:
- **NextCell zero-shot with this checkpoint cannot work**, regardless of prompt or
  sampling. Sampling instead of greedy will give different noise, not a cell.
- The **TimeBetweenCells zero-shot numbers in `out/*_seq8k`** read the model's logits
  over the same untrained numeric tokens at an untrained `<eoq>`. They are not
  predictions from a trained TBC head either. (The Stage-2 fine-tune you already ran —
  `stage2_finetune.py`, `out/finetune_pdk4_predict_summary.csv` — is the version where
  the time head *was* trained on this trajectory format; its Δt deltas are ~0.)
- The 1B checkpoint is the same situation (`MaxToki-1B-HF` is also a Stage-1 release).

What would make NextCell possible:
1. **Stage-2 training** on aging-SKM cell paragraphs (NVIDIA's `assemble-paragraphs` →
   `finetune` path with `--task-ratio` < 1 so NextCell examples are included), starting
   from the 217M Stage-1 weights. That teaches `<boq>/<eoq>/Δt` and multi-cell context.
   Your `stage2_finetune.py` trains only a TBC head on frozen weights with a TBC-only
   row format, so it does not teach generation.
2. Or a **Stage-1-native experiment**: no time tokens, no question block — prompt with
   a *partial* cell (e.g. the old cell's top-N genes with PDK4 removed/moved) and let
   the Stage-1 model complete the rest of the ranking. That is in-distribution for this
   checkpoint and still gives a perturbed-vs-baseline ranked list, but it is
   "complete this cell", not "predict the next cell in time".

### 3.3 The token dictionary is shifted against the weights — this affects the TBC results too

Checked 2026-10-08 by loading the converted distcp weights (`weights/__0_*.distcp`) and
comparing to `MaxToki-217M-HF/model.safetensors` row by row:

- Rows 0–20274 of the BioNeMo embedding and LM head are a **byte-exact positional copy**
  of the HF model (max |diff| = 0.0). Rows 20275–23276 have row norms 0.702 ± 0.014 —
  the signature of `N(0, 0.02)` init (trained rows: 0.73 ± 0.09). **Untrained.**
- The HF model's vocabulary (`deltaai/src/maxToki/resources/token_dictionary_v1.json`,
  20275 entries) is `<pad>=0, <mask>=1, <bos>=2, <eos>=3, genes 4…20274`.
- The dictionary shipped next to the checkpoint (`MaxToki-217M-bionemo/context/token_dictionary.json`,
  23277 entries) is `<pad>=0, <mask>=1, <eos>=2, genes 3…20273, numeric 20274…23273,
  <boq>=23274, <eoq>=23275, <bos>=23276`. **Every gene id is HF id − 1.**

So in every run that used this checkpoint+dictionary (all zero-shot TBC runs in `out/`,
and the NextCell smokes), gene X was embedded with the row trained for the gene *before*
X in ENSG order, `<eos>` was embedded as HF's `<bos>`, and `<bos>`, `<boq>`, `<eoq>` and
all 3000 Δt tokens were random vectors. The 1B checkpoint's dictionary (`20277`, HF
layout + `<boq>/<eoq>`) does not have the shift, so this is specific to the 217M
"full" dictionary someone assembled for the time tokens.

`delta/configs/token_dictionary_217m_aligned.json` = HF layout + numeric 20275…23274 +
`<boq>=23275`, `<eoq>=23276` (same vocab size, so the weights load unchanged; the time
tokens are still random — nothing trained exists for them).

**Test (jobs 22766096 / 22766317, H200, `delta/slurm/_run_tbc_dict_test.sh`)** — the
exact PDK4-inhibit TBC spec behind `out/combined_aggregated_8k_evenly_3gene.png`
(`pdk4_evenly_seq8k.yaml`, 2000 OM queries, 3 YM2 context cells), four ways.
Re-randomization = re-draw rows ≥ 20275 of embedding + LM head with seed 1
(`delta/slurm/_torch_pipeline_entry_rerand.py`, applied to both model loads).

| run | dictionary | untrained rows | mean Δt ± SEM | per-cell r vs A | baseline pred. mean |
|---|---|---|---|---|---|
| original (May) | shifted | as converted | −37.7 ± 2.2 | 0.90 | −2210 |
| **A** | shifted | as converted | −37.4 ± 2.2 | — | −2212 |
| **B** | shifted | re-drawn | −62.2 ± 2.3 | 0.24 | −625 |
| **C** | aligned | as converted | −176 ± 11 | 0.19 | −2384 |
| **D** | aligned | re-drawn | −682 ± 25 | (0.34 vs C) | −623 |

Figure: `delta/figs/tbc_dict_test_22766317.{png,svg,csv}`.

Reading it:
- A reproduces May (r = 0.90; bf16 noise only). The pipeline is deterministic.
- **Re-drawing the untrained rows changes the per-cell predictions almost completely**
  (baseline predictions A vs B: r = −0.14; Δt r = 0.24). The `<eoq>`→numeric-token
  readout is a function of the random init, not of trained weights.
- **The absolute predictions are not times.** Baseline Δt predictions average −2200
  (true Δt_q for these cells is −67…0). Only the perturbed−baseline *difference* was
  ever small enough to look like pseudotime units.
- Aligning the dictionary (C) changes Δt 5× (−37 → −176) and the per-cell values
  (r = 0.19 vs A): the May number also depended on the gene shift.
- 51 % of cells have Δt = 0 in every variant: PDK4 is absent from those query cells,
  baseline and perturbed rows are identical, so the readout is identical. This part is
  structural, not learned. Among PDK4-present cells the mean is negative in all four
  variants (−76 / −127 / −359 / −1388) with more cells < 0 than > 0 — the *sign* of the
  PDK4 effect survived re-randomization while its magnitude and per-cell pattern did not.
  Whether that sign is biology or a systematic effect of "the token before `<eoq>` is now
  PDK4" (inhibit moves the gene to the last position) is untested; a random-gene inhibit
  control (`scripts/torch_pipeline/negative_control.py`) is the way to tell. Note the
  published-style figure has IRS2 *inhibit* at +41, so inhibition is not uniformly
  negative.

Conclusion for the zero-shot TBC results: with this checkpoint there is no trained time
head. The Δt numbers in `out/*_seq8k` (and the figure) are produced by random `<eoq>`/Δt
rows applied to gene-shifted inputs; they are reproducible run-to-run because the random
rows are fixed in the checkpoint, not because they measure time. A trained TBC head
requires Stage-2 training (or NVIDIA's Stage-2 weights, which are not in the HF release).

## 4. Order of work

1. **Sanity gate** — **DONE, FAILED (§3.1)**, and the cause is identified (§3.2): the
   checkpoint is Stage-1 only; `<boq>/<eoq>/Δt` tokens are untrained. No prompt or
   sampling change fixes this. Options are Stage-2 training or a Stage-1-native
   cell-completion experiment (§3.2).
   - check whether the muscle-fine-tuned checkpoint (`out/finetune_pdk4_*`,
     `scripts/torch_pipeline/stage2_finetune.py`) generates sane cells.
   If none produces muscle-like cells, NextCell is not usable here and TimeBetweenCells
   stays the readout.
2. **Only if generation looks like muscle:** add the Δt_q knob and the younger-scorer,
   switch the spec to `context.strategy: self` (or `prefix` + `include_self`) on old
   cells, run on H200 (~2 min per generation at the 4096 cap; ~1 min if `max_tokens`
   is cut to ~2200).

## 5. Cost facts for planning (H200, 217M, greedy, 4096-token cap)

- ~33 tok/s → ~2.1 min per generated cell; 2 generations per query.
- 20.9 GB peak GPU memory (`buffer_size_gb: 16`).
- 1 h interactive slot fits 10 queries. Request `--cpus-per-task=2` on H200 interactive
  or the job queues behind nodes with no idle CPUs.
