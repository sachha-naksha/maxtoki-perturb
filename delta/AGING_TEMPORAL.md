# Joint TBC and NextCell on aging SKM

## Stage 2 gene-inhibition probes: PDK4, IRS2, IRS1 (2026-10-09)

Paired TimeBetweenCells probes were run with both Stage 2 cross-donor
checkpoints (out/aging_skm_cross_donor_training_v1: selected 100-step,
OM9-minimum validation loss 1.942; final 500-step, 1.95). Each probe reuses
the historical 2,000 old-donor query cells (722 OM6, 1,278 OM9; the same
cells for every gene) and the historical fixed YM2 context
(CELL1761_N1_2_1_11_1, CELL3002_N1_2_1_11_1, CELL1480_N1_1_11_1 at
pseudotime 1, 57, 100), rebuilt in the Stage 2 prompt grammar: official
vocabulary, inter-context intervals 56 and 43 inserted, query time never
supplied, numeric sentinel 0, 8k prompt budget, RoPE factor 4, SDPA,
label_scalar 200. Inhibition moves the existing gene token to the end of the
query ranking; absent-gene prompts are byte-identical and were verified to
give exactly zero delta. Delta is predicted perturbed interval minus predicted
control interval, in stored pseudotime units. CIs are 95 percent percentile
bootstraps over query cells within donor, with donor weights fixed; they do
not cover checkpoint or training-run variability.

Provenance of the perturbed prompts differs by gene. PDK4 used the stored
historical perturbed rows, cross-checked against the official-vocabulary
rerun (out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554). IRS2 used
its stored historical rows remapped through token names; no official rerun
exists, so no cross-check. IRS1 has no historical run at all: its inhibition
was derived by moving IRS1 to the end of the stored PDK4 baseline prompts,
which are byte-identical to the IRS2 baseline prompts across all 2,000
queries (control prompts are gene-independent). The IRS1 absent-gene check,
remap check and inhibition-semantics check all passed, but there is no
historical IRS1 readout to compare against.

Control predictions are identical across genes (same control prompts).
The 100-step checkpoint emits a constant -44.00 (population std 0.0005) for
every prompt; its MAE of 20.0 against the stored query-minus-context interval
is lower than the 500-step checkpoint's 31.8 only because the constant sits
near the target mean (-31.4). The 500-step checkpoint varies with the prompt
(std 13.7) but is less accurate and had a marginally worse validation loss.
Neither checkpoint predicts these intervals usefully.

Mean delta among queries where the gene was present before editing
(equal-weight pooled OM6+OM9; per-donor in bar_statistics.csv):

| Gene | Present / 2,000 | 100-step (selected) | 500-step (final) | 500-step OM6 | 500-step OM9 | Historical / 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDK4 | 983 | +0.00006 | -0.26 (-0.48, -0.04) | -0.21 (-0.51, +0.10) | -0.30 (-0.63, +0.05) | -0.38 |
| IRS2 | 256 | +0.0003 | +3.01 (+2.50, +3.56) | +2.25 (+1.56, +3.02) | +3.58 (+2.85, +4.33) | +1.59 |
| IRS1 | 123 | -0.0002 | +4.00 (+3.35, +4.66) | +5.21 (+4.17, +6.23) | +3.25 (+2.41, +4.08) | none |

Pooled over all 2,000 queries (absent genes contribute zero) the 500-step
means are PDK4 -0.13 (-0.23, -0.01), IRS2 +0.39 (+0.31, +0.47) and IRS1
+0.25 (+0.19, +0.31). The historical column divides the archived readout by
its 200 output multiplier only; vocabulary, prompts and weights differ.

Descriptive same-cell agreement of the 500-step deltas with the historical
per-cell deltas: PDK4 sign agreement 0.50 and Spearman 0.01 (no per-cell
correspondence); IRS2 sign agreement 0.77 and Spearman 0.54. The 100-step
IRS2 deltas, although of order 1e-4, rank-correlate 0.89 with the historical
IRS2 deltas, so the collapsed checkpoint retains a faint input-dependent
residual. IRS1 500-step and 100-step deltas disagree in sign for 79 percent
of nonzero cells.

Interpretation. The 100-step checkpoint ignores its input, so its zero
deltas are not a null result. At 500 steps the model responds to the prompt
edit: PDK4 inhibition gives a shift of about 2 percent of the prediction
spread whose CI barely excludes zero and which has no per-cell relation to
the historical run; IRS2 and IRS1 inhibition give positive shifts of 3 to 4
pseudotime units with CIs well clear of zero, consistent in sign across the
training donor OM6 and the validation donor OM9, and for IRS2 consistent in
sign with the archived readout. These are predicted prompt-edit responses of
a 500-step, two-donor Stage 2 model with no demonstrated interval accuracy,
not measured perturbation biology. The released 217M weights are Stage 1
only (see delta/NEXTCELL_WHAT_WE_LEARNED.md section 3.2): the interval
tokens, question block and multi-cell context were learned solely from this
1,000-trajectory run, so none of these numbers speak to what the published
trajectory-trained MaxToki would predict. OM6 cells were in training; OM9
selected the checkpoint. The IRS1 gene-present sample is 123 cells (47 OM6,
76 OM9).

Jobs (one GPU, 2 CPUs; all exit 0:0; 12-query independent-reload repeat was
bit-identical for every checkpoint):

| Gene | Prep (CPU) | Inference (GPU) | Plot (CPU) | Output |
| --- | --- | --- | --- | --- |
| PDK4 | 22784066 | 22784079, 13m53s, A100-40GB | 22784230 | out/pdk4_stage2_tbc_v1 |
| IRS2 | 22784473 | 22784474, 13m08s, A100 | 22784488 | out/irs2_stage2_tbc_v1 |
| IRS1 | 22784599 | 22784609, 12m40s, A100 | 22784747 | out/irs1_stage2_tbc_v1 |

IRS1 prep job 22784584 failed in the pytest stage on a stub record in a new
unit test (missing direction/apply_to/n_context_cells fields) before writing
any output; the test was fixed and 22784599 passed all 27 tests. GPU jobs
peaked at 8.3 GB host RAM; the run wrappers now request 16 GB, 30 minutes
and include gpuA40x4-interactive. Each output directory holds data/
(manifest, references, paired datasets), runs/{selected,final}/ (rows.json,
scores.npz, run.json, repeatability.json) and plots/ (summary.json,
bar_statistics.csv, <gene>_inhibition_tbc{,_gene_present,_gene_absent} and,
for PDK4 and IRS2, <gene>_historical_comparison).

Scripts: scripts/torch_pipeline/prepare_pdk4_stage2.py now takes --gene
(PDK4, IRS2, IRS1), makes --corrected optional, and derives the
perturbation from another gene's baseline prompts when the source run
belongs to a different gene; run_pdk4_stage2.py reads the gene from the
prepared manifest; plot_pdk4_stage2.py takes --gene-symbol. Wrappers:
delta/slurm/_{prepare,run,plot}_{pdk4,irs2,irs1}_stage2.sh. The
interactive QOS allows two queued jobs per user, so chains were submitted
step by step.


## Corrected active Stage 2: train YM2 + OM6; validate OM9


Completed: GPU job 22783214 exited 0:0 after 23m40s, finishing all 500
optimizer steps and OM9 inference. Comparison job 22783224 exited 0:0 after
52s and passed its 23 tests. The best checkpoint was saved after 100 optimizer
steps (filename step=99); its full mixed validation loss was 1.94202459.
The separate final 500-step checkpoint is also preserved. Inference used the
selected 100-step checkpoint, not the final checkpoint.

Matched OM9 results, averaged equally across the three Annotations:

| Metric | Queries | MaxToki | Context linear | Fitted training baseline |
| --- | ---: | ---: | ---: | ---: |
| TBC MAE (lower better) | 100 | 34.917 | 18.178 | 17.850 (ridge clock) |
| NC Jaccard@100 (higher better) | 10 | 0.1459 | 0.0900 | 0.1691 (linear rank trend) |

The corrected run still underperforms the fitted baselines on both tasks.
All 10 NC generations reached the 2048-token limit without EOS; none contained
invalid or duplicate gene tokens. The NC sample remains small. These are
OM9 validation results, because OM9 selected the checkpoint.

Baseline fits use 1,339 cells from the full prepared training set. This matches
the prepared training-cell pool, but does not prove identical cell exposure to
the selected early checkpoint, which consumed only 400 mixed training rows.
The full 500-step run consumed 2,000 rows. Do not describe this as an exact
exposure-matched comparison to the 100-step checkpoint.

Results: out/aging_skm_cross_donor_comparison_v1/comparison_metrics.json.
Training summary: out/aging_skm_cross_donor_training_v1/training_summary.json.


The user corrected the earlier split: YM2-only optimization did not train
the young-to-old trajectory. The active run restarts from the original
BioNeMo checkpoint and trains joint TBC/NC on both YM2 and OM6. OM9 is excluded
from optimization and baseline fitting and is used for validation/checkpoint
selection. There is no independent test split in this three-donor experiment.

Prepared data: out/aging_skm_stage2_cross_donor_v1. CPU job 22783203 passed
43 tests and completed preparation. Each of 1,000 training trajectories uses
three same-Annotation context cells spanning both training donors, with
YM2 before OM6 and within-donor pseudotime ordering. Queries cover both
donors (499 YM2 / 501 OM6) and all three eligible Annotations. This produces
2,000 mixed rows (one TBC and one NC per trajectory), involving 1,339 distinct
training cells from the 2,452-cell training pool. This is a sampled initial
run, not an assertion that every eligible cell is optimized on.

Validation contains 100 unique OM9 queries, each paired with contexts from
the training donors, yielding 200 mixed validation rows. All validation rows
are scored during training. Signed raw obs[Pseudotime] differences remain
ground truth, including negative cross-donor intervals where the distributions
overlap. Donor age never replaces pseudotime.

The versioned runtime aging_stage2_runtime_v1 corrects NC autoregressive
answer masking (including EOQ-to-BOS and BOS-to-first-gene), normalizes both
numeric expectations and targets by label_scalar=200 before MSE, restores raw
pseudotime units at output, and saves checkpoints after current validation.
Old checkpoints retain their original runtime semantics.

GPU job 22783214 started on gpue05 (one H200); its eight runtime tests passed.
CPU matched-baseline job 22783217 completed with 23 tests passing.
CPU comparison job 22783224 is dependent on successful GPU completion.
The first full validation completed after 100 optimizer steps (joint loss
1.94202), and both best and last checkpoint directories were saved successfully.
This is a training-health check; it is not yet a model/baseline performance result.

The GPU runner delta/slurm/_run_aging_trajectory.sh runs runtime tests,
500 optimizer steps from the original checkpoint, then 100 TBC and 10 NC
validation queries using the checkpoint with minimum full validation loss.
It requests one GPU, two CPUs, 32 GB RAM and one hour across compatible
H200/A100 interactive partitions. Training outputs and protocol hashes are
preserved separately from historical runs.

Baselines fit the same training donors and, for the primary matched comparison,
the same distinct cells actually present in training rows. The full reference
bank remains available to verify validation contexts. Primary fits will be in
out/aging_skm_cross_donor_matched_baselines_v1; the earlier full-pool fit in
out/aging_skm_cross_donor_baselines_v1 is retained as an audit artifact.



## Historical benchmark: YM2-only training, old-donor transfer

The donor direction is YM2 -> OM6 or YM2 -> OM9. OM6 and OM9 are parallel old
donors, not successive aging stages. The earlier donor-local queries below
are historical runs; they did not evaluate this cross-donor aging direction.

Prepare with scripts/torch_pipeline/prepare_aging_direction.py. Context uses
three early YM2 cells of the same Annotation, selected independently of the
old query expression and pseudotime. The source-only lower pseudotime quartile
defines early context; the threshold expands minimally if fewer than three
distinct young time values are available. Targets retain the exact observed
pseudotime differences. Young/old pseudotime distributions overlap; donor age
is never substituted for pseudotime.

Baseline fitting uses only YM2 cells:

- Context linear projection (TBC): fit rank-expression versus the three known
  relative context times, then project the query expression onto that trend.
- YM2 ridge clock (TBC): fit expression-to-pseudotime regression within each
  Annotation and align its time origin using the supplied timed context.
- Context linear extrapolation (NC): extrapolate the context gene rank scores
  to the supplied query interval.
- YM2 linear rank trend (NC): fit each gene rank score versus pseudotime in the
  YM2 bank; use the context-inferred clock origin and supplied query interval.
- Nearest-expression context is a secondary TBC diagnostic only.

The TBC predictors never receive query pseudotime. NC predictors never use
query expression. Both model and baselines receive the same quantized prompt
intervals; scoring uses exact ground truth. Fits, reference hashes, row-level
predictions, signed-interval strata, per-Annotation results, and equal-weight
Annotation macro averages are stored with the benchmark.

Preparation job 22782700 passed 23 tests and saved
out/aging_skm_young_to_old_v1. There are 100 unique old queries per donor.
OM6 has 97 positive / 3 negative intervals; OM9 has 96 positive / 1 zero /
3 negative intervals. All are retained. Baseline job 22782710 passed 24 tests
and completed fits/scoring in out/aging_skm_young_to_old_baselines_v1.

Initial baseline-only results on all 100 queries/donor (Annotation macro
averages; these are not yet the matched 10-query NC model comparison):

| Baseline metric | OM6 | OM9 |
| --- | ---: | ---: |
| Context-linear TBC MAE | 51.33 | 49.46 |
| YM2 ridge-clock TBC MAE | 27.06 | 27.15 |
| Context-linear NC Jaccard@100 | 0.018 | 0.024 |
| YM2 linear-trend NC Jaccard@100 | 0.130 | 0.152 |

The oversized two-GPU request was cancelled. Job 22782723 uses one H200,
two CPUs and 32 GB host RAM and completed 100 TBC and 10 NC queries per old
donor in 24m53s (exit 0:0). Outputs are in out/aging_skm_young_to_old_eval_v1.
Dependent job 22782725 completed the scoring tests and matched comparison in
40s (exit 0:0), writing out/aging_skm_young_to_old_comparison_v1. These jobs use the existing 500-step
checkpoint, which was trained on within-YM2 trajectories; it has not been
retrained on cross-donor aging trajectories.

Completed matched results (equal-weight averages across the three Annotations):

| Task / target donor | Queries | MaxToki | Context linear | YM2 reference baseline |
| --- | ---: | ---: | ---: | ---: |
| TBC MAE / OM6 | 100 | 274.59 | 51.33 | 27.06 (ridge clock) |
| TBC MAE / OM9 | 100 | 468.03 | 49.46 | 27.15 (ridge clock) |
| NC Jaccard@100 / OM6 | 10 | 0.117 | 0.018 | 0.124 (linear rank trend) |
| NC Jaccard@100 / OM9 | 10 | 0.129 | 0.030 | 0.149 (linear rank trend) |

Lower TBC MAE is better; higher NC Jaccard is better. All methods had 100%
coverage on these matched queries. MaxToki underperforms the YM2-fitted
reference baselines in both tasks, although NC outperforms the context-only
linear extrapolation. All 20 NC outputs reached the 2048-token limit without
EOS; none had invalid or duplicate gene tokens. This remains a small NC sample.
Full per-Annotation, signed-interval, row-level and paired results are in
out/aging_skm_young_to_old_comparison_v1/comparison_metrics.json and
comparison_per_row.json.

Interpretation: Annotation-balanced sampling is not population prevalence.
Lead with Annotation macro averages and show per-Annotation results. ID1+
has only 11 young reference cells and three early context cells reused across
queries, so its performance estimate has limited independent context diversity.
Historical baseline outputs below are superseded for current comparisons.


The target interval is obs[Pseudotime] of the query minus obs[Pseudotime] of the
last context cell. Inter-context intervals use the same difference convention.
Donor age is never substituted for pseudotime. Cells without finite pseudotime
are excluded and recorded in excluded_cells.json; source cells remain intact. Continuous differences are saved
as ground truth; numeric-token rounding is recorded separately. The default
time_scale=1 preserves the existing pseudotime coordinate; increase it only when
the scaled differences fit the numeric vocabulary. Predictions are divided by
time_scale once at scoring.

Run all workloads on Delta compute nodes using bhdw. From maxtoki-perturb:

```bash
sbatch delta/slurm/_prepare_aging_temporal.sh out/aging_skm_temporal
sbatch delta/slurm/_run_aging_temporal.sh train \
  --data out/aging_skm_temporal \
  --checkpoint /projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo \
  --output out/aging_skm_joint_training --steps 500
```

Wait for preparation to succeed before training (or use --dependency=afterok:JOB).
Preparation runs the regression tests, reads authoritative H5AD obs, verifies
the rebuilt single-cell tokens, and creates a 1:1 mixture of supervised TBC and
NextCell examples. Context and query cells share donor and Annotation.
Default splits: YM2 train, OM6 validation, OM9 test. Override donor lists,
pseudotime column, trajectory column and tokenization budgets on prepare.
These three donors support a small transfer experiment; they do not reproduce
the paper's population aging corpus. Context cells in held-out donors are
provided at evaluation for in-context learning; none are used for optimization.

Training updates the backbone and LM head jointly with the upstream mixed MSE/CE
loss and SDPA attention. This is not frozen-backbone TBC-only regression.
The runner explicitly supplies the manifest label_scalar to the headless model constructor
during both training and inference. New preparations default to NVIDIA's label_scalar=200 normalization; the earlier smoke dataset used 1. Predictions restore the original pseudotime units.
Inference explicitly restores training RoPE factor=4 and SDPA; upstream defaults
would otherwise use factor=8 even when loading this trained checkpoint.
Historical checkpoint weights, scripts and outputs are left intact.

Successful training adds aging_temporal_manifest.json inside each saved
checkpoint. Choose a checkpoint using validation loss; use its directory:

```bash
sbatch delta/slurm/_run_aging_temporal.sh predict \
  --data out/aging_skm_temporal --checkpoint /path/to/trained/checkpoint \
  --task tbc --output out/aging_skm_tbc
sbatch delta/slurm/_run_aging_temporal.sh predict \
  --data out/aging_skm_temporal --checkpoint /path/to/trained/checkpoint \
  --task nc --limit 2 --output out/aging_skm_nc_smoke
sbatch delta/slurm/_run_aging_temporal.sh score \
  --data out/aging_skm_temporal --predictions out/aging_skm_tbc
sbatch delta/slurm/_run_aging_temporal.sh score \
  --data out/aging_skm_temporal --predictions out/aging_skm_nc_smoke
```

TBC metrics.json reports MAE/MSE/Pearson against exact pseudotime differences
and quantized-target errors. NextCell reports Jaccard@100, rank correlation on
shared genes, and EOS/invalid/duplicate diagnostics. The zero-interval and
copy-last-context baselines have been retired from active scoring; use the
young-to-old benchmark below for baseline comparisons.
NextCell target genes occur only in training answers and reference artifacts,
never in generation prompts. TBC inference rows use a dummy numeric answer.

Inference rejects Stage 1 and mismatched tokenizer/provenance. This workflow
trains a new headless SKM model; an external explicit-head aging HF checkpoint
requires NVIDIA import_regression_hf and its original tokenizer, interval units,
RoPE and label scaling. It must not be relabeled as this SKM-trained checkpoint.

## Validation on the actual SKM data

CPU job 22782148 passed all 42 regression tests and prepared
out/aging_skm_temporal_joint_v1. Of 3989 cells, 259 YM2 cells have missing
Pseudotime and are audited in excluded_cells.json. The remaining 3730 cells
supply 2000 training trajectories and 20 trajectories per evaluation split.

Joint GPU training job 22782175 completed a five-step execution check.
The smoke checkpoint filename's val_loss=0 is not evidence of validation quality.
Corrected TBC job 22782236 and NC job 22782252 completed inference and scoring:

- TBC: 20 OM9 queries, MAE 43.84 and MSE 2332.73; zero-interval baseline
  MAE 18.90 and MSE 695.70.
- NC: two OM9 queries, mean Jaccard@100 0.000; copy-context baseline 0.144.
  Both outputs reached EOS with no invalid or duplicate tokens.

These are pipeline smoke weights, not a calibrated temporal model. Longer
training and validation-based checkpoint selection remain necessary before
interpreting predictions. At the smoke-check stage no 500-step training run had been performed; the subsequent Stage 2 run is recorded below.
The first TBC output (out/aging_skm_tbc_smoke_v1) used mismatched upstream
RoPE defaults; its run.json marks it invalid. Corrected outputs are
out/aging_skm_tbc_smoke_v2 and out/aging_skm_nc_smoke_v2.

The container mount failure came from inherited APPTAINER_BIND and
SINGULARITY_BIND variables pointing to administrative-container paths absent
on compute nodes. Both runners now clear all four bind variables before
constructing their compute-node mounts. Missing --output-weights, a pandas
string fixture incompatibility, and the inference RoPE mismatch were corrected
and then exercised successfully. Missing pseudotime is excluded rather than
imputed. Add --score to predict to score automatically after inference.

## Stage 2 training run

Preparation job 22782348 creates out/aging_skm_stage2_v1 with label_scalar=200,
2000 training trajectories and 100 trajectories each for validation and test.
Training job 22782349 uses the pretrained BioNeMo checkpoint, 500 optimizer
steps, learning rate 5e-5, global batch size 4, RoPE factor 4, SDPA,
and NVIDIA's mixed TBC MSE / NC cross-entropy loss. It validates every 50 steps
over up to 100 batches. Output: out/aging_skm_stage2_training_v1.
The normalization changes the loss scale, not the observed pseudotime or time tokens.

Training job 22782349 completed successfully (exit 0:0) at 500 optimizer steps,
2000 consumed examples. Its final checkpoint is
aging_skm_joint/dev/checkpoints/epoch=0-val_loss=2.18-step=499-consumed_samples=2000.0-last
under the training output above. training_run.json records the settings and
validation history. Checkpoint filenames use the preceding validation pass,
so they cannot directly rank the saved weights. Explicit held-out evaluation
job 22782497 scores the final checkpoint into out/aging_skm_stage2_eval_v1.

Evaluation job 22782497 completed successfully (exit 0:0). Final-checkpoint
results, with intervals restored to original pseudotime units:

| Task / split | Queries | Model | Baseline |
| --- | ---: | --- | --- |
| TBC / OM6 validation | 100 | MAE 312.37; MSE 468676.16 | Zero: MAE 23.36; MSE 1047.76 |
| TBC / OM9 test | 100 | MAE 375.67; MSE 581660.74 | Zero: MAE 19.78; MSE 697.88 |
| NC / OM9 test smoke | 2 | Jaccard@100 0.074 | Copy context: 0.144 |

One NC output reached the 2048-token limit without EOS; both contained no
invalid or duplicate gene tokens. The final model underperforms these simple
baselines. Completed Stage 2 optimization is not evidence of a useful temporal
model. No best-checkpoint claim is made: these results use the last checkpoint.
The complete settings and summary are in
out/aging_skm_stage2_training_v1/training_run.json; per-query metrics are under
out/aging_skm_stage2_eval_v1. This run used the documented upstream 200 label
normalization consistently during training and inference; all interval tokens
and references retain their observed pseudotime units.

Re-run evaluation on compute nodes with:
```bash
sbatch delta/slurm/_evaluate_aging_stage2.sh   out/aging_skm_stage2_training_v1 out/aging_skm_stage2_v1 out/fresh_stage2_eval
```
