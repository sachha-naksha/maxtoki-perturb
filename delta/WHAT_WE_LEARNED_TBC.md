# What we learned: PDK4 zero-shot TimeBetweenCells reproduction

Run completed on Delta compute node `gpua023.delta.ncsa.illinois.edu` under `bhdw-delta-gpu`, Slurm job 22768554 (GPU: NVIDIA A100-SXM4-40GB, 40960 MiB).

## Result

The original numerical effect was not reproduced after correcting the vocabulary.

| Measure | Original | Official-ID rerun |
|---|---:|---:|
| Old query cells | 2000 | 2000 |
| PDK4-present queries | 983 | 983 |
| Mean ΔΔt, all cells | -37.707003 | -179.420755 |
| Mean ΔΔt, PDK4 present | -76.718215 | -365.047315 |
| Mean squared ΔΔt | 11440.962849 | 272276.260771 |
| Mean baseline prediction | -2209.480469 | -2384.770508 |

Across-cell ΔΔt Pearson r: 0.213730; PDK4-present r: 0.082845.
Mean absolute difference from original ΔΔt: 227.541715.
Sign agreement among PDK4-present queries: 65.62%.
All 1017 PDK4-absent queries have exactly zero paired effect; their inputs are unchanged by inhibition.

## Matched setup and checks

- Original: `out/pdk4_217m_inhibit_evenly_seq8k`; new results: `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554`. Original artifacts were preserved.
- Stage-1 MaxToki-217M checkpoint, one GPU, microbatch 4 (pipeline default), bf16-mixed, TP=PP=CP=1, sequence capacity 8192.
- Three YM2 (34-year-old) context cells selected evenly across pseudotime: ['CELL1761_N1_2_1_11_1', 'CELL3002_N1_2_1_11_1', 'CELL1480_N1_1_11_1'].
- All 2000 OM6/OM9 (80-year-old) cells as queries; PDK4 inhibition moves its existing gene token to the end of the query gene ranking.
- PDK4 is `ENSG00000004799`, official token ID 63 (original incorrectly used 62).
- Cell IDs, context IDs, query pseudotimes, donor labels, and PDK4-presence flags match row for row.
- All 2000 baseline and 2000 perturbed sequences match the old sequences exactly after mapping old IDs to the official IDs. This isolates the vocabulary change while retaining the original TBC prompt grammar.

## Dictionary handling

The requested NVIDIA resource, `resources/token_dictionary_v1.json`, contains 20275 entries and no temporal question/time tokens.
Both model `context/token_dictionary.json` files remain exact copies of that official file.
This TBC run uses `delta/configs/token_dictionary_tbc_official_extended.json`: every official mapping plus the checkpoint's 3000 numeric entries and `<boq>/<eoq>`, appended without changing any official ID.
Its 23277-entry mapping agrees with the corrected checkpoint mapping; BOS=2, EOS=3, BOQ=23275, EOQ=23276.
Official resource SHA256: `b4577f7a983ae1878e1c5db8862ccb46987e904bc2759bd305960fbb838ccb15`.
Extended dictionary SHA256: `79f7b2f8446633373ecbf239611bfe2373cad829951449c7a49b76c3bdfaa8a1`.
No checkpoint weights were trained or re-randomized in this run.

## Interpretation and limits

ΔΔt here means `predicted_Δt_inhibited − predicted_Δt_baseline`. The saved key `delta_t` and standard plots' “Δt” refer to this paired difference.
The quantity called `mean_mse` by the pipeline is mean squared paired difference, not prediction error against a time target.
The headless readout is the numeric-token probability-weighted expectation at `<eoq>`; it is not an independently trained age regressor.

A separate source-level scaling issue exists in this local BioNeMo code: `MaxTokiFineTuneModel.__init__` defaults `label_scalar` to 200 and `forward` multiplies the expected numeric value by it. The config's `configure_model` passes the numeric mask/map but does not pass `label_scalar`, despite `MaxTokiMultitaskFineTuneConfig.label_scalar` being 1. This reproduction leaves that historical behavior intact. Thus the reported raw ΔΔt is 200 times the numeric-readout difference; dividing the new mean by 200 gives -0.897104, and the original mean gives -0.188535. This rescaling alone does not validate the untrained readout. Training/inference scaling must be explicitly matched before Stage 2 results are interpreted.
Source: `deltaai/src/maxToki/sub-packages/bionemo-maxtoki/src/bionemo/maxtoki/model.py`, constructor, `configure_model`, and `forward`.

Prior checkpoint and re-randomization audits in `delta/NEXTCELL_WHAT_WE_LEARNED.md` establish that these release weights are Stage 1 and the temporal rows are untrained.
Correct gene IDs restore the pretrained gene semantics, but do not train the temporal task.
Consequently even a persistent negative mean does not establish rejuvenation, chronological years, or calibrated pseudotime change.

For comparison with the historical run, this rerun retains its TBC grammar: adjacent context cells without inter-cell time tokens, then `<boq> query_genes <eoq> dummy_numeric`.
NVIDIA's temporal paragraph format includes inter-cell time tokens. Matching that format and validating a trained temporal checkpoint remain necessary before biological interpretation.
Cell-level error bars in the comparison plot are descriptive; two old donors do not provide independent donor-level evidence of a causal effect.

## Artifacts and reproduction

- Standard ΔΔt plots: `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554/viz/` (distribution, pseudotime, donor, and PDK4-presence views).
- Comparison plot: `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554/comparison/delta_delta_time_comparison.png` and `.svg`.
- Per-cell and per-donor comparisons: `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554/comparison/per_cell_comparison.csv`, `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554/comparison/per_donor_comparison.csv`.
- Full comparison metrics: `/projects/bhdw/asachan/methods/Earth_RL/maxtoki-perturb/out/pdk4_217m_inhibit_evenly_seq8k_official_delta_22768554/comparison/comparison_metrics.json`.
- Launch script: `delta/slurm/_run_pdk4_tbc_reproduce.sh`; comparison script: `delta/scripts/compare_tbc_reproduction.py`.

From the repository root: `sbatch delta/slurm/_run_pdk4_tbc_reproduce.sh`.
