"""Compare the corrected-vocabulary PDK4 TBC rerun with its historical results."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run comparisons inside a Slurm compute allocation')
    import numpy as np
    import pandas as pd
    from datasets import load_from_disk
    from scipy.stats import pearsonr, spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--rerun', type=Path, required=True)
    p.add_argument('--legacy-dictionary', type=Path, required=True)
    p.add_argument('--dictionary', type=Path, required=True)
    args = p.parse_args()
    old = load_from_disk(str(args.original / 'baseline.dataset'))
    new = load_from_disk(str(args.rerun / 'baseline.dataset'))
    assert len(old) == len(new) == 2000
    for column in ['cell_id', 'context_cell_ids', 'query_pseudotime', 'group', 'gene_present_in_query']:
        assert list(old[column]) == list(new[column]), f'Pairing/selection mismatch: {column}'
    legacy = json.loads(args.legacy_dictionary.read_text())
    official_extended = json.loads(args.dictionary.read_text())
    official_path = Path('data/zero_shot/resources/token_dictionary_v1.json')
    official = json.loads(official_path.read_text())
    assert all(official_extended[k] == v for k, v in official.items())
    remap = {tid: official_extended[token] for token, tid in legacy.items()}
    remapped_counts = {}
    for condition in ['baseline', 'perturbed']:
        previous = load_from_disk(str(args.original / f'{condition}.dataset'))
        current = load_from_disk(str(args.rerun / f'{condition}.dataset'))
        assert list(previous['cell_id']) == list(current['cell_id'])
        count = 0
        for before, after in zip(previous['input_ids'], current['input_ids']):
            assert [remap[token] for token in before] == after, f'{condition} differs beyond ID remapping'
            count += 1
        remapped_counts[condition] = count
    a = np.load(args.original / 'scores.npz', allow_pickle=True)
    b = np.load(args.rerun / 'scores.npz', allow_pickle=True)
    da = a['delta_t'].astype(float).ravel()
    db = b['delta_t'].astype(float).ravel()
    present = b['gene_present'].astype(bool).ravel()
    assert np.array_equal(a['gene_present'], b['gene_present'])
    assert np.allclose(db, b['perturbed'].ravel() - b['baseline'].ravel())
    assert np.isfinite(da).all() and np.isfinite(db).all()
    assert np.count_nonzero(db[~present]) == 0, 'Absent-gene control changed'
    frame = pd.DataFrame({
        'cell_id': list(new['cell_id']), 'donor': list(new['group']),
        'pseudotime': list(new['query_pseudotime']), 'pdk4_present': present,
        'original_baseline': a['baseline'].ravel(), 'rerun_baseline': b['baseline'].ravel(),
        'original_perturbed': a['perturbed'].ravel(), 'rerun_perturbed': b['perturbed'].ravel(),
        'original_delta_delta_t': da, 'rerun_delta_delta_t': db,
    })
    out = args.rerun / 'comparison'
    out.mkdir(exist_ok=True)
    frame.to_csv(out / 'per_cell_comparison.csv', index=False)
    grouped = []
    for donor in sorted(frame.donor.unique()):
        for present_only in [False, True]:
            mask = (frame.donor == donor).to_numpy()
            if present_only:
                mask &= present
            for name, values in [('Original', da), ('Official IDs', db)]:
                v = values[mask]
                grouped.append({'donor': donor, 'subset': 'PDK4 present' if present_only else 'All',
                                'run': name, 'n': len(v), 'mean_delta_delta_t': float(v.mean()),
                                'sem': float(v.std(ddof=1) / np.sqrt(len(v)))})
    donor_frame = pd.DataFrame(grouped)
    donor_frame.to_csv(out / 'per_donor_comparison.csv', index=False)
    def describe(scores, delta):
        return {'mean_delta_delta_t': float(delta.mean()),
                'mean_delta_delta_t_present': float(delta[present].mean()),
                'mean_squared_delta_delta_t': float(np.mean(delta**2)),
                'mean_baseline_prediction': float(scores['baseline'].mean()),
                'mean_perturbed_prediction': float(scores['perturbed'].mean())}
    metrics = {
        'job_id': os.environ['SLURM_JOB_ID'], 'compute_node': socket.gethostname(), 'gpu_info': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip().splitlines()[0], 'original': str(args.original), 'rerun': str(args.rerun),
        'n_rows': len(db), 'n_pdk4_present': int(present.sum()), 'n_absent_zero_effect': int((~present).sum()),
        'context_cell_ids': new[0]['context_cell_ids'], 'remapped_rows_match': remapped_counts,
        'official_gene_ids_preserved': True,
        'official_dictionary_sha256': hashlib.sha256(official_path.read_bytes()).hexdigest(),
        'extended_dictionary_sha256': hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        'original_stats': describe(a, da), 'rerun_stats': describe(b, db),
        'delta_delta_t_pearson_r': float(pearsonr(da, db).statistic),
        'delta_delta_t_spearman_r': float(spearmanr(da, db).statistic),
        'delta_delta_t_present_pearson_r': float(pearsonr(da[present], db[present]).statistic),
        'baseline_pearson_r': float(pearsonr(a['baseline'].ravel(), b['baseline'].ravel()).statistic),
        'delta_delta_t_mae_vs_original': float(np.mean(np.abs(db-da))),
        'present_sign_agreement': float(np.mean(np.sign(da[present]) == np.sign(db[present]))),
        'numerically_reproduced': bool(np.allclose(da, db, rtol=1e-4, atol=1e-3)),
        'headless_scale_constructor_default': 200.0,
    }
    (out / 'comparison_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(da[present], db[present], s=8, alpha=.3)
    limits = [min(da[present].min(), db[present].min()), max(da[present].max(), db[present].max())]
    axes[0].plot(limits, limits, '--', color='gray')
    axes[0].set(xlabel='Original ΔΔt', ylabel='Official-ID rerun ΔΔt',
                title=f'PDK4-present cells (n={present.sum()})\nr={metrics["delta_delta_t_present_pearson_r"]:.3f}')
    for label, values in [('Original', da), ('Official IDs', db)]:
        axes[1].hist(values[present], bins=45, density=True, histtype='step', linewidth=2, label=label)
    axes[1].set(xlabel='ΔΔt = predicted Δt inhibited − baseline', ylabel='Density', title='PDK4-present distribution')
    axes[1].legend()
    donors = sorted(frame.donor.unique())
    for offset, name in [(-.18, 'Original'), (.18, 'Official IDs')]:
        rows = donor_frame[(donor_frame['run'] == name) & (donor_frame['subset'] == 'All')].set_index('donor').loc[donors]
        axes[2].bar(np.arange(len(donors))+offset, rows.mean_delta_delta_t, width=.36,
                    yerr=1.96*rows['sem'], capsize=4, label=name)
    axes[2].set_xticks(np.arange(len(donors)), donors)
    axes[2].set(ylabel='Mean ΔΔt (model readout units)', title='All old query cells by donor\nError bars: ±1.96 cell-level SEM')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('PDK4 inhibition: original versus official gene IDs; untrained temporal readout', fontsize=12)
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(out / f'delta_delta_time_comparison.{extension}', dpi=160)
    plt.close(fig)
    assert len(list((args.rerun / 'viz').glob('*.png'))) >= 8, 'Standard rerun plots missing'
    oldstats = metrics['original_stats']; newstats = metrics['rerun_stats']
    report = f'''# What we learned: PDK4 zero-shot TimeBetweenCells reproduction

Run completed on Delta compute node `{metrics['compute_node']}` under `bhdw-delta-gpu`, Slurm job {metrics['job_id']} (GPU: {metrics['gpu_info']}).

## Result

The original numerical effect {'was' if metrics['numerically_reproduced'] else 'was not'} reproduced after correcting the vocabulary.

| Measure | Original | Official-ID rerun |
|---|---:|---:|
| Old query cells | {len(db)} | {len(db)} |
| PDK4-present queries | {present.sum()} | {present.sum()} |
| Mean ΔΔt, all cells | {oldstats['mean_delta_delta_t']:.6f} | {newstats['mean_delta_delta_t']:.6f} |
| Mean ΔΔt, PDK4 present | {oldstats['mean_delta_delta_t_present']:.6f} | {newstats['mean_delta_delta_t_present']:.6f} |
| Mean squared ΔΔt | {oldstats['mean_squared_delta_delta_t']:.6f} | {newstats['mean_squared_delta_delta_t']:.6f} |
| Mean baseline prediction | {oldstats['mean_baseline_prediction']:.6f} | {newstats['mean_baseline_prediction']:.6f} |

Across-cell ΔΔt Pearson r: {metrics['delta_delta_t_pearson_r']:.6f}; PDK4-present r: {metrics['delta_delta_t_present_pearson_r']:.6f}.
Mean absolute difference from original ΔΔt: {metrics['delta_delta_t_mae_vs_original']:.6f}.
Sign agreement among PDK4-present queries: {metrics['present_sign_agreement']:.2%}.
All {metrics['n_absent_zero_effect']} PDK4-absent queries have exactly zero paired effect; their inputs are unchanged by inhibition.

## Matched setup and checks

- Original: `{args.original}`; new results: `{args.rerun}`. Original artifacts were preserved.
- Stage-1 MaxToki-217M checkpoint, one GPU, microbatch 4 (pipeline default), bf16-mixed, TP=PP=CP=1, sequence capacity 8192.
- Three YM2 (34-year-old) context cells selected evenly across pseudotime: {metrics['context_cell_ids']}.
- All 2000 OM6/OM9 (80-year-old) cells as queries; PDK4 inhibition moves its existing gene token to the end of the query gene ranking.
- PDK4 is `ENSG00000004799`, official token ID 63 (original incorrectly used 62).
- Cell IDs, context IDs, query pseudotimes, donor labels, and PDK4-presence flags match row for row.
- All 2000 baseline and 2000 perturbed sequences match the old sequences exactly after mapping old IDs to the official IDs. This isolates the vocabulary change while retaining the original TBC prompt grammar.

## Dictionary handling

The requested NVIDIA resource, `resources/token_dictionary_v1.json`, contains 20275 entries and no temporal question/time tokens.
Both model `context/token_dictionary.json` files remain exact copies of that official file.
This TBC run uses `{args.dictionary}`: every official mapping plus the checkpoint's 3000 numeric entries and `<boq>/<eoq>`, appended without changing any official ID.
Its 23277-entry mapping agrees with the corrected checkpoint mapping; BOS=2, EOS=3, BOQ=23275, EOQ=23276.
Official resource SHA256: `{metrics['official_dictionary_sha256']}`.
Extended dictionary SHA256: `{metrics['extended_dictionary_sha256']}`.
No checkpoint weights were trained or re-randomized in this run.

## Interpretation and limits

ΔΔt here means `predicted_Δt_inhibited − predicted_Δt_baseline`. The saved key `delta_t` and standard plots' “Δt” refer to this paired difference.
The quantity called `mean_mse` by the pipeline is mean squared paired difference, not prediction error against a time target.
The headless readout is the numeric-token probability-weighted expectation at `<eoq>`; it is not an independently trained age regressor.

A separate source-level scaling issue exists in this local BioNeMo code: `MaxTokiFineTuneModel.__init__` defaults `label_scalar` to 200 and `forward` multiplies the expected numeric value by it. The config's `configure_model` passes the numeric mask/map but does not pass `label_scalar`, despite `MaxTokiMultitaskFineTuneConfig.label_scalar` being 1. This reproduction leaves that historical behavior intact. Thus the reported raw ΔΔt is 200 times the numeric-readout difference; dividing the new mean by 200 gives {newstats['mean_delta_delta_t']/200:.6f}, and the original mean gives {oldstats['mean_delta_delta_t']/200:.6f}. This rescaling alone does not validate the untrained readout. Training/inference scaling must be explicitly matched before Stage 2 results are interpreted.
Source: `deltaai/src/maxToki/sub-packages/bionemo-maxtoki/src/bionemo/maxtoki/model.py`, constructor, `configure_model`, and `forward`.

Prior checkpoint and re-randomization audits in `delta/NEXTCELL_WHAT_WE_LEARNED.md` establish that these release weights are Stage 1 and the temporal rows are untrained.
Correct gene IDs restore the pretrained gene semantics, but do not train the temporal task.
Consequently even a persistent negative mean does not establish rejuvenation, chronological years, or calibrated pseudotime change.

For comparison with the historical run, this rerun retains its TBC grammar: adjacent context cells without inter-cell time tokens, then `<boq> query_genes <eoq> dummy_numeric`.
NVIDIA's temporal paragraph format includes inter-cell time tokens. Matching that format and validating a trained temporal checkpoint remain necessary before biological interpretation.
Cell-level error bars in the comparison plot are descriptive; two old donors do not provide independent donor-level evidence of a causal effect.

## Artifacts and reproduction

- Standard ΔΔt plots: `{args.rerun}/viz/` (distribution, pseudotime, donor, and PDK4-presence views).
- Comparison plot: `{out}/delta_delta_time_comparison.png` and `.svg`.
- Per-cell and per-donor comparisons: `{out}/per_cell_comparison.csv`, `{out}/per_donor_comparison.csv`.
- Full comparison metrics: `{out}/comparison_metrics.json`.
- Launch script: `delta/slurm/_run_pdk4_tbc_reproduce.sh`; comparison script: `delta/scripts/compare_tbc_reproduction.py`.

From the repository root: `sbatch delta/slurm/_run_pdk4_tbc_reproduce.sh`.
'''
    Path('delta/WHAT_WE_LEARNED_TBC.md').write_text(report)
    print(json.dumps(metrics, indent=2))
    print('Wrote delta/WHAT_WE_LEARNED_TBC.md')


if __name__ == '__main__':
    main()
