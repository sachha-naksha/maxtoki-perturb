"""Rebuild aging SKM counts and per-cell RVE sequences from official resources.

Run from the repository root with Python providing anndata and datasets:
    python scripts/torch_pipeline/rebuild_aging_skm.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from preprocess import preprocess
from tokenizer import CellTokenizer


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this rebuild inside a Slurm compute allocation")
    import anndata as ad
    from datasets import Dataset, load_from_disk
    import pyarrow as pa
    from scipy import sparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/zero_shot/rna_zero_shot_input.h5ad'))
    parser.add_argument('--h5ad-out', type=Path, default=Path('data/zero_shot/rna_zero_shot.preprocessed.h5ad'))
    parser.add_argument('--dataset-out', type=Path, default=Path('data/zero_shot/aging_skm.dataset'))
    parser.add_argument('--token-dictionary', type=Path, default=Path('data/zero_shot/resources/token_dictionary_v1.json'))
    parser.add_argument('--gene-median', type=Path, default=Path('data/zero_shot/resources/gene_median_dictionary_v1.json'))
    args = parser.parse_args()
    if args.dataset_out.exists():
        raise FileExistsError(f'Dataset already exists: {args.dataset_out}; choose a fresh --dataset-out')
    if args.source.resolve() == args.h5ad_out.resolve():
        raise ValueError('Source and output H5AD must differ')
    temporary = args.h5ad_out.with_name(args.h5ad_out.stem + f'.rebuilding-{os.environ["SLURM_JOB_ID"]}.h5ad')
    if temporary.exists():
        raise FileExistsError(temporary)
    tokenizer = CellTokenizer(args.token_dictionary, args.gene_median)
    summary = preprocess(str(args.source), str(temporary), counts_layer='counts', keep_ages=[34, 80])
    adata = ad.read_h5ad(temporary)
    ids = adata.var['ensembl_id'].astype(str)
    mapped = ids.map(tokenizer.token_dict)
    if mapped.isna().any():
        raise ValueError('Unmapped genes remain in rebuilt H5AD')
    adata.var['maxtoki_token'] = mapped.astype(np.int64)
    counts = adata.X
    values = counts.data if sparse.issparse(counts) else np.asarray(counts).ravel()
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.round(values)).any():
        raise ValueError('X must contain finite, nonnegative integer raw counts')
    sequences = []
    for i in range(adata.n_obs):
        row = counts[i]
        expression = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        sequences.append(tokenizer.tokenize_expression(ids, expression, n_counts=float(adata.obs['nCount_RNA'].iloc[i])))
    gene_ids = set(tokenizer._gene_ids.values())
    for sequence in sequences:
        assert sequence[0] == tokenizer.bos_id and sequence[-1] == tokenizer.eos_id
        assert set(sequence[1:-1]) <= gene_ids
    obs = adata.obs.copy()
    obs.insert(0, 'cell_id', adata.obs_names.astype(str))
    # Arrow/HF cannot cast an all-null categorical dictionary to a null feature.
    # Keep the original categorical metadata in H5AD; export its values to HF.
    for column in obs.select_dtypes(include=['category']).columns:
        obs[column] = obs[column].astype(object)
    table = pa.Table.from_pandas(obs.reset_index(drop=True), preserve_index=False)
    table = table.append_column('input_ids', pa.array(sequences, type=pa.list_(pa.int64())))
    table = table.append_column('length', pa.array([len(sequence) for sequence in sequences]))
    # Explicit content fingerprint avoids the container's Arrow/dill incompatibility.
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        fingerprint = hashlib.sha256(sink.getvalue()).hexdigest()
    dataset = Dataset(table, fingerprint=fingerprint)
    dataset.save_to_disk(str(args.dataset_out))
    # Reload both formats and validate the persisted data before replacing the old H5AD.
    adata.write_h5ad(temporary)
    reloaded = ad.read_h5ad(temporary)
    persisted = load_from_disk(str(args.dataset_out))
    assert list(persisted['cell_id']) == list(reloaded.obs_names.astype(str))
    assert list(persisted['input_ids']) == sequences
    assert np.array_equal(reloaded.var['maxtoki_token'], mapped.to_numpy())
    assert (reloaded.X != counts).nnz == 0 if sparse.issparse(counts) else np.array_equal(reloaded.X, counts)
    backup = None
    if args.h5ad_out.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = args.h5ad_out.with_name(args.h5ad_out.stem + f'.before-vocab-rebuild-{stamp}.h5ad')
        shutil.copy2(args.h5ad_out, backup)
    temporary.replace(args.h5ad_out)
    summary.update({
        'source': str(args.source), 'h5ad': str(args.h5ad_out), 'backup': str(backup) if backup else None,
        'dataset': str(args.dataset_out), 'token_dictionary': str(args.token_dictionary),
        'token_dictionary_url': 'https://github.com/NVIDIA/maxToki/blob/main/resources/token_dictionary_v1.json',
        'token_dictionary_sha256': sha256(args.token_dictionary), 'gene_median_sha256': sha256(args.gene_median),
        'vocab_size': len(tokenizer.token_dict), 'bos_id': tokenizer.bos_id, 'eos_id': tokenizer.eos_id,
        'donors': {str(k): int(v) for k, v in reloaded.obs['sample'].value_counts().items()},
        'length_min': min(map(len, sequences)), 'length_max': max(map(len, sequences)),
        'reloaded_and_verified': True,
    })
    (args.dataset_out / 'rebuild_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
