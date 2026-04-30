"""Build a HuggingFace ``datasets.Dataset`` of TimeBetweenCells records for
the BioNeMo ``MaxTokiDataModule.predict_dataset_path``.

Per cell c we emit two records (in the same row order, so they join trivially):

    baseline:  [<bos>, c, <eos>, <boq>, c,             <eoq>, dummy_numeric]
    perturbed: [<bos>, c, <eos>, <boq>, perturb(c, g), <eoq>, dummy_numeric]

The trailing ``dummy_numeric`` is required because the upstream
``MaxTokiTokenizer.collate_batch_multitask`` calls ``determine_task_type``
which needs a numeric token after ``<eoq>`` to classify the row as
TimeBetweenCells. Its value never enters the prediction - the headless
predict step gathers the model's regression output at the ``<eoq>``
position only.

The model's predicted timestep at ``<eoq>`` is interpreted as:
    "how much time elapsed between the past cell (in <bos>...<eos>)
     and the query cell (in <boq>...<eoq>)?"

For the baseline (cell vs. itself) the model should predict ~0 timesteps.
For the perturbed (cell vs. inhibited cell) it predicts the model's
estimate of how much aging the perturbation induced - this is the
zero-shot signal we score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .perturbation import Direction, perturb_tokens
from .tokenizer import CellTokenizer, MODEL_INPUT_SIZE, pick_dummy_numeric_token


def _resolve_ensembl_ids(adata) -> np.ndarray:
    var = adata.var
    if "ensembl_id" in var.columns:
        return var["ensembl_id"].values
    if "feature_id" in var.columns:
        return var["feature_id"].values
    names = np.asarray(adata.var_names.values)
    if names.size and isinstance(names[0], str) and names[0].startswith("ENSG"):
        return names
    raise ValueError(
        "Need var['ensembl_id'] / var['feature_id'] / ENSG-prefixed var_names."
    )


def _row_counts(adata, i: int) -> np.ndarray:
    import scipy.sparse as sp

    src = adata.raw.X if adata.raw is not None else adata.X
    row = src[i]
    if sp.issparse(row):
        row = np.asarray(row.todense()).ravel()
    else:
        row = np.asarray(row).ravel()
    return row


def _tokenize_cells(
    h5ad_path: str,
    tokenizer: CellTokenizer,
    rve_max_len: int,
) -> Iterator[tuple[str, list[int]]]:
    """Stream (cell_id, rve_tokens) for each cell.

    ``rve_tokens`` is the rank-value gene token list including BOS/EOS, but
    truncated so two copies + 4 special tokens (<boq>, <eoq>, +2 wrappers
    aren't both needed since the second copy reuses the same tokens) fit in
    the model context."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    ensembl_ids = _resolve_ensembl_ids(adata)
    cell_ids = adata.obs_names.astype(str).values
    for i in range(adata.n_obs):
        counts = _row_counts(adata, i)
        n_counts_val = (
            float(adata.obs["n_counts"].iloc[i])
            if "n_counts" in adata.obs.columns
            else None
        )
        toks = tokenizer.tokenize_expression(
            ensembl_ids, counts, n_counts=n_counts_val, max_len=rve_max_len
        )
        yield str(cell_ids[i]), toks


def _build_record(
    cell_tokens: list[int],
    query_tokens: list[int],
    boq_id: int,
    eoq_id: int,
    dummy_numeric: int,
    cell_id: str,
    gene_token: int,
    direction: Direction,
    is_baseline: bool,
) -> dict:
    # cell_tokens already includes <bos>...<eos>; query_tokens too. We strip
    # the query's BOS/EOS (it's wrapped by <boq>/<eoq> instead).
    bos = cell_tokens[0]
    eos = cell_tokens[-1]
    query_genes = query_tokens[1:-1]
    input_ids = (
        list(cell_tokens)               # [<bos>, ranked_genes, <eos>]
        + [boq_id]                      # <boq>
        + query_genes                   # ranked_genes' (= same or perturbed)
        + [eoq_id]                      # <eoq>
        + [dummy_numeric]               # dummy regression-label slot
    )
    # Sanity unused vars - kept explicit so future readers see grammar
    del bos, eos
    return {
        "input_ids": input_ids,
        "cell_id": cell_id,
        "gene_token": int(gene_token),
        "direction": direction,
        "condition": "baseline" if is_baseline else "perturbed",
        "n_query_tokens": len(query_genes),
    }


def build_paired_dataset(
    h5ad_path: str | Path,
    tokenizer: CellTokenizer,
    gene_token: int,
    direction: Direction = "inhibit",
    out_dir_baseline: str | Path = "./out/baseline.dataset",
    out_dir_perturbed: str | Path = "./out/perturbed.dataset",
    seq_length: int = 16384,
) -> tuple[Path, Path, dict]:
    """Tokenize all cells and write two HF datasets (baseline, perturbed).

    Records in both datasets are emitted in the same order, so the i-th row
    of one corresponds to the i-th row of the other.
    """
    import datasets

    if not tokenizer.has_temporal_tokens:
        raise RuntimeError(
            "tokenizer is missing <boq>/<eoq>/numeric tokens - point "
            "MAXTOKI_TOKEN_DICT at the full BioNeMo dictionary."
        )
    boq_id = tokenizer.boq_id
    eoq_id = tokenizer.eoq_id
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    dummy_numeric = pick_dummy_numeric_token(tokenizer)

    # Each record uses 2x cell_tokens + 4 special tokens. Cap the per-cell
    # rank-value length so that final input_ids <= seq_length.
    # final_len = 2 * len(rve_tokens_including_bos_eos) - 2 + 3  (one BOS+EOS
    # is shared between context and query payload, plus boq+eoq+dummy = 3)
    # Solve for max RVE genes:
    max_rve_len = (seq_length - 3 + 2) // 2
    if max_rve_len < 4:
        raise ValueError(f"seq_length={seq_length} too small for paired records")
    rve_max_len = min(max_rve_len, MODEL_INPUT_SIZE)

    base_records: list[dict] = []
    pert_records: list[dict] = []
    for cell_id, cell_tokens in _tokenize_cells(h5ad_path, tokenizer, rve_max_len):
        perturbed_tokens = perturb_tokens(
            cell_tokens, gene_token, direction, bos_id, eos_id
        )
        base_records.append(
            _build_record(
                cell_tokens, cell_tokens, boq_id, eoq_id, dummy_numeric,
                cell_id, gene_token, direction, is_baseline=True,
            )
        )
        pert_records.append(
            _build_record(
                cell_tokens, perturbed_tokens, boq_id, eoq_id, dummy_numeric,
                cell_id, gene_token, direction, is_baseline=False,
            )
        )

    out_dir_baseline = Path(out_dir_baseline)
    out_dir_perturbed = Path(out_dir_perturbed)
    out_dir_baseline.parent.mkdir(parents=True, exist_ok=True)
    out_dir_perturbed.parent.mkdir(parents=True, exist_ok=True)

    base_ds = datasets.Dataset.from_list(base_records)
    pert_ds = datasets.Dataset.from_list(pert_records)
    base_ds.save_to_disk(str(out_dir_baseline))
    pert_ds.save_to_disk(str(out_dir_perturbed))

    summary = {
        "n_cells": len(base_records),
        "seq_length": seq_length,
        "rve_max_len": rve_max_len,
        "max_input_len": max(len(r["input_ids"]) for r in base_records),
        "gene_token": int(gene_token),
        "direction": direction,
        "boq_id": boq_id,
        "eoq_id": eoq_id,
        "dummy_numeric_token_id": dummy_numeric,
    }
    return out_dir_baseline, out_dir_perturbed, summary
